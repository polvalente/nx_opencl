defmodule NxCL.Executor.Plan do
  @moduledoc false
  # Cuts a program that holds `cond`, `while` or `host_call` into steps for `NxCL.Executor`.
  #
  # A scope (the program, or one sub-program) becomes a list of steps:
  #
  #   * `:segment` - a run of straight-line instructions, sent to the native
  #     executor as its own small program. Its inputs are the values it reads
  #     from the scope's table; its outputs are the results a later step reads.
  #   * `:cond` and `:while` - run by the executor, which reads the predicate
  #     and runs the sub-scopes it needs.
  #   * `:host_call` - a callback. The executor reads the operands back, calls
  #     the function and uploads the results. The step holds only the index of
  #     its entry in the program's side table, so cached plans never hold a function.
  #     A host call is never dropped for unused results: it always runs.
  #
  # Each step lists the table keys to drop once its last reader has run.
  # Captures and constants stay in the table until the run ends, because
  # sub-programs share them.

  alias NxCL.Executor
  alias NxCL.Program
  alias NxCL.Program.Instruction

  @control [:cond, :while, :host_call]

  defstruct [:scope, :const_keys, :const_wire]

  @type t :: %__MODULE__{}

  @doc "Plans `program`. Compiles nothing; segments compile on first run."
  @spec build(Program.t()) :: t()
  def build(%Program{} = program) do
    {scope, needed} =
      scope(program.instructions, program.outputs, base_types(program), program.constants)

    keys = Enum.sort(needed)
    const_wire = Enum.map(keys, fn {:const, c} -> const_slot(program.constants, c) end)
    outs = for k <- 0..(length(keys) - 1)//1, do: {:const, k}

    %__MODULE__{
      scope: scope,
      const_keys: keys,
      const_wire: {[], [], const_wire, [], outs}
    }
  end

  @doc "True when the program holds a control or host-call instruction."
  @spec control?(Program.t()) :: boolean()
  def control?(%Program{instructions: instructions}),
    do: Enum.any?(instructions, &(&1.op in @control))

  defp base_types(program) do
    inputs =
      for {{type, shape}, i} <- Enum.with_index(program.inputs),
          into: %{},
          do: {{:input, i}, {type, shape}}

    captures =
      for {t, i} <- Enum.with_index(program.captures),
          into: %{},
          do: {{:capture, i}, {t.type, t.shape}}

    consts =
      for {{_value, type}, i} <- Enum.with_index(program.constants),
          into: %{},
          do: {{:const, i}, {type, {}}}

    inputs |> Map.merge(captures) |> Map.merge(consts)
  end

  defp const_slot(constants, c), do: Executor.encode_const(Enum.at(constants, c))

  # Returns `{scope, needed_consts}`. A const is needed as a buffer when a
  # control step or a scope output reads it; segments hold their own consts.
  defp scope(instructions, outputs, types, constants) do
    indexed = Enum.with_index(instructions)
    types = Enum.reduce(indexed, types, &result_types/2)

    {raw, needed} =
      indexed
      |> Enum.chunk_by(fn {instr, _i} -> instr.op in @control end)
      |> Enum.flat_map_reduce(const_refs(outputs), fn chunk, needed ->
        step_chunk(chunk, types, constants, needed)
      end)

    {steps, _live} =
      raw
      |> Enum.reverse()
      |> Enum.map_reduce(MapSet.new(outputs), fn step, live ->
        live_defs = Enum.filter(step.defs, &MapSet.member?(live, &1))

        free =
          step.reads
          |> Enum.uniq()
          |> Enum.filter(&(elem(&1, 0) in [:input, :carry, :result]))
          |> Enum.reject(&MapSet.member?(live, &1))

        {finish(step, live_defs, free, types, constants),
         MapSet.union(live, MapSet.new(step.reads))}
      end)

    scope = %{
      steps: Enum.reverse(steps),
      outputs: outputs,
      out_types: Enum.map(outputs, &(types |> Map.fetch!(&1) |> elem(0)))
    }

    {scope, needed}
  end

  defp result_types({%Instruction{op: op} = instr, i}, types) when op in @control do
    instr.out_type
    |> Enum.zip(instr.out_shape)
    |> Enum.with_index()
    |> Enum.reduce(types, fn {ts, k}, acc -> Map.put(acc, {:result, i, k}, ts) end)
  end

  defp result_types({instr, i}, types),
    do: Map.put(types, {:result, i}, {instr.out_type, instr.out_shape})

  defp const_refs(refs), do: refs |> Enum.filter(&match?({:const, _}, &1)) |> MapSet.new()

  # -- Raw steps ---------------------------------------------------------------

  defp step_chunk([{%Instruction{op: op}, _} | _] = chunk, types, constants, needed)
       when op in @control do
    Enum.map_reduce(chunk, needed, fn {instr, i}, needed ->
      control_step(instr, i, types, constants, needed)
    end)
  end

  defp step_chunk(chunk, _types, _constants, needed) do
    [{_, start} | _] = chunk
    results = for {_, i} <- chunk, do: i
    local = MapSet.new(results)

    operands = Enum.flat_map(chunk, fn {instr, _} -> instr.operands end)
    ext = operands |> Enum.reject(&external_free?(&1, local)) |> Enum.uniq()
    consts = operands |> Enum.filter(&match?({:const, _}, &1)) |> Enum.uniq()

    step = %{
      kind: :segment,
      chunk: chunk,
      start: start,
      ext: ext,
      used_consts: consts,
      reads: ext,
      defs: Enum.map(results, &{:result, &1})
    }

    {[step], needed}
  end

  # A const or a result of the same segment needs no input slot.
  defp external_free?({:const, _}, _local), do: true
  defp external_free?({:result, m}, local), do: MapSet.member?(local, m)
  defp external_free?(_ref, _local), do: false

  defp control_step(instr, i, types, constants, needed) do
    shared = Map.filter(types, fn {key, _} -> elem(key, 0) in [:capture, :const] end)

    carry =
      instr.operands
      |> Enum.with_index()
      |> Map.new(fn {ref, k} -> {{:carry, k}, Map.fetch!(types, ref)} end)

    sub_types = Map.merge(shared, carry)

    {subs, needed} =
      Enum.map_reduce(instr.subprograms, MapSet.union(needed, const_refs(instr.operands)), fn sub,
                                                                                              acc ->
        {scope, inner} = scope(sub.instructions, sub.outputs, sub_types, constants)
        {scope, MapSet.union(acc, inner)}
      end)

    step = %{
      kind: instr.op,
      instr: instr,
      subs: subs,
      reads: instr.operands,
      defs: for(k <- 0..(length(instr.out_type) - 1)//1, do: {:result, i, k})
    }

    {step, needed}
  end

  # -- Final steps -------------------------------------------------------------

  defp finish(%{kind: :segment} = step, live_defs, free, types, constants) do
    local = MapSet.new(for {_, i} <- step.chunk, do: i)
    ext_pos = step.ext |> Enum.with_index() |> Map.new()
    const_pos = step.used_consts |> Enum.with_index() |> Map.new()

    map = fn
      {:const, _} = ref ->
        {:const, Map.fetch!(const_pos, ref)}

      {:result, m} = ref ->
        if MapSet.member?(local, m),
          do: {:result, m - step.start},
          else: {:input, Map.fetch!(ext_pos, ref)}

      ref ->
        {:input, Map.fetch!(ext_pos, ref)}
    end

    instrs =
      for {instr, _} <- step.chunk,
          do: Executor.encode_instruction(%{instr | operands: Enum.map(instr.operands, map)})

    outs = Enum.map(live_defs, fn {:result, m} -> {:result, m - step.start} end)

    wire =
      {Enum.map(step.ext, &Map.fetch!(types, &1)), [],
       Enum.map(step.used_consts, fn {:const, c} -> const_slot(constants, c) end), instrs, outs}

    %{
      kind: :segment,
      key: {Executor, :segment, :erlang.md5(:erlang.term_to_binary(wire))},
      wire: wire,
      ins: step.ext,
      n_consts: length(step.used_consts),
      outs: live_defs,
      free: free
    }
  end

  defp finish(%{kind: :cond} = step, live_defs, free, _types, _constants) do
    clauses =
      step.subs
      |> Enum.drop(-1)
      |> Enum.chunk_every(2)
      |> Enum.map(fn [pred, body] -> {pred, body} end)

    %{
      kind: :cond,
      operands: step.reads,
      clauses: clauses,
      else: List.last(step.subs),
      live: live_pairs(step.defs, live_defs),
      free: free
    }
  end

  defp finish(%{kind: :while} = step, live_defs, free, _types, _constants) do
    [cond_scope, body_scope] = step.subs

    static =
      case step.instr.attrs do
        [:static, count] -> count
        _ -> nil
      end

    %{
      kind: :while,
      operands: step.reads,
      cond: cond_scope,
      body: body_scope,
      static: static,
      live: live_pairs(step.defs, live_defs),
      free: free
    }
  end

  defp finish(%{kind: :host_call} = step, live_defs, free, _types, _constants) do
    [index] = step.instr.attrs

    %{
      kind: :host_call,
      operands: step.reads,
      index: index,
      live: live_pairs(step.defs, live_defs),
      free: free
    }
  end

  defp live_pairs(defs, live_defs) do
    defs
    |> Enum.with_index()
    |> Enum.filter(fn {key, _pos} -> key in live_defs end)
    |> Enum.map(fn {key, pos} -> {pos, key} end)
  end
end

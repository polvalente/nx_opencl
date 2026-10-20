defmodule NxCL.Compiler.Executor.Plan do
  @moduledoc false
  # Cuts a program with `cond`, `while` or `host_call` into steps for
  # `NxCL.Compiler.Executor`. A scope (the program or a nested body) becomes:
  #
  #   * `:segment` - straight-line instructions, run natively as a small program
  #   * `:cond`, `:while` - run by the executor, which reads the predicate
  #   * `:host_call` - holds only an index into `:host_calls`, so a cached plan
  #     never holds a function. It always runs, even with unused results.
  #
  # Each step lists the table keys to drop after its last reader. Captures and
  # constants stay until the run ends, because nested bodies share them.
  #
  # Segments share the bounded `NxCL.Compiler.ProgramCache`; an evicted segment
  # compiles again on its next run. A plan holds no device state, so
  # `ProgramCache` keys it by the program signature alone; the compiled segments
  # are keyed by device too.
  #
  # `scope/4` builds the steps in two passes, because a step can only drop a key
  # once it knows whether anything later reads it:
  #
  #   1. Forward: cut the instruction list into chunks and make one raw step per
  #      chunk. A raw step records what it reads and what it defines, and also
  #      collects the constants that control steps and outputs read as buffers.
  #   2. Backward: walk the raw steps from the last one with the set of keys
  #      still live. A result is live if a later step or an output reads it. A
  #      key a step reads that is not live afterwards goes in its `:free` list.
  #      `finish/5` then turns the raw step into its final shape, with the wire
  #      of a segment, once it knows which results to keep.

  alias NxCL.Compiler.Executor.Wire
  alias NxCL.Compiler.Instruction
  alias NxCL.Compiler.Program

  @control [:cond, :while, :host_call]

  defstruct [:scope, :const_keys, :const_wire, :const_key]

  @typedoc "A key of the buffer table: an input, capture, constant, result or carry ref."
  @type key :: Instruction.ref()

  @typedoc "`{position, key}` of each result a control step leaves live."
  @type live :: [{non_neg_integer(), key()}]

  @type scope :: %{steps: [step()], outputs: [key()], out_types: [Nx.Type.t()]}

  @type step ::
          %{
            kind: :segment,
            key: {:segment, binary()},
            wire: Wire.t(),
            ins: [key()],
            n_consts: non_neg_integer(),
            outs: [key()],
            free: [key()]
          }
          | %{
              kind: :cond,
              operands: [key()],
              clauses: [{scope(), scope()}],
              else: scope(),
              live: live(),
              free: [key()]
            }
          | %{
              kind: :while,
              operands: [key()],
              cond: scope(),
              body: scope(),
              static: non_neg_integer() | nil,
              live: live(),
              free: [key()]
            }
          | %{
              kind: :host_call,
              operands: [key()],
              index: non_neg_integer(),
              live: live(),
              free: [key()]
            }

  @type t :: %__MODULE__{
          scope: scope(),
          const_keys: [key()],
          const_wire: Wire.t(),
          const_key: {:consts, binary()}
        }

  @doc "Plans `program`. Compiles nothing; segments compile on first run."
  @spec build(Program.t()) :: t()
  def build(%Program{} = program) do
    {scope, needed} =
      scope(program.instructions, program.outputs, base_types(program), program.constants)

    keys = Enum.sort(needed)
    const_wire = Enum.map(keys, fn {:const, c} -> const_slot(program.constants, c) end)
    outs = for k <- 0..(length(keys) - 1)//1, do: {:const, k}
    wire = Wire.build([], [], const_wire, [], outs)

    %__MODULE__{
      scope: scope,
      const_keys: keys,
      const_wire: wire,
      const_key: {:consts, :erlang.md5(:erlang.term_to_binary(wire))}
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

  defp const_slot(constants, c), do: Wire.const(Enum.at(constants, c))

  # Returns `{scope, needed_consts}`: constants a control step or output reads as
  # a buffer. Segments hold their own constants.
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
    ext = operands |> Enum.reject(&internal?(&1, local)) |> Enum.uniq()
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

  # True for a ref a segment needs no outside buffer for: a constant (the
  # segment holds its own) or a result of the same segment.
  defp internal?({:const, _}, _local), do: true
  defp internal?({:result, m}, local), do: MapSet.member?(local, m)
  defp internal?(_ref, _local), do: false

  defp control_step(instr, i, types, constants, needed) do
    shared = Map.filter(types, fn {key, _} -> elem(key, 0) in [:capture, :const] end)

    carry =
      instr.operands
      |> Enum.with_index()
      |> Map.new(fn {ref, k} -> {{:carry, k}, Map.fetch!(types, ref)} end)

    sub_types = Map.merge(shared, carry)

    {subs, needed} =
      Enum.map_reduce(
        bodies(instr),
        MapSet.union(needed, const_refs(instr.operands)),
        fn {body_instrs, outs}, acc ->
          {scope, inner} = scope(body_instrs, outs, sub_types, constants)
          {scope, MapSet.union(acc, inner)}
        end
      )

    step = %{
      kind: instr.op,
      instr: instr,
      subs: subs,
      reads: instr.operands,
      defs: for(k <- 0..(length(instr.out_type) - 1)//1, do: {:result, i, k})
    }

    {step, needed}
  end

  # A `while` has `[cond, body]`; a `cond` has `[pred_1, body_1, ..., else]`.
  defp bodies(%Instruction{op: :while, attrs: attrs}),
    do: [Keyword.fetch!(attrs, :cond), Keyword.fetch!(attrs, :body)]

  defp bodies(%Instruction{op: :cond, attrs: attrs}) do
    Enum.flat_map(Keyword.fetch!(attrs, :clauses), fn {pred, body} -> [pred, body] end) ++
      [Keyword.fetch!(attrs, :else)]
  end

  defp bodies(%Instruction{op: :host_call}), do: []

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
          do: Wire.instruction(%{instr | operands: Enum.map(instr.operands, map)})

    outs = Enum.map(live_defs, fn {:result, m} -> {:result, m - step.start} end)

    wire =
      Wire.build(
        Enum.map(step.ext, &Map.fetch!(types, &1)),
        [],
        Enum.map(step.used_consts, fn {:const, c} -> const_slot(constants, c) end),
        instrs,
        outs
      )

    %{
      kind: :segment,
      key: {:segment, :erlang.md5(:erlang.term_to_binary(wire))},
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

    static = Keyword.get(step.instr.attrs, :static)

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

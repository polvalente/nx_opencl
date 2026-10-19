defmodule NxCL.Compiler.Lower.HostCall do
  @moduledoc """
  Lowers `io_call` and `runtime_call` nodes to one `:host_call` instruction each.

  The operands are the leaves of the node's input container, in flatten order.
  The instruction has list-valued `:out_type` and `:out_shape`, one per output
  leaf, and its results are `{:result, i, k}`, as for `cond` and `while`. Its one
  attr is an index into `NxCL.Compiler.Program`'s `:host_calls` side table, whose entries
  `NxCL.Compiler.Executor.Host` reads at run time:

    * `:kind` - `:io_call` (side effect; the outputs are the operands, unchanged)
      or `:runtime_call` (the function returns the outputs).
    * `:fun` - the callback, or `nil` for a named `io_call` with no callback.
    * `:opts` - the options a `runtime_call` passes to its function.
    * `:in_template` - the input container with template tensors.
    * `:out_template` - the output container with template tensors.

  """

  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias NxCL.Compiler.Instruction
  alias NxCL.Compiler.Lower
  alias NxCL.Compiler.Lower.Shared
  alias NxCL.Compiler.Program

  @doc "Lowers an `io_call` or `runtime_call` node."
  @spec node(Lower.state(), Shared.tensor()) :: Lower.state()
  def node(state, %T{data: %Nx.Defn.Expr{op: :io_call, args: [input, spec, template, _ref]}} = n) do
    entry = %{
      kind: :io_call,
      fun: callback(spec),
      opts: [],
      in_template: Composite.traverse(input, &Nx.to_template/1),
      out_template: template
    }

    emit(state, n, input, entry)
  end

  def node(
        state,
        %T{data: %Nx.Defn.Expr{op: :runtime_call, args: [input, fun, template, opts]}} = n
      ) do
    entry = %{
      kind: :runtime_call,
      fun: fun,
      opts: opts,
      in_template: Composite.traverse(input, &Nx.to_template/1),
      out_template: template
    }

    emit(state, n, input, entry)
  end

  # Emits a `:host_call` for `entry` reading the leaves of `input`, and binds the
  # node's id to its result. A tuple-typed node maps to `{:tuple, refs}`.
  @spec emit(Lower.state(), Shared.tensor(), Nx.Container.t(), Program.host_call()) ::
          Lower.state()
  defp emit(state, %T{data: %Nx.Defn.Expr{id: id}} = n, input, entry) do
    operands =
      [input] |> Composite.flatten_list() |> Enum.map(&Map.fetch!(state.refs, &1.data.id))

    {instr, refs} = instruction(entry, operands, length(state.host_calls), length(state.instrs))
    ref = Shared.node_ref(n, refs)

    %{
      state
      | instrs: [instr | state.instrs],
        host_calls: [entry | state.host_calls],
        refs: Map.put(state.refs, id, ref)
    }
  end

  # The `:host_call` instruction for `entry` over `operands`, and its result refs.
  # `index` is the entry's place in the program's `:host_calls`; `at` is the
  # instruction's place in the instruction list.
  @spec instruction(
          Program.host_call(),
          [Instruction.ref()],
          non_neg_integer(),
          non_neg_integer()
        ) ::
          {Instruction.t(), [Instruction.ref()]}
  defp instruction(entry, operands, index, at) do
    leaves = Composite.flatten_list([entry.out_template])

    instr = %Instruction{
      op: :host_call,
      operands: operands,
      out_type: Enum.map(leaves, & &1.type),
      out_shape: Enum.map(leaves, & &1.shape),
      attrs: [index]
    }

    {instr, Shared.result_refs(at, length(leaves))}
  end

  # A named `io_call` runs its own callback (`nil` for none); the `:hooks` option that overrides it is rejected.
  defp callback({:fn, fun}), do: fun
  defp callback({:named, _name, fun}) when is_function(fun) or is_nil(fun), do: fun

  defp callback(spec),
    do: raise(ArgumentError, "unsupported io_call callback spec: #{inspect(spec)}")
end

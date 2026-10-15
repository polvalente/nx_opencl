defmodule NxCL.Lower.HostCall do
  @moduledoc """
  Lowers `io_call` and `runtime_call` nodes to one `:host_call` instruction each.

  The operands are the leaves of the node's input container, in flatten order.
  The instruction has list-valued `:out_type` and `:out_shape`, one per output
  leaf, and its results are `{:result, i, k}`, as for `cond` and `while`. Its one
  attr is an index into `NxCL.Program`'s `:host_calls` side table, whose entries
  `NxCL.Executor.Host` reads at run time:

    * `:kind` - `:io_call` (side effect; the outputs are the operands, unchanged)
      or `:runtime_call` (the function returns the outputs).
    * `:fun` - the callback, or `nil` for a named `io_call` with no callback.
    * `:opts` - the options a `runtime_call` passes to its function.
    * `:in_template` - the input container with template tensors.
    * `:out_template` - the output container with template tensors.

  `emit/4` is also the entry for a later stage that runs an op on the host
  because no device kernel exists.
  """

  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias NxCL.Program.Instruction

  @doc "Lowers an `io_call` or `runtime_call` node."
  @spec node(map(), map()) :: map()
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

  @doc """
  Emits a `:host_call` for `entry` reading the leaves of `input`, and binds the
  node's id to its result. A tuple-typed node maps to `{:tuple, refs}`.
  """
  @spec emit(map(), map(), Nx.Container.t(), map()) :: map()
  def emit(state, %T{data: %Nx.Defn.Expr{id: id}} = n, input, entry) do
    operands =
      [input] |> Composite.flatten_list() |> Enum.map(&Map.fetch!(state.refs, &1.data.id))

    leaves = Composite.flatten_list([entry.out_template])
    index = length(state.host_calls)

    instr = %Instruction{
      op: :host_call,
      operands: operands,
      out_type: Enum.map(leaves, & &1.type),
      out_shape: Enum.map(leaves, & &1.shape),
      attrs: [index]
    }

    at = length(state.instrs)
    refs = for k <- 0..(length(leaves) - 1)//1, do: {:result, at, k}

    ref =
      case n.type do
        {:tuple, _} -> {:tuple, refs}
        _ -> hd(refs)
      end

    %{
      state
      | instrs: [instr | state.instrs],
        host_calls: [entry | state.host_calls],
        refs: Map.put(state.refs, id, ref)
    }
  end

  # A named `io_call` runs its callback; the `:hooks` option that overrides it is rejected.
  defp callback({:fn, fun}), do: fun
  defp callback({:named, _name, fun}), do: fun
end

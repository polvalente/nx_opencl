defmodule NxCL.Lower.Block do
  @moduledoc """
  The block hook: decides how each `Nx.Block` node lowers.

  Every block struct of the pinned Nx has one entry in `decisions/0`:

    * `:native` - `NxCL.Lower` has a clause that emits instructions for the
      struct itself.
    * `:default` - the block's `default_expr` lowers inline, in the parent
      program. Its parameters map to the refs of the block's arguments.
    * `{:host, reason}` - the block runs on the host as one `:host_call`
      instruction, with `Nx.BinaryBackend` as the executor (gate G5).

  A struct outside the table (a user block) lowers through its `default_expr`.
  `NxCL.Lower.BlockTest` fails when a struct of `Nx.Block.*` has no entry.
  """

  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias NxCL.Defn.Tree
  alias NxCL.Lower.Control
  alias NxCL.Lower.HostCall

  @host_reason "No device kernel; runs through host_call on Nx.BinaryBackend (G5)."

  @decisions %{
    Nx.Block.LogicalNot => :native,
    Nx.Block.Phase => :native,
    Nx.Block.CumulativeSum => :native,
    Nx.Block.CumulativeProduct => :native,
    Nx.Block.CumulativeMin => :native,
    Nx.Block.CumulativeMax => :native,
    Nx.Block.LinAlg.Cholesky => :native,
    Nx.Block.LinAlg.Solve => :default,
    Nx.Block.LinAlg.Determinant => :default,
    Nx.Block.AllClose => :default,
    Nx.Block.Take => :default,
    Nx.Block.TakeAlongAxis => :default,
    Nx.Block.TopK => :default,
    Nx.Block.FFT2 => :default,
    Nx.Block.IFFT2 => :default,
    Nx.Block.RFFT => :default,
    Nx.Block.IRFFT => :default,
    Nx.Block.LinAlg.QR => {:host, @host_reason},
    Nx.Block.LinAlg.Eigh => {:host, @host_reason},
    Nx.Block.LinAlg.SVD => {:host, @host_reason},
    Nx.Block.LinAlg.LU => {:host, @host_reason}
  }

  @doc "The decision table, keyed by block struct module."
  @spec decisions() :: %{module() => :native | :default | {:host, String.t()}}
  def decisions(), do: @decisions

  @doc "The decision for a block struct. A struct with no entry lowers its default."
  @spec decision(struct()) :: :native | :default | {:host, String.t()}
  def decision(%{__struct__: mod}), do: Map.get(@decisions, mod, :default)

  @doc """
  Lowers a block node that has no native clause. `expand` is
  `(node, state) -> state` for one graph node.
  """
  @spec lower(map(), map(), (map(), map() -> map())) :: map()
  def lower(state, %T{data: %Nx.Defn.Expr{args: [struct, args, default, fun]}} = node, expand) do
    case decision(struct) do
      :default ->
        inline(state, node, {args, default}, expand)

      {:host, _reason} ->
        host(state, node, {struct, args, default, fun})

      :native ->
        raise ArgumentError,
              "does not yet lower op #{inspect(struct.__struct__)} (a native block with no clause)"
    end
  end

  @doc """
  Lowers `node`, a block with `{args, default}`, through its default expression.

  The default is a graph over inner parameters, one per block argument. This
  walks it in post-order, points each parameter at the ref of the matching
  argument, and lowers the other nodes with `expand`. The block node gets the
  ref of the default's result; a tuple result gets `{:tuple, refs}`.
  """
  @spec inline(map(), map(), {[map()], map() | tuple()}, (map(), map() -> map())) :: map()
  def inline(state, %T{data: %Nx.Defn.Expr{id: id}}, {args, default}, expand) do
    # Nodes of the default belong to this scope for the `cond` dependency walk.
    state = %{state | avail: Map.merge(state.avail, Nx.Defn.Tree.scope_ids(default))}

    state =
      default
      |> Tree.post_order(&Control.scope_deps(&1, state))
      |> Enum.reduce(state, fn
        %T{data: %Nx.Defn.Expr{id: param, op: :parameter, args: [pos]}}, state ->
          arg = Enum.at(args, pos)
          put_in(state.refs[param], Map.fetch!(state.refs, arg.data.id))

        node, state ->
          expand.(node, state)
      end)

    put_in(state.refs[id], result_ref(state, default))
  end

  defp result_ref(state, %T{data: %Nx.Defn.Expr{id: id}}), do: Map.fetch!(state.refs, id)

  defp result_ref(state, tuple) when is_tuple(tuple),
    do: {:tuple, tuple |> Tuple.to_list() |> Enum.map(&result_ref(state, &1))}

  # One `:host_call`: the block's own function runs on `Nx.BinaryBackend` tensors.
  defp host(state, node, {struct, args, default, fun}) do
    tensors = Enum.filter(args, &match?(%T{}, &1))
    entry = host_entry(struct, tensors, Composite.traverse(default, &Nx.to_template/1), fun)
    HostCall.emit(state, node, List.to_tuple(tensors), entry)
  end

  @doc """
  The `:host_calls` entry for a host block: `tensors` are the block's tensor
  arguments, `out_template` the output container, `fun` the block's default.
  """
  @spec host_entry(struct(), [T.t()], Nx.Container.t(), function()) :: map()
  def host_entry(struct, tensors, out_template, fun) do
    %{
      kind: :runtime_call,
      fun: fn input, _opts -> run_on_host(fun, struct, input) end,
      opts: [],
      in_template: List.to_tuple(Enum.map(tensors, &Nx.to_template/1)),
      out_template: out_template
    }
  end

  # Runs the default expression under the evaluator on `Nx.BinaryBackend`, whatever
  # the caller's default compiler or backend is, so the host copy never re-enters
  # this compiler or the device backend.
  defp run_on_host(fun, struct, input) do
    args = input |> Tuple.to_list() |> Enum.map(&Nx.backend_copy(&1, Nx.BinaryBackend))

    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      Nx.Defn.jit(
        fn args -> apply(fun, [struct | Tuple.to_list(args)]) end,
        compiler: Nx.Defn.Evaluator
      ).(List.to_tuple(args))
    end)
  end
end

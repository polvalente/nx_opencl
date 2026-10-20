defmodule NxCL.Compiler.Lower.Block do
  @moduledoc false
  # Decides how each `Nx.Block` node lowers: `:native` (a clause in
  # `NxCL.Compiler.Lower`) or `:default` (its `default_expr`, inline). A struct
  # outside the table lowers through its default.

  alias Nx.Tensor, as: T
  alias NxCL.Compiler.Lower
  alias NxCL.Compiler.Lower.Control
  alias NxCL.Compiler.Lower.Shared
  alias NxCL.Compiler.Tree

  @decisions %{
    Nx.Block.LogicalNot => :native,
    Nx.Block.Phase => :native,
    Nx.Block.CumulativeSum => :native,
    Nx.Block.CumulativeProduct => :native,
    Nx.Block.CumulativeMin => :native,
    Nx.Block.CumulativeMax => :native,
    Nx.Block.LinAlg.Cholesky => :native,
    Nx.Block.LinAlg.Solve => :default,
    Nx.Block.LinAlg.QR => :default,
    Nx.Block.LinAlg.Eigh => :default,
    Nx.Block.LinAlg.SVD => :default,
    Nx.Block.LinAlg.LU => :default,
    Nx.Block.LinAlg.Determinant => :default,
    Nx.Block.AllClose => :default,
    Nx.Block.Take => :default,
    Nx.Block.TakeAlongAxis => :default,
    Nx.Block.TopK => :default,
    Nx.Block.FFT2 => :default,
    Nx.Block.IFFT2 => :default,
    Nx.Block.RFFT => :default,
    Nx.Block.IRFFT => :default
  }

  @doc "The decision for a block struct. A struct with no entry lowers its default."
  @spec decision(struct()) :: :native | :default
  def decision(%{__struct__: mod}), do: Map.get(@decisions, mod, :default)

  @doc """
  Lowers a block node that has no native clause. `expand` lowers one graph node.
  """
  @spec lower(Lower.state(), Shared.tensor(), (Shared.tensor(), Lower.state() -> Lower.state())) ::
          Lower.state()
  def lower(state, %T{data: %Nx.Defn.Expr{args: [struct, args, default, _fun]}} = node, expand) do
    case decision(struct) do
      :default ->
        inline(state, node, {args, default}, expand)

      :native ->
        raise ArgumentError,
              "does not yet lower op #{inspect(struct.__struct__)} (a native block with no clause)"
    end
  end

  # Lowers a block through its default expression. Each parameter of the default
  # takes the ref of the matching argument; `expand` lowers the other nodes. A
  # tuple result gets `{:tuple, refs}`.
  defp inline(state, %T{data: %Nx.Defn.Expr{id: id}}, {args, default}, expand) do
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
end

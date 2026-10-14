defmodule NxCL.Lower.Block do
  @moduledoc """
  Lowers an `Nx.Block` node through its `default_expr`, inline.

  The default expression is a graph over inner parameters, one per block
  argument. `inline/4` walks it in post-order, points each parameter at the ref
  of the matching block argument, and lowers the other nodes with the caller's
  `expand` function. The block node gets the ref of the default's result.

  Stage 12 replaces this with the general block hook and native lowering.
  """

  alias Nx.Tensor, as: T
  alias NxCL.Defn.Tree

  @doc """
  Lowers `node`, a block with `{args, default}`. `expand` is `(node, state) -> state`
  for one graph node. Only a block with one tensor result is supported.
  """
  @spec inline(map(), map(), {[map()], map()}, (map(), map() -> map())) :: map()
  def inline(state, %T{data: %Nx.Defn.Expr{id: id}}, {args, %T{} = default}, expand) do
    state =
      default
      |> Tree.post_order()
      |> Enum.reduce(state, fn
        %T{data: %Nx.Defn.Expr{id: param, op: :parameter, args: [pos]}}, state ->
          arg = Enum.at(args, pos)
          put_in(state.refs[param], Map.fetch!(state.refs, arg.data.id))

        node, state ->
          expand.(node, state)
      end)

    put_in(state.refs[id], Map.fetch!(state.refs, default.data.id))
  end
end

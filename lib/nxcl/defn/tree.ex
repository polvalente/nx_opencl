defmodule NxCL.Defn.Tree do
  @moduledoc """
  Post-order traversal of `Nx.Defn.Expr` graphs.

  Nx 1.0 ships no post-order walker (`Nx.Defn.Tree` has `apply_args/3,4`,
  `put_args/2` and `scope_ids/1,2` only), so this module ports
  the walker of the MLX compiler. It depends only on `Nx.Defn.{Tree, Composite}`.

  ## Scope model

  `while`, `fun` and `block` nodes each open an inner scope. `post_order/2`
  returns them as opaque nodes: their inner sub-graphs do not appear in the
  result. `cond` clauses share the parent scope and are walked normally.
  """

  alias Nx.Defn.Composite
  alias Nx.Defn.Tree
  alias Nx.Tensor, as: T

  @doc """
  Returns the `%Nx.Tensor{}` nodes reachable from `output` in post-order.

  `output` is any `Nx.Container`. Each node appears once, keyed by
  `node.data.id`, and after all of its same-scope operands.

  `scope_dependencies` is called as `scope_dependencies.(node)` and returns
  `{:ok, deps}` to replace the node's own operands with `deps`, or `:default`.
  """
  @spec post_order(Nx.Container.t(), (Nx.Tensor.t() -> {:ok, [Nx.Tensor.t()]} | :default)) ::
          [Nx.Tensor.t()]
  def post_order(output, scope_dependencies \\ fn _node -> :default end) do
    roots = Composite.flatten_list([output])
    {_visited, rev} = Enum.reduce(roots, {MapSet.new(), []}, &visit(&1, &2, scope_dependencies))
    Enum.reverse(rev)
  end

  defp visit(%T{data: %Nx.Defn.Expr{id: id}} = node, {visited, output}, scope_dependencies) do
    if MapSet.member?(visited, id) do
      {visited, output}
    else
      # Mark the node before recursing so shared subexpressions are visited once.
      visited = MapSet.put(visited, id)
      {visited, output} = visit_scope_deps(node, {visited, output}, scope_dependencies)
      {visited, [node | output]}
    end
  end

  # A `fun`'s args hold inner-scope parameter templates, not parent-scope deps.
  defp visit_scope_deps(%T{data: %Nx.Defn.Expr{op: :fun}}, acc, _scope_dependencies), do: acc

  defp visit_scope_deps(node, acc, scope_dependencies) do
    case scope_dependencies.(node) do
      {:ok, deps} ->
        Enum.reduce(deps, acc, &visit(&1, &2, scope_dependencies))

      :default ->
        {_, acc} =
          Tree.apply_args(node, :scope, acc, fn dep, a ->
            {dep, visit(dep, a, scope_dependencies)}
          end)

        acc
    end
  end
end

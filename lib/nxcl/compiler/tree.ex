defmodule NxCL.Compiler.Tree do
  @moduledoc false
  # Post-order traversal of `Nx.Defn.Expr` graphs. `while`, `fun` and `block`
  # nodes open an inner scope and come back as opaque nodes. `cond` clauses
  # share the parent scope and are walked.

  alias Nx.Defn.Composite
  alias Nx.Defn.Tree
  alias Nx.Tensor, as: T

  @doc """
  Returns the nodes reachable from `output` in post-order, each once (by
  `node.data.id`) and after its same-scope operands.

  `scope_dependencies.(node)` returns `{:ok, deps}` to replace the node's
  operands, or `:default`.
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

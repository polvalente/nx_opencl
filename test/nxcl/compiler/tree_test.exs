defmodule NxCL.Compiler.TreeTest do
  use ExUnit.Case, async: true
  import Nx.Defn

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T
  alias NxCL.Compiler.Tree

  defn count_up(n) do
    while n, Nx.less(n, 10) do
      Nx.add(n, 1)
    end
  end

  # Every same-scope operand in `order` sits before its consumer.
  defp assert_post_order(order) do
    index = Map.new(Enum.with_index(order), fn {t, i} -> {t.data.id, i} end)

    for {node, node_idx} <- Enum.with_index(order) do
      Nx.Defn.Tree.apply_args(node, :scope, :ok, fn dep, :ok ->
        # Inner-scope deps are not in the order; skip them.
        with {:ok, dep_idx} <- Map.fetch(index, dep.data.id) do
          assert dep_idx < node_idx
        end

        {dep, :ok}
      end)
    end
  end

  defp ops(order), do: Enum.map(order, & &1.data.op)

  test "a linear chain comes out dependency first" do
    expr = Nx.Defn.debug_expr_apply(fn x -> x |> Nx.negate() |> Nx.abs() end, [Nx.tensor(1.0)])
    order = Tree.post_order(expr)

    assert ops(order) == [:parameter, :negate, :abs]
    assert_post_order(order)
  end

  test "a shared subexpression appears once" do
    expr =
      Nx.Defn.debug_expr_apply(
        fn x ->
          y = Nx.negate(x)
          Nx.add(y, y)
        end,
        [Nx.tensor(1.0)]
      )

    assert ops(Tree.post_order(expr)) == [:parameter, :negate, :add]
  end

  test "every root of a container is collected, in output order" do
    {e1, e2} =
      Nx.Defn.debug_expr_apply(fn x -> {Nx.negate(x), Nx.abs(x)} end, [Nx.tensor(1.0)])

    assert ops(Tree.post_order({e1, e2})) == [:parameter, :negate, :abs]
    assert ops(Tree.post_order({e2, e1})) == [:parameter, :abs, :negate]
  end

  test "ids are unique on a diamond" do
    expr =
      Nx.Defn.debug_expr_apply(
        fn x, y ->
          a = Nx.add(x, y)
          c = x |> Nx.multiply(a) |> Nx.subtract(y)
          Nx.add(a, c)
        end,
        [Nx.tensor(1.0), Nx.tensor(2.0)]
      )

    order = Tree.post_order(expr)
    assert_post_order(order)
    ids = Enum.map(order, & &1.data.id)
    assert ids == Enum.uniq(ids)
  end

  test "a while node is opaque and its body does not leak" do
    expr = Nx.Defn.debug_expr_apply(&count_up/1, [Nx.tensor(0)])
    order = Tree.post_order(expr)

    assert {[%T{data: %Expr{op: :while, args: [_, _, pred, body]}}], rest} =
             Enum.split_with(order, &(&1.data.op == :while))

    assert ops(rest) == [:parameter]
    assert_post_order(order)

    assert [:parameter, :constant, :less] = ops(Tree.post_order(pred))
    assert [:constant, :parameter, :add] = ops(Tree.post_order(body))
  end

  test "scope_dependencies can redirect a node's operands" do
    expr = Nx.Defn.debug_expr_apply(fn x, y -> Nx.add(x, y) end, [Nx.tensor(1.0), Nx.tensor(2.0)])
    %T{data: %Expr{args: [left, _right]}} = expr

    order =
      Tree.post_order(expr, fn
        %T{data: %Expr{op: :add}} -> {:ok, [left]}
        _ -> :default
      end)

    assert ops(order) == [:parameter, :add]
  end
end

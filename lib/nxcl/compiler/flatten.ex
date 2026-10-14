defmodule NxCL.Compiler.Flatten do
  @moduledoc false

  alias NxCL.Compiler.FlatOp

  @unary_ops [
    :exp,
    :expm1,
    :log,
    :log1p,
    :sigmoid,
    :cos,
    :sin,
    :tan,
    :cosh,
    :sinh,
    :tanh,
    :acosh,
    :asinh,
    :atanh,
    :sqrt,
    :rsqrt,
    :cbrt,
    :negate,
    :sign,
    :abs,
    :bitwise_not,
    :is_nan,
    :is_infinity,
    :conjugate,
    :population_count,
    :count_leading_zeros,
    :floor,
    :ceil,
    :round,
    :erf,
    :erfc,
    :erf_inv,
    :acos,
    :asin,
    :atan,
    :real,
    :imag
  ]

  @binary_ops [
    :add,
    :subtract,
    :multiply,
    :divide,
    :pow,
    :remainder,
    :atan2,
    :max,
    :min,
    :bitwise_and,
    :bitwise_or,
    :bitwise_xor,
    :left_shift,
    :right_shift,
    :equal,
    :not_equal,
    :greater,
    :less,
    :greater_equal,
    :less_equal,
    :logical_and,
    :logical_or,
    :logical_xor,
    :quotient
  ]

  @doc """
  Walk an Nx.Defn.Expr tree, collect all nodes, topologically sort them,
  and return a list of FlatOp structs in dependency order.
  """
  @spec flatten(Nx.Tensor.t(struct())) :: [FlatOp.t()]
  def flatten(%Nx.Tensor{data: %Nx.Defn.Expr{}} = expr) do
    nodes = %{}
    {nodes, _} = collect(expr, nodes)

    sorted = topo_sort(nodes)
    flat_ops = Enum.map(sorted, fn id -> to_flat_op(nodes[id]) end)
    compute_consumers(flat_ops)
  end

  # Collect all Expr nodes into a map keyed by id
  defp collect(%Nx.Tensor{data: %Nx.Defn.Expr{id: id}} = tensor, acc) do
    if Map.has_key?(acc, id) do
      {acc, id}
    else
      acc = Map.put(acc, id, tensor)

      {acc, _} =
        tensor.data.args
        |> Enum.reduce({acc, []}, fn
          %Nx.Tensor{data: %Nx.Defn.Expr{}} = child, {a, ids} ->
            {a, child_id} = collect(child, a)
            {a, [child_id | ids]}

          # Lists of tensors (e.g. in concatenate)
          list, {a, ids} when is_list(list) ->
            collect_list(list, {a, ids})

          _other, {a, ids} ->
            {a, ids}
        end)

      {acc, id}
    end
  end

  defp collect_list(list, acc) do
    Enum.reduce(list, acc, fn
      %Nx.Tensor{data: %Nx.Defn.Expr{}} = child, {a, ids} ->
        {a, child_id} = collect(child, a)
        {a, [child_id | ids]}

      _other, acc_inner ->
        acc_inner
    end)
  end

  # Topological sort using Kahn's algorithm
  defp topo_sort(nodes) do
    # Build adjacency: for each node, what are its dependencies?
    deps =
      Map.new(nodes, fn {id, tensor} ->
        dep_ids =
          tensor.data.args
          |> Enum.flat_map(fn
            %Nx.Tensor{data: %Nx.Defn.Expr{id: dep_id}} ->
              [dep_id]

            list when is_list(list) ->
              Enum.flat_map(list, fn
                %Nx.Tensor{data: %Nx.Defn.Expr{id: dep_id}} -> [dep_id]
                _ -> []
              end)

            _ ->
              []
          end)
          |> Enum.filter(&Map.has_key?(nodes, &1))

        {id, dep_ids}
      end)

    # In-degree: each node's count = number of its dependencies
    in_degree =
      Enum.reduce(deps, Map.new(nodes, fn {id, _} -> {id, 0} end), fn {id, dep_ids}, deg ->
        Map.put(deg, id, length(dep_ids))
      end)

    # Kahn's: start with nodes that have no dependencies
    queue =
      in_degree
      |> Enum.filter(fn {_id, deg} -> deg == 0 end)
      |> Enum.map(fn {id, _} -> id end)
      |> Enum.sort()

    do_topo_sort(queue, deps, in_degree, nodes, [])
  end

  defp do_topo_sort([], _deps, _in_degree, _nodes, result), do: Enum.reverse(result)

  defp do_topo_sort([current | rest], deps, in_degree, nodes, result) do
    # Find all nodes that depend on current
    dependents =
      Enum.filter(deps, fn {_id, dep_ids} -> current in dep_ids end)
      |> Enum.map(fn {id, _} -> id end)

    # Decrease their in-degree
    {in_degree, new_ready} =
      Enum.reduce(dependents, {in_degree, []}, fn dep_id, {deg, ready} ->
        new_deg = deg[dep_id] - 1
        deg = Map.put(deg, dep_id, new_deg)

        if new_deg == 0 do
          {deg, [dep_id | ready]}
        else
          {deg, ready}
        end
      end)

    queue = rest ++ Enum.sort(new_ready)
    do_topo_sort(queue, deps, in_degree, nodes, [current | result])
  end

  defp to_flat_op(%Nx.Tensor{data: %Nx.Defn.Expr{} = expr} = tensor) do
    base = %FlatOp{
      id: expr.id,
      op: expr.op,
      shape: tensor.shape,
      type: tensor.type,
      consumers: []
    }

    case expr.op do
      :parameter ->
        [param_index] = expr.args
        %{base | param_index: param_index, args: []}

      :constant ->
        [value] = expr.args
        %{base | value: value, args: []}

      :tensor ->
        [literal_tensor] = expr.args
        %{base | tensor: literal_tensor, args: []}

      :dot ->
        [t1, c1, b1, t2, c2, b2] = expr.args
        %{base | args: [ref(t1), ref(t2)], contract_axes: {c1, c2}, batch_axes: {b1, b2}}

      op when op in @unary_ops ->
        [input] = expr.args
        %{base | args: [ref(input)]}

      op when op in @binary_ops ->
        [left, right] = expr.args
        %{base | args: [ref(left), ref(right)]}

      # For ops we don't handle specially, just collect tensor refs
      _other ->
        args =
          expr.args
          |> Enum.flat_map(fn
            %Nx.Tensor{data: %Nx.Defn.Expr{id: id}} ->
              [{:ref, id}]

            list when is_list(list) ->
              Enum.flat_map(list, fn
                %Nx.Tensor{data: %Nx.Defn.Expr{id: id}} -> [{:ref, id}]
                other -> [{:literal, other}]
              end)

            other ->
              [{:literal, other}]
          end)

        %{base | args: args}
    end
  end

  defp ref(%Nx.Tensor{data: %Nx.Defn.Expr{id: id}}), do: {:ref, id}

  defp compute_consumers(flat_ops) do
    # Build a map of id -> list of consumer ids
    consumer_map =
      Enum.reduce(flat_ops, %{}, fn op, acc ->
        op.args
        |> Enum.flat_map(fn
          {:ref, id} -> [id]
          _ -> []
        end)
        |> Enum.reduce(acc, fn dep_id, a ->
          Map.update(a, dep_id, [op.id], &[op.id | &1])
        end)
      end)

    Enum.map(flat_ops, fn op ->
      %{op | consumers: Map.get(consumer_map, op.id, [])}
    end)
  end
end

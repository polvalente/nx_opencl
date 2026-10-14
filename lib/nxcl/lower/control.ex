defmodule NxCL.Lower.Control do
  @moduledoc """
  Lowers `cond` and `while` nodes to one instruction each, with sub-programs.

  A `while` instruction has the operands of its initial carry and two
  sub-programs, `[cond, body]`. Inside them `{:carry, i}` is the i-th carry slot.
  A static counted loop adds the attrs `[:static, count]`.

  A `cond` instruction has as operands the outer values its clauses read, and
  the sub-programs `[pred_1, body_1, ..., pred_n, body_n, else]`. Inside them
  `{:carry, i}` is the i-th operand. A clause shares the parent scope in Nx; here
  it reads the parent only through these operands.

  Both instructions give their results as `{:result, i, k}`, the k-th output of
  the i-th instruction, and set `:out_type` and `:out_shape` to lists. A node
  with a tuple type maps to `{:tuple, refs}`.
  """

  alias Nx.Defn.Composite
  alias Nx.Defn.Tree
  alias Nx.Tensor, as: T
  alias NxCL.Lower
  alias NxCL.Program.Instruction

  @doc """
  Dependencies of a node for `NxCL.Defn.Tree.post_order/2`. A node the scope has a
  ref for stops the walk. A `cond` depends on the outer values its clauses read,
  not on its clause bodies.
  """
  @spec scope_deps(map(), map()) :: {:ok, [map()]} | :default
  def scope_deps(%T{data: %Nx.Defn.Expr{id: id}}, %{refs: refs}) when is_map_key(refs, id),
    do: {:ok, []}

  def scope_deps(%T{data: %Nx.Defn.Expr{op: :cond}} = node, %{avail: avail}),
    do: {:ok, free_nodes(node, avail)}

  def scope_deps(_node, _state), do: :default

  @doc "Lowers a `cond` node: one instruction, clause sub-programs, explicit operands."
  @spec cond_node(map(), map()) :: map()
  def cond_node(state, %T{data: %Nx.Defn.Expr{id: id, args: [clauses, last]}} = node) do
    {operands, seeds} = bind(state, free_nodes(node, state.avail))

    {subs, state} =
      Enum.reduce(clauses, {[], state}, fn {pred, body}, {subs, state} ->
        {pred_sub, state} = Lower.lower_scope(state, [pred], seeds, :cond)
        {body_sub, state} = Lower.lower_scope(state, leaves(body), seeds, :cond)
        {[body_sub, pred_sub | subs], state}
      end)

    {else_sub, state} = Lower.lower_scope(state, leaves(last), seeds, :cond)

    instr = %Instruction{
      op: :cond,
      operands: operands,
      out_type: Enum.map(leaves(last), & &1.type),
      out_shape: Enum.map(leaves(last), & &1.shape),
      subprograms: Enum.reverse([else_sub | subs])
    }

    put_outputs(state, id, node, instr)
  end

  @doc "Lowers a `while` node: one instruction with `[cond, body]` sub-programs."
  @spec while_node(map(), map()) :: map()
  def while_node(
        state,
        %T{data: %Nx.Defn.Expr{id: id, args: [initial, arg, pred, block]}} = node
      ) do
    initial = leaves(initial)
    arg = leaves(arg)
    body = leaves(block)

    operands = Enum.map(initial, &Map.fetch!(state.refs, &1.data.id))

    seeds =
      arg |> Enum.with_index() |> Map.new(fn {p, i} -> {p.data.id, {:carry, i}} end)

    {cond_sub, state} = Lower.lower_scope(state, [pred], seeds, :while)
    {body_sub, state} = Lower.lower_scope(state, body, seeds, :while)

    attrs =
      case static_count(initial, arg, pred, body) do
        {:ok, count} -> [:static, count]
        :error -> []
      end

    instr = %Instruction{
      op: :while,
      operands: operands,
      out_type: Enum.map(initial, & &1.type),
      out_shape: Enum.map(initial, & &1.shape),
      attrs: attrs,
      subprograms: [cond_sub, body_sub]
    }

    put_outputs(state, id, node, instr)
  end

  defp put_outputs(state, id, node, instr) do
    index = length(state.instrs)
    refs = for k <- 0..(length(instr.out_type) - 1)//1, do: {:result, index, k}

    ref =
      case node.type do
        {:tuple, _} -> {:tuple, refs}
        _ -> hd(refs)
      end

    %{state | instrs: [instr | state.instrs], refs: Map.put(state.refs, id, ref)}
  end

  defp leaves(container), do: Composite.flatten_list([container])

  # -- Outer values of a cond ------------------------------------------------

  # The nodes the clauses read that the enclosing scope provides: parameters and
  # nodes `avail` names. The walk stops at them. Any other node belongs to the cond.
  defp free_nodes(%T{data: %Nx.Defn.Expr{args: [clauses, last]}}, avail) do
    roots = Enum.flat_map(clauses, fn {pred, body} -> [pred | leaves(body)] end) ++ leaves(last)

    {_seen, free} =
      Enum.reduce(roots, {MapSet.new(), []}, &collect(&1, &2, avail))

    Enum.reverse(free)
  end

  defp collect(%T{data: %Nx.Defn.Expr{id: id, op: op}} = node, {seen, free} = acc, avail) do
    cond do
      MapSet.member?(seen, id) ->
        acc

      op in [:constant, :tensor, :fun] ->
        {MapSet.put(seen, id), free}

      op == :parameter or is_map_key(avail, id) ->
        {MapSet.put(seen, id), [node | free]}

      true ->
        {_node, acc} =
          Tree.apply_args(node, :scope, {MapSet.put(seen, id), free}, fn dep, acc ->
            {dep, collect(dep, acc, avail)}
          end)

        acc
    end
  end

  # Operands for the free nodes, and the ref each node has inside the sub-programs.
  defp bind(state, free) do
    {operands, seeds} =
      Enum.reduce(free, {[], %{}}, fn node, {operands, seeds} ->
        {inner, operands} = bind_ref(Map.fetch!(state.refs, node.data.id), operands)
        {operands, Map.put(seeds, node.data.id, inner)}
      end)

    {operands, seeds}
  end

  defp bind_ref({:tuple, refs}, operands) do
    {inner, operands} = Enum.map_reduce(refs, operands, &bind_ref/2)
    {{:tuple, inner}, operands}
  end

  defp bind_ref(ref, operands) do
    case Enum.find_index(operands, &(&1 == ref)) do
      nil -> {{:carry, length(operands)}, operands ++ [ref]}
      index -> {{:carry, index}, operands}
    end
  end

  # -- Static trip count ------------------------------------------------------

  # A counter that starts at a literal, steps by a literal and is compared with a
  # literal bound has a known trip count; the executor then reads no predicate.
  defp static_count([start | _], [counter | _], pred, [next | _]) do
    with {:ok, from} <- literal(start),
         true <- integer_type?(counter.type),
         {:ok, kind, bound} <- compare(pred, counter),
         {:ok, step} <- increment(next, counter),
         {:ok, count} <- trip_count(kind, from, bound, step),
         true <- fits?(counter.type, from + count * step) do
      {:ok, count}
    else
      _ -> :error
    end
  end

  defp static_count(_initial, _arg, _pred, _body), do: :error

  defp literal(%T{data: %Nx.Defn.Expr{op: :constant, args: [n]}}) when is_integer(n),
    do: {:ok, n}

  defp literal(_node), do: :error

  defp integer_type?({kind, _}), do: kind in [:s, :u]

  defp compare(
         %T{data: %Nx.Defn.Expr{op: op, args: [%T{data: %Nx.Defn.Expr{id: id}}, bound]}},
         %T{data: %Nx.Defn.Expr{id: id}}
       )
       when op in [:less, :less_equal, :greater, :greater_equal] do
    case literal(bound) do
      {:ok, n} -> {:ok, op, n}
      :error -> :error
    end
  end

  defp compare(_pred, _counter), do: :error

  defp increment(
         %T{data: %Nx.Defn.Expr{op: :add, args: [a, b]}},
         %T{data: %Nx.Defn.Expr{id: id}}
       ) do
    case {a, b} do
      {%T{data: %Nx.Defn.Expr{id: ^id}}, step} -> literal(step)
      {step, %T{data: %Nx.Defn.Expr{id: ^id}}} -> literal(step)
      _ -> :error
    end
  end

  defp increment(_next, _counter), do: :error

  defp trip_count(:less_equal, from, bound, step) when step > 0,
    do: {:ok, if(bound >= from, do: div(bound - from, step) + 1, else: 0)}

  defp trip_count(:less, from, bound, step) when step > 0,
    do: {:ok, if(bound > from, do: div(bound - from - 1, step) + 1, else: 0)}

  defp trip_count(:greater_equal, from, bound, step) when step < 0,
    do: {:ok, if(from >= bound, do: div(from - bound, -step) + 1, else: 0)}

  defp trip_count(:greater, from, bound, step) when step < 0,
    do: {:ok, if(from > bound, do: div(from - bound - 1, -step) + 1, else: 0)}

  defp trip_count(_kind, _from, _bound, _step), do: :error

  # The counter must not wrap on the way, or the loop would not stop where the count says.
  defp fits?({:s, bits}, n), do: n >= -Bitwise.bsl(1, bits - 1) and n < Bitwise.bsl(1, bits - 1)
  defp fits?({:u, bits}, n), do: n >= 0 and n < Bitwise.bsl(1, bits)
end

defmodule NxCL.Compiler.Lower.Control do
  @moduledoc false
  # Lowers `cond` and `while` to one instruction each, with nested bodies (see
  # `NxCL.Compiler.Instruction`). A `while` has its initial carry as operands. A
  # `cond` has the outer values its clauses read; a clause reads the parent only
  # through them. Inside a body, `{:carry, i}` is the i-th operand.

  import NxCL.Compiler.Lower.Shared, only: [leaves: 1]

  alias Nx.Defn.Tree
  alias Nx.Tensor, as: T
  alias NxCL.Compiler.Instruction
  alias NxCL.Compiler.Lower
  alias NxCL.Compiler.Lower.Shared

  @doc """
  Dependencies of a node for `NxCL.Compiler.Tree.post_order/2`. A node with a
  ref stops the walk. A `cond` depends on the outer values its clauses read,
  not on its bodies.
  """
  @spec scope_deps(Shared.tensor(), Lower.state()) :: {:ok, [Shared.tensor()]} | :default
  def scope_deps(%T{data: %Nx.Defn.Expr{id: id}}, %{refs: refs}) when is_map_key(refs, id),
    do: {:ok, []}

  def scope_deps(%T{data: %Nx.Defn.Expr{op: :cond}} = node, %{avail: avail}),
    do: {:ok, free_nodes(node, avail)}

  def scope_deps(_node, _state), do: :default

  @typedoc "Lowers one graph node, as `NxCL.Compiler.Lower` does."
  @type expand :: (Shared.tensor(), Lower.state() -> Lower.state())

  # Deeper nesting raises instead of exhausting the stack.
  @max_depth 64

  @doc """
  Lowers `roots` as a body of `parent`. Returns the body and the parent with the
  captures, constants and host calls the body added. `seeds` maps node ids to the
  `{:carry, i}` refs the scope starts with. A `while` body may also see the
  parent's nodes in `:outer`, only to report a read of them. `expand` lowers one
  node.
  """
  @spec scope(
          Lower.state(),
          [Shared.tensor()],
          %{reference() => Lower.node_ref()},
          :while | :cond,
          expand()
        ) :: {Instruction.body(), Lower.state()}
  def scope(parent, roots, seeds, kind, expand) do
    if parent.depth >= @max_depth do
      raise ArgumentError, "cond and while nest more than #{@max_depth} levels deep"
    end

    tuple = List.to_tuple(roots)
    owner = Tree.scope_ids(tuple)

    inner = %{
      parent
      | instrs: [],
        refs: seeds,
        avail: Map.merge(owner, seeds),
        outer: if(kind == :while, do: parent.avail, else: %{}),
        depth: parent.depth + 1,
        scope: :sub
    }

    ordered = NxCL.Compiler.Tree.post_order(tuple, &scope_deps(&1, inner))

    inner =
      Enum.reduce(ordered, inner, fn %T{data: %Nx.Defn.Expr{id: id}} = node, state ->
        cond do
          is_map_key(state.refs, id) -> state
          reads_outer?(state, node) -> raise_outer(node)
          true -> expand.(node, state)
        end
      end)

    outputs = Enum.map(roots, &Map.fetch!(inner.refs, &1.data.id))
    body = {Enum.reverse(inner.instrs), outputs}

    {body,
     %{
       parent
       | captures: inner.captures,
         constants: inner.constants,
         host_calls: inner.host_calls
     }}
  end

  # A literal tensor or constant may sit in both scopes.
  defp reads_outer?(%{outer: outer}, %T{data: %Nx.Defn.Expr{id: id, op: op}}),
    do: op not in [:constant, :tensor] and is_map_key(outer, id)

  defp raise_outer(%T{data: %Nx.Defn.Expr{op: op}}) do
    raise ArgumentError,
          "an inner scope reads #{inspect(op)} from its parent scope; " <>
            "pass the value in as an argument of the while or cond"
  end

  @doc "Lowers a `cond` node."
  @spec cond_node(Lower.state(), Shared.tensor(), expand()) :: Lower.state()
  def cond_node(state, %T{data: %Nx.Defn.Expr{id: id, args: [clauses, last]}} = node, expand) do
    {operands, seeds} = bind(state, free_nodes(node, state.avail))

    {lowered, state} =
      Enum.map_reduce(clauses, state, fn {pred, body}, state ->
        {pred_body, state} = scope(state, [pred], seeds, :cond, expand)
        {inner_body, state} = scope(state, leaves(body), seeds, :cond, expand)
        {{pred_body, inner_body}, state}
      end)

    {else_body, state} = scope(state, leaves(last), seeds, :cond, expand)

    instr = %Instruction{
      op: :cond,
      operands: operands,
      out_type: Enum.map(leaves(last), & &1.type),
      out_shape: Enum.map(leaves(last), & &1.shape),
      attrs: [clauses: lowered, else: else_body]
    }

    put_outputs(state, id, node, instr)
  end

  @doc "Lowers a `while` node."
  @spec while_node(Lower.state(), Shared.tensor(), expand()) :: Lower.state()
  def while_node(
        state,
        %T{data: %Nx.Defn.Expr{id: id, args: [initial, arg, pred, block]}} = node,
        expand
      ) do
    initial = leaves(initial)
    arg = leaves(arg)
    body = leaves(block)

    operands = Enum.map(initial, &Map.fetch!(state.refs, &1.data.id))

    seeds =
      arg |> Enum.with_index() |> Map.new(fn {p, i} -> {p.data.id, {:carry, i}} end)

    {cond_body, state} = scope(state, [pred], seeds, :while, expand)
    {inner_body, state} = scope(state, body, seeds, :while, expand)

    attrs =
      case static_count(initial, arg, pred, body) do
        {:ok, count} -> [cond: cond_body, body: inner_body, static: count]
        :error -> [cond: cond_body, body: inner_body]
      end

    instr = %Instruction{
      op: :while,
      operands: operands,
      out_type: Enum.map(initial, & &1.type),
      out_shape: Enum.map(initial, & &1.shape),
      attrs: attrs
    }

    put_outputs(state, id, node, instr)
  end

  defp put_outputs(state, id, node, instr) do
    refs = Shared.result_refs(length(state.instrs), length(instr.out_type))
    ref = Shared.node_ref(node, refs)

    %{state | instrs: [instr | state.instrs], refs: Map.put(state.refs, id, ref)}
  end

  # The nodes the clauses read that the enclosing scope provides (parameters and
  # `avail` nodes). Any other node belongs to the cond.
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

  # A counter that starts, steps and ends at literals has a known trip count, so
  # the executor reads no predicate.
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

  # The counter must not wrap, or the loop would not stop at the count.
  defp fits?({:s, bits}, n), do: n >= -Bitwise.bsl(1, bits - 1) and n < Bitwise.bsl(1, bits - 1)
  defp fits?({:u, bits}, n), do: n >= 0 and n < Bitwise.bsl(1, bits)
end

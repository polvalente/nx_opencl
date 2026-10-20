defmodule NxCL.Compiler.Lower.Shared do
  @moduledoc false
  # Helpers for `NxCL.Compiler.Lower`, its parts and `NxCL.Compiler.Evaluator`.
  # A `t:builder/0` is any map with `:instrs` and `:constants` lists.

  alias Nx.Defn.Composite
  alias Nx.Tensor, as: T
  alias NxCL.Compiler.Instruction
  alias NxCL.Compiler.Ops
  alias NxCL.Compiler.Tree

  @typedoc "An `Nx.Tensor` of any backend, as an expression node or an operand."
  @type tensor :: %T{}

  @type builder :: %{
          required(:instrs) => [Instruction.t()],
          required(:constants) => [{term(), Nx.Type.t()}],
          optional(atom()) => term()
        }

  # Binary ops a `reduce` body may hold; the result has the operand type.
  @step_ops ~w(add subtract multiply divide quotient remainder pow atan2 max min bitwise_and
               bitwise_or bitwise_xor left_shift right_shift)a

  # Ops Nx 1.0 rejects on complex tensors.
  @complex_rejected ~w(atan2 bitwise_and bitwise_not bitwise_or bitwise_xor bitcast ceil clip
                       count_leading_zeros erf erf_inv erfc floor greater greater_equal
                       left_shift less less_equal max min population_count quotient remainder
                       right_shift round)a

  @scan_blocks %{
    Nx.Block.CumulativeSum => :cumulative_sum,
    Nx.Block.CumulativeProduct => :cumulative_product,
    Nx.Block.CumulativeMin => :cumulative_min,
    Nx.Block.CumulativeMax => :cumulative_max
  }

  @doc "The binary ops a `reduce` body may hold."
  @spec step_ops() :: [atom()]
  def step_ops(), do: @step_ops

  @doc "The scan block structs and the op each one lowers to."
  @spec scan_blocks() :: %{module() => atom()}
  def scan_blocks(), do: @scan_blocks

  @doc "Adds an instruction and returns its ref."
  @spec push(builder(), Instruction.t()) :: {builder(), Instruction.ref()}
  def push(state, instruction) do
    ref = {:result, length(state.instrs)}
    {%{state | instrs: [instruction | state.instrs]}, ref}
  end

  @doc "Adds a scalar constant and returns its ref."
  @spec put_const(builder(), term(), Nx.Type.t()) :: {builder(), Instruction.ref()}
  def put_const(state, value, type) do
    ref = {:const, length(state.constants)}
    {%{state | constants: [{value, type} | state.constants]}, ref}
  end

  @doc """
  Raises `ArgumentError` for an op Nx rejects on complex when any tensor is complex.
  Nx front-end functions raise first for all of these but `atan2`, which Nx lets
  through to the backend. A graph built by hand with `Nx.Defn.Expr` skips the
  front end too.
  """
  @spec check_complex(atom(), tensor(), [tensor()]) :: :ok
  def check_complex(op, node, operands) do
    if op in @complex_rejected and Enum.any?([node | operands], &Nx.Type.complex?(&1.type)) do
      raise ArgumentError, "#{inspect(op)} does not support complex tensors"
    end

    :ok
  end

  @doc "The reduced axes, sorted; `nil` means every axis."
  @spec axes(tensor(), [non_neg_integer()] | nil) :: [non_neg_integer()]
  def axes(%T{shape: shape}, nil), do: Enum.to_list(0..(tuple_size(shape) - 1)//1)
  def axes(_x, axes), do: Enum.sort(axes)

  @doc "True when `axes` is the identity permutation `[0, 1, ...]`."
  @spec identity?([non_neg_integer()]) :: boolean()
  def identity?(axes), do: axes == Enum.to_list(0..(length(axes) - 1)//1)

  @doc """
  True for a `reduce` body of one `add`, `multiply`, `max` or `min` over the two
  parameters. The op is associative and commutative, so the device may fold in a tree.
  """
  @spec monoid?([{atom(), non_neg_integer(), non_neg_integer()}]) :: boolean()
  def monoid?([{op, a, b}]) when op in [:add, :multiply, :max, :min],
    do: {a, b} in [{0, 1}, {1, 0}]

  def monoid?(_steps), do: false

  @doc """
  Reads the body of a `reduce` or `window_reduce` as steps `{op, a, b}`. Variable
  0 and 1 are the two parameters; step `k` makes variable `2 + k`. `what` names
  the op in error messages. Any node but a binary op over earlier variables raises
  `ArgumentError` ("does not lower op ...").
  """
  @spec reduce_steps([tensor()], tensor(), Nx.Type.t(), atom()) ::
          [{atom(), non_neg_integer(), non_neg_integer()}]
  def reduce_steps(params, body, type, what) do
    if length(params) != 2 do
      unsupported(what, "(the function must take two arguments)")
    end

    {vars, steps} =
      body
      |> Tree.post_order()
      |> Enum.reduce({%{}, []}, fn
        %T{data: %Nx.Defn.Expr{id: id, op: :parameter, args: [pos]}}, {vars, steps} ->
          {Map.put(vars, id, pos), steps}

        %T{data: %Nx.Defn.Expr{id: id, op: op, args: [l, r]}} = node, {vars, steps}
        when op in @step_ops ->
          if Enum.any?([node, l, r], &(&1.type != type)) do
            unsupported(op, "(mixed types in a #{what} body)")
          end

          a = Map.get(vars, l.data.id) || unsupported(op, "(an operand of a #{what} body)")
          b = Map.get(vars, r.data.id) || unsupported(op, "(an operand of a #{what} body)")
          {Map.put(vars, id, 2 + length(steps)), [{op, a, b} | steps]}

        %T{data: %Nx.Defn.Expr{op: op}}, _acc ->
          unsupported(op, "(in a #{what} body)")
      end)

    steps = Enum.reverse(steps)

    if steps == [] or Map.fetch!(vars, body.data.id) != 1 + length(steps) do
      unsupported(what, "(the body must end in a binary op)")
    end

    steps
  end

  @doc """
  Pushes the steps with `refs` as variables 0 and 1: `fun(total, init)` for a
  `reduce` whose parts were folded apart. Returns the ref of the last step.
  """
  @spec fold_steps(
          builder(),
          [{atom(), non_neg_integer(), non_neg_integer()}],
          [Instruction.ref()],
          Nx.Type.t(),
          tuple()
        ) :: {builder(), Instruction.ref()}
  def fold_steps(state, steps, refs, type, shape) do
    {state, refs} =
      Enum.reduce(steps, {state, refs}, fn {op, a, b}, {state, refs} ->
        {state, ref} =
          push(state, Ops.binary(op, Enum.at(refs, a), Enum.at(refs, b), type, shape))

        {state, refs ++ [ref]}
      end)

    {state, List.last(refs)}
  end

  defp unsupported(op, detail),
    do: raise(ArgumentError, "does not lower op #{inspect(op)} #{detail}")

  @doc "How often each ref is read, by instructions or as an output."
  @spec ref_counts([Instruction.t()], [Instruction.ref()]) :: %{
          Instruction.ref() => non_neg_integer()
        }
  def ref_counts(instructions, outputs) do
    instructions
    |> Enum.flat_map(& &1.operands)
    |> Enum.concat(outputs)
    |> Enum.frequencies()
  end

  @doc "The tensors of a container, flattened."
  @spec leaves(Nx.Container.t()) :: [tensor()]
  def leaves(container), do: Composite.flatten_list([container])

  @doc """
  The refs `{:result, at, k}` for the `count` results of instruction `at`.
  """
  @spec result_refs(non_neg_integer(), non_neg_integer()) :: [Instruction.ref()]
  def result_refs(at, count), do: for(k <- 0..(count - 1)//1, do: {:result, at, k})

  @doc """
  `{:tuple, refs}` for a tuple-typed node, the first ref otherwise.
  """
  @spec node_ref(tensor(), [Instruction.ref()]) ::
          Instruction.ref() | {:tuple, [Instruction.ref()]}
  def node_ref(%T{type: {:tuple, _}}, refs), do: {:tuple, refs}
  def node_ref(_node, refs), do: hd(refs)

  @doc """
  Crops or zero-pads `axis` of `ref` to the length of `out` on that axis.
  """
  @spec fit_axis(
          builder(),
          Instruction.ref(),
          tuple(),
          Nx.Type.t(),
          tuple(),
          non_neg_integer()
        ) :: {builder(), Instruction.ref()}
  def fit_axis(state, ref, shape, type, out, axis) do
    have = elem(shape, axis)
    want = elem(out, axis)
    fitted = put_elem(shape, axis, want)

    cond do
      have == want -> {state, ref}
      have > want -> crop_axis(state, ref, type, fitted)
      true -> pad_axis(state, ref, type, {shape, fitted, axis})
    end
  end

  defp crop_axis(state, ref, type, fitted) do
    rank = tuple_size(fitted)

    {starts, state} =
      Enum.map_reduce(1..rank//1, state, fn _axis, state ->
        {state, ref} = put_const(state, 0, {:s, 64})
        {ref, state}
      end)

    strides = List.duplicate(1, rank)
    push(state, Ops.slice(ref, starts, type, fitted, Tuple.to_list(fitted), strides))
  end

  defp pad_axis(state, ref, type, {shape, fitted, axis}) do
    {state, zero} = put_const(state, 0, type)
    extra = elem(fitted, axis) - elem(shape, axis)
    config = for a <- 0..(tuple_size(shape) - 1), do: {0, if(a == axis, do: extra, else: 0), 0}
    push(state, Ops.pad(ref, zero, type, fitted, config))
  end
end

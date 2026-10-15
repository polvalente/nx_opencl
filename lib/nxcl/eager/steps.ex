defmodule NxCL.Eager.Steps do
  @moduledoc false
  # Reads the body of a `reduce` or `window_reduce` function as steps, for the
  # eager backend. The backend gets an Elixir function, not an `Nx.Defn.Expr`
  # graph, so it runs the function once on two scalar tensors of this module.
  # Each binary op the body calls records one step. The steps have the shape
  # `NxCL.Lower.reduce_steps/3` returns: `{op, a, b}` over variables 0 and 1 (the
  # two arguments) and `2 + k` (the result of step `k`).
  #
  # An op that is not a binary step op has no function here, so the call raises
  # and the trace turns that into the lowering's "does not yet lower op" error.

  alias Nx.Tensor, as: T
  alias NxCL.Lower

  defstruct [:var]

  @key {__MODULE__, :trace}

  @doc """
  Steps of `fun`, a two-argument function over scalars of `type`. `what` names the
  caller (`:reduce` or `:window_reduce`) in error messages.
  """
  @spec trace(function(), Nx.Type.t(), atom()) :: [{atom(), non_neg_integer(), non_neg_integer()}]
  def trace(fun, type, what) do
    Process.put(@key, {what, []})

    try do
      result = fun.(var(0, type), var(1, type))
      {_what, steps} = Process.get(@key)
      steps = Enum.reverse(steps)

      case result do
        %T{data: %__MODULE__{var: v}} when steps != [] and v == 1 + length(steps) -> steps
        _ -> unsupported(what, "(the body must end in a binary op)")
      end
    rescue
      e in UndefinedFunctionError ->
        if e.module == __MODULE__ do
          unsupported(e.function, "(in a #{what} body)")
        else
          reraise e, __STACKTRACE__
        end
    after
      Process.delete(@key)
    end
  end

  defp var(n, type), do: %T{type: type, shape: {}, names: [], data: %__MODULE__{var: n}}

  for op <- Lower.step_ops() do
    @doc false
    @spec unquote(op)(map(), map(), map()) :: map()
    # credo:disable-for-next-line Credo.Check.Readability.Specs
    def unquote(op)(out, left, right), do: record(unquote(op), out, left, right)
  end

  defp record(op, %T{type: type} = out, left, right) do
    {what, steps} = Process.get(@key)

    if Enum.any?([out, left, right], &(&1.type != type)) do
      unsupported(op, "(mixed types in a #{what} body)")
    end

    a = operand(op, what, left)
    b = operand(op, what, right)
    n = 2 + length(steps)
    Process.put(@key, {what, [{op, a, b} | steps]})
    %{out | data: %__MODULE__{var: n}}
  end

  defp operand(_op, _what, %T{data: %__MODULE__{var: v}}), do: v
  defp operand(op, what, _other), do: unsupported(op, "(an operand of a #{what} body)")

  defp unsupported(op, detail),
    do: raise(ArgumentError, "does not yet lower op #{inspect(op)} #{detail}")
end

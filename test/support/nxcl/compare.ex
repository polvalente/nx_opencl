defmodule NxCL.TestSupport.Compare do
  @moduledoc """
  Runs a function through a candidate compiler and through `Nx.Defn.Evaluator` on
  `Nx.BinaryBackend`, then compares.

  Type, shape and names must match exactly, and so must integer values. Float
  values match within a tolerance per type. NaN must match NaN; the sign of zero
  counts only with `signed_zero: true`.

  `NxCL.Compiler` rows run twice, through the compiler and eagerly (arguments on
  `NxCL.Backend`, the evaluator walking the graph), against the same expected value.
  """

  import ExUnit.Assertions

  @default_tol %{
    {:f, 64} => 1.0e-12,
    {:f, 32} => 1.0e-5,
    {:f, 16} => 1.0e-2,
    {:bf, 16} => 1.0e-2,
    {:f, 8} => 1.0e-1
  }

  @doc "The default absolute and relative tolerance for `type` (a complex type uses that of its parts)."
  @spec default_tolerance(Nx.Type.t()) :: float()
  def default_tolerance({:c, size}), do: default_tolerance({:f, div(size, 2)})
  def default_tolerance(type), do: Map.fetch!(@default_tol, type)

  @doc """
  Checks `fun` on `args` against the reference. Raises `ExUnit.AssertionError`
  on any mismatch.

  Options: `:compiler` (default `NxCL.Compiler`), `:compiler_opts`, `:rtol`,
  `:atol` (override the per-type tolerance), `:signed_zero` (compare the sign
  of zero), `:backend` (default `true`; set `false` to skip the eager run, for a
  row whose eager run adds no coverage and costs too much time).
  """
  @spec assert_same(function(), list(), keyword()) :: :ok
  def assert_same(fun, args, opts \\ []) do
    assert_expected(fun, args, reference(fun, args), opts)
  end

  @doc """
  Like `assert_same/3`, but compares against `expected`, a value the caller
  states, instead of running the reference. For a row where `Nx.BinaryBackend`
  is wrong or too slow; the row's comment says why. Takes the same options.
  """
  @spec assert_expected(function(), list(), Nx.Container.t(), keyword()) :: :ok
  def assert_expected(fun, args, expected, opts \\ []) do
    compiler = Keyword.get(opts, :compiler, NxCL.Compiler)
    compiler_opts = Keyword.get(opts, :compiler_opts, [])

    actual = apply(Nx.Defn.jit(fun, [compiler: compiler] ++ compiler_opts), args)

    try do
      compare_containers(expected, actual, opts)
    rescue
      e in ExUnit.AssertionError ->
        reraise %{e | message: e.message <> "\nseed: #{seed()}"}, __STACKTRACE__
    end

    if compiler == NxCL.Compiler and Keyword.get(opts, :backend, true) do
      assert_backend(fun, args, expected, opts)
    end

    :ok
  end

  # `NxCL.Backend` is also the default, so literals the function builds live on it.
  defp assert_backend(fun, args, expected, opts) do
    actual =
      Nx.with_default_backend(NxCL.Backend, fn ->
        apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), Enum.map(args, &to_nxcl_backend/1))
      end)

    try do
      compare_containers(expected, actual, opts)
    rescue
      e in ExUnit.AssertionError ->
        reraise %{e | message: e.message <> "\npath: backend\nseed: #{seed()}"},
                __STACKTRACE__
    end

    :ok
  end

  @doc "The seed ExUnit runs with; generators derive their keys from it."
  @spec seed() :: non_neg_integer()
  def seed, do: ExUnit.configuration()[:seed] || 0

  defp reference(fun, args) do
    binary_args = Enum.map(args, &to_binary_backend/1)
    apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), binary_args)
  end

  defp to_binary_backend(%Nx.Tensor{} = t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  defp to_binary_backend(container) when is_map(container) or is_tuple(container),
    do: Nx.Defn.Composite.traverse(container, &Nx.backend_copy(&1, Nx.BinaryBackend))

  defp to_binary_backend(other), do: other

  defp to_nxcl_backend(%Nx.Tensor{} = t), do: Nx.backend_copy(t, NxCL.Backend)

  defp to_nxcl_backend(container) when is_map(container) or is_tuple(container),
    do: Nx.Defn.Composite.traverse(container, &Nx.backend_copy(&1, NxCL.Backend))

  defp to_nxcl_backend(other), do: other

  defp compare_containers(expected, actual, opts) do
    assert structure(expected) == structure(actual),
           "container structure differs:\n  expected #{inspect(structure(expected))}\n  actual   #{inspect(structure(actual))}"

    expected_leaves = Nx.Defn.Composite.flatten_list([expected])
    actual_leaves = Nx.Defn.Composite.flatten_list([actual])

    expected_leaves
    |> Enum.zip(actual_leaves)
    |> Enum.with_index()
    |> Enum.each(fn {{e, a}, i} -> compare_leaf(e, a, i, opts) end)
  end

  defp structure(%Nx.Tensor{}), do: :tensor

  defp structure(tuple) when is_tuple(tuple),
    do: {:tuple, tuple |> Tuple.to_list() |> Enum.map(&structure/1)}

  defp structure(%{} = map) when not is_struct(map),
    do: {:map, map |> Enum.sort() |> Enum.map(fn {k, v} -> {k, structure(v)} end)}

  defp structure(other), do: {:other, other}

  defp compare_leaf(%Nx.Tensor{} = e, %Nx.Tensor{} = a, i, opts) do
    assert e.type == a.type, "leaf #{i}: type #{inspect(a.type)}, expected #{inspect(e.type)}"

    assert e.shape == a.shape,
           "leaf #{i}: shape #{inspect(a.shape)}, expected #{inspect(e.shape)}"

    assert e.names == a.names,
           "leaf #{i}: names #{inspect(a.names)}, expected #{inspect(e.names)}"

    e = Nx.backend_copy(e, Nx.BinaryBackend)
    a = Nx.backend_copy(a, Nx.BinaryBackend)
    compare_values(e, a, i, opts)
  end

  defp compare_values(%Nx.Tensor{type: {:c, size}} = e, a, i, opts) do
    part = {:f, div(size, 2)}

    for {name, fun} <- [real: &Nx.real/1, imag: &Nx.imag/1] do
      compare_values(Nx.as_type(fun.(e), part), Nx.as_type(fun.(a), part), "#{i}.#{name}", opts)
    end

    :ok
  end

  defp compare_values(%Nx.Tensor{type: {kind, _}} = e, a, i, _opts) when kind in [:s, :u] do
    assert Nx.to_binary(e) == Nx.to_binary(a),
           "leaf #{i}: integer values differ\n  expected #{inspect(Nx.to_flat_list(e), limit: 20, charlists: :as_lists)}\n  actual   #{inspect(Nx.to_flat_list(a), limit: 20, charlists: :as_lists)}"
  end

  defp compare_values(%Nx.Tensor{type: type} = e, a, i, opts) do
    tol = Map.fetch!(@default_tol, type)
    rtol = Keyword.get(opts, :rtol, tol)
    atol = Keyword.get(opts, :atol, tol)
    signed_zero = Keyword.get(opts, :signed_zero, false)

    nan_e = Nx.is_nan(e) |> Nx.to_flat_list()
    nan_a = Nx.is_nan(a) |> Nx.to_flat_list()

    if nan_e != nan_a do
      idx = Enum.find_index(Enum.zip(nan_e, nan_a), fn {x, y} -> x != y end)

      flunk(
        "leaf #{i}: NaN mismatch at flat index #{idx}: expected #{inspect(Enum.at(Nx.to_flat_list(e), idx))}, actual #{inspect(Enum.at(Nx.to_flat_list(a), idx))}"
      )
    end

    e_list = Nx.to_flat_list(e)
    a_list = Nx.to_flat_list(a)

    e_list
    |> Enum.zip(a_list)
    |> Enum.with_index()
    |> Enum.each(fn {{x, y}, idx} ->
      unless close?(x, y, rtol, atol, signed_zero) do
        flunk(
          "leaf #{i}: value mismatch at flat index #{idx}: expected #{inspect(x)}, actual #{inspect(y)} (rtol #{rtol}, atol #{atol})"
        )
      end
    end)
  end

  defp close?(:nan, :nan, _, _, _), do: true
  defp close?(x, y, _, _, _) when is_atom(x) or is_atom(y), do: x == y

  defp close?(x, y, rtol, atol, signed_zero) do
    cond do
      x == 0 and y == 0 -> not signed_zero or sign_bit(x) == sign_bit(y)
      true -> abs(x - y) <= atol + rtol * abs(x)
    end
  end

  defp sign_bit(x) do
    <<s::1, _::63>> = <<x * 1.0::float-64>>
    s
  end
end

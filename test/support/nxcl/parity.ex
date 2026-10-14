defmodule NxCL.Parity do
  @moduledoc """
  Differential harness: runs a function through a candidate compiler and
  through `Nx.Defn.Evaluator` on `Nx.BinaryBackend`, then compares the results.

  Type, shape, and names must match exactly. Integer values must match
  exactly. Float values match within a tolerance per type. A passing call
  records one `{op, dtype, path}` cell for each distinct input and output type.
  """

  import ExUnit.Assertions

  alias NxCL.Parity.Recorder

  @default_tol %{
    {:f, 64} => 1.0e-12,
    {:f, 32} => 1.0e-5,
    {:f, 16} => 1.0e-2,
    {:bf, 16} => 1.0e-2,
    {:f, 8} => 1.0e-1
  }

  @capabilities %{
    fp64: "cl_khr_fp64",
    fp16: "cl_khr_fp16",
    int64_atomics: "cl_khr_int64_base_atomics"
  }

  @doc """
  Checks `fun` on `args` against the reference. Raises `ExUnit.AssertionError`
  on any mismatch.

  Options: `:compiler` (default `NxCL.Compiler`), `:compiler_opts`, `:rtol`,
  `:atol` (override the per-type tolerance), `:signed_zero` (compare the sign
  of zero), `:record` (default `true`; record the passing cells).
  """
  @spec assert_parity(atom(), function(), list(), keyword()) :: :ok
  def assert_parity(op, fun, args, opts \\ []) do
    compiler = Keyword.get(opts, :compiler, NxCL.Compiler)
    compiler_opts = Keyword.get(opts, :compiler_opts, [])

    expected = reference(fun, args)

    actual = apply(Nx.Defn.jit(fun, [compiler: compiler] ++ compiler_opts), args)

    try do
      compare_containers(expected, actual, opts)
    rescue
      e in ExUnit.AssertionError ->
        reraise %{e | message: e.message <> "\nop: #{inspect(op)}\nseed: #{seed()}"},
                __STACKTRACE__
    end

    if compiler == NxCL.Compiler and Keyword.get(opts, :record, true) do
      for dtype <- cell_dtypes(args, expected), do: Recorder.record(op, dtype, :compiler)
    end

    :ok
  end

  @doc "The seed ExUnit runs with; generators derive their keys from it."
  @spec seed() :: non_neg_integer()
  def seed, do: ExUnit.configuration()[:seed] || 0

  # -- Reference -----------------------------------------------------------

  defp reference(fun, args) do
    binary_args = Enum.map(args, &to_binary_backend/1)
    apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), binary_args)
  end

  defp to_binary_backend(%Nx.Tensor{} = t), do: Nx.backend_copy(t, Nx.BinaryBackend)
  defp to_binary_backend(other), do: other

  # -- Comparison ----------------------------------------------------------

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
           "leaf #{i}: integer values differ\n  expected #{inspect(Nx.to_flat_list(e), limit: 20)}\n  actual   #{inspect(Nx.to_flat_list(a), limit: 20)}"
  end

  defp compare_values(%Nx.Tensor{type: type} = e, a, i, opts) do
    tol = Map.fetch!(@default_tol, type)
    rtol = Keyword.get(opts, :rtol, tol)
    atol = Keyword.get(opts, :atol, tol)
    signed_zero = Keyword.get(opts, :signed_zero, false)

    # NaN positions must match before any value is compared.
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

  # -- Recorded cells ------------------------------------------------------

  defp cell_dtypes(args, expected) do
    input_types = for %Nx.Tensor{type: type} <- args, do: type

    output_types =
      for %Nx.Tensor{type: type} <- Nx.Defn.Composite.flatten_list([expected]), do: type

    Enum.uniq(input_types ++ output_types)
  end

  # -- Device capabilities -------------------------------------------------

  @doc """
  Extension string of device 0. Tests override it with
  `Application.put_env(:nx_opencl, :parity_extensions, "...")`.
  """
  @spec device_extensions() :: String.t()
  def device_extensions do
    case Application.fetch_env(:nx_opencl, :parity_extensions) do
      {:ok, ext} -> ext
      :error -> device_info()["extensions"] || ""
    end
  end

  @doc "True when the device has the extension for `capability` (`:fp64`, `:fp16` or `:int64_atomics`)."
  @spec capability?(:fp64 | :fp16 | :int64_atomics) :: boolean()
  def capability?(capability) do
    device_extensions()
    |> String.split(" ", trim: true)
    |> Enum.member?(@capabilities[capability])
  end

  @doc "Message for a skipped capability, or `nil` when the device has it."
  @spec skip_reason(:fp64 | :fp16 | :int64_atomics) :: String.t() | nil
  def skip_reason(capability) do
    unless capability?(capability) do
      "skipped: device #{device_name()} lacks #{@capabilities[capability]}"
    end
  end

  @doc "ExUnit exclude filters for every capability the device lacks."
  @spec unsupported_filters() :: [{:requires, :fp64 | :fp16 | :int64_atomics}]
  def unsupported_filters do
    for cap <- Map.keys(@capabilities), skip_reason(cap), do: {:requires, cap}
  end

  @doc "Name of device 0, or `\"unknown\"` when no device opens."
  @spec device_name() :: String.t()
  def device_name, do: device_info()["name"] || "unknown"

  defp device_info do
    case :persistent_term.get({__MODULE__, :device_info}, nil) do
      nil ->
        info = fetch_device_info()
        :persistent_term.put({__MODULE__, :device_info}, info)
        info

      info ->
        info
    end
  end

  defp fetch_device_info do
    case NxCL.Native.device_ctx_create(0) do
      {:error, _} -> %{}
      ctx -> ctx |> NxCL.Native.device_info() |> Map.new()
    end
  rescue
    _ -> %{}
  end
end

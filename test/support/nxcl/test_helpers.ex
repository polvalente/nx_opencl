defmodule NxCL.TestHelpers do
  @moduledoc false

  @spec floats_to_binary([float()]) :: binary()
  def floats_to_binary(floats) when is_list(floats) do
    floats
    |> Enum.map(fn f -> <<f::float-32-native>> end)
    |> IO.iodata_to_binary()
  end

  @spec binary_to_floats(binary()) :: [float()]
  def binary_to_floats(binary) when is_binary(binary) do
    for <<f::float-32-native <- binary>>, do: f
  end

  @spec assert_floats_close([float()], [float()], float()) :: :ok
  def assert_floats_close(actual, expected, atol \\ 1.0e-5) do
    if length(actual) != length(expected) do
      raise ExUnit.AssertionError,
        message:
          "Float lists differ in length: #{length(actual)} actual, #{length(expected)} expected"
    end

    for {a, e} <- Enum.zip(actual, expected) do
      diff = abs(a - e)

      unless diff <= atol do
        raise ExUnit.AssertionError,
          message: """
          Floats not close enough.
          Expected: #{inspect(e)}
          Actual:   #{inspect(a)}
          Diff:     #{diff}
          Tolerance: #{atol}
          """
      end
    end

    :ok
  end

  @spec assert_tensors_close(Nx.Tensor.t(), Nx.Tensor.t(), float()) :: :ok
  def assert_tensors_close(actual, expected, atol \\ 1.0e-5) do
    diff = Nx.subtract(actual, expected) |> Nx.abs()
    max_diff = diff |> Nx.reduce_max() |> Nx.to_number()

    unless max_diff <= atol do
      raise ExUnit.AssertionError,
        message: """
        Tensors not close enough.
        Max absolute error: #{max_diff}
        Tolerance: #{atol}
        """
    end

    :ok
  end
end

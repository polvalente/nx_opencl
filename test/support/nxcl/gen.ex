defmodule NxCL.TestSupport.Gen do
  @moduledoc """
  Seeded input generators for `NxCL.TestSupport.Compare`. The seed is ExUnit's seed, which
  `Compare` prints on failure.
  """

  @shapes [
    {},
    {1},
    {63},
    {65},
    {1000},
    {4097},
    {3, 5},
    {2, 3, 4},
    {2, 1, 3, 1, 2}
  ]

  @dtypes for kind <- [:s, :u, :f, :bf, :c, :f8_e4m3fn],
              size <- [2, 4, 8, 16, 32, 64, 128],
              (try do
                 Nx.Type.normalize!({kind, size}) == {kind, size}
               rescue
                 _ -> false
               end),
              do: {kind, size}

  @doc "Shapes the generators cover: scalar, one element, off-by-one around 64, large, ranks 2, 3, and 5. Nx 1.0 rejects zero-size dimensions, so `{0}` is absent."
  @spec shapes() :: [tuple()]
  def shapes, do: @shapes

  @doc "Every dtype the pinned Nx accepts, by kind and size."
  @spec dtypes() :: [Nx.Type.t()]
  def dtypes, do: @dtypes

  @doc "Random tensor on `Nx.BinaryBackend`. Floats lie in [-4, 4]; integers span a small range."
  @spec random(tuple(), Nx.Type.t(), keyword()) :: Nx.Tensor.t()
  def random(shape, type, opts \\ []) do
    key = Nx.Random.key(Keyword.get(opts, :seed, NxCL.TestSupport.Compare.seed()))
    {tensor, _key} = do_random(key, shape, type)
    tensor
  end

  defp do_random(key, shape, {:c, size}) do
    part = {:f, div(size, 2)}
    {re, key} = do_random(key, shape, part)
    {im, key} = do_random(key, shape, part)
    {Nx.as_type(Nx.complex(Nx.as_type(re, :f32), Nx.as_type(im, :f32)), {:c, size}), key}
  end

  defp do_random(key, shape, {kind, _} = type) when kind in [:f, :bf, :f8_e4m3fn] do
    {t, key} = Nx.Random.uniform(key, -4.0, 4.0, shape: shape, type: :f32)
    {Nx.as_type(t, type), key}
  end

  defp do_random(key, shape, {:s, _} = type) do
    {t, key} = Nx.Random.randint(key, -8, 8, shape: shape, type: :s32)
    {Nx.as_type(t, type), key}
  end

  defp do_random(key, shape, {:u, _} = type) do
    {t, key} = Nx.Random.randint(key, 0, 8, shape: shape, type: :s32)
    {Nx.as_type(t, type), key}
  end

  @doc """
  Edge values for `type` as a 1-D tensor: zeros, signed zero, NaN, infinities
  for floats; minimum and maximum for integers.
  """
  @spec edge_values(Nx.Type.t()) :: Nx.Tensor.t()
  def edge_values({kind, _} = type) when kind in [:f, :bf, :f8_e4m3fn] do
    Nx.tensor([0.0, -0.0, 1.0, -1.0, :nan, :infinity, :neg_infinity], type: type)
  end

  def edge_values({kind, _} = type) when kind in [:s, :u] do
    Nx.tensor(
      [
        0,
        1,
        Nx.to_number(Nx.Constants.min_finite(type)),
        Nx.to_number(Nx.Constants.max_finite(type))
      ],
      type: type
    )
  end

  @doc """
  Values just past the range of `type`, held in `s64` (or `f64` for floats),
  so a cast to `type` overflows. `f64` has no wider type, so it is absent.
  """
  @spec overflow_values(Nx.Type.t()) :: Nx.Tensor.t()
  def overflow_values({:s, _} = type), do: past_range(type)
  def overflow_values({:u, _} = type), do: past_range(type)

  def overflow_values({kind, size} = type) when kind in [:f, :bf, :f8_e4m3fn] and size < 64 do
    max = type |> Nx.Constants.max_finite() |> Nx.to_number()
    Nx.tensor([max * 2.0, -max * 2.0], type: :f64)
  end

  defp past_range(type) do
    max = Nx.to_number(Nx.Constants.max_finite(type))
    min = Nx.to_number(Nx.Constants.min_finite(type))
    Nx.tensor([max + 1, min - 1, max * 2], type: :s64)
  end
end

defmodule NxCL.FFTTest do
  @moduledoc """
  Differential rows for `fft`, `ifft` and the `fft2`, `ifft2`, `rfft` and `irfft`
  blocks.

  Lengths 2, 8, 100, 1024 and 4097 cover the radix-2 passes and Bluestein. Float
  inputs scale by `1 / sqrt(n)` for `fft` and `sqrt(n)` for `ifft`, so the output
  has unit-size entries and the harness tolerance applies (the absolute error of an
  `f32` transform grows with the size of its output). Integer inputs cannot scale,
  so they stop at length 8: at length 100 an `f32` transform of integers up to 8 is
  off by about 2e-5, past the 1e-5 tolerance.

  `Nx.BinaryBackend` is wrong in two cases, so those rows state the answer. At
  length 1 it returns an empty tensor; the answer is the input as a complex tensor.
  For axis 0 of a rank 3 tensor with a new `:length`, it differs from the transform
  of the same axis moved last; the answer is that last-axis transform. For `f64`
  and `c128` the reference is also too inaccurate at length 4097 (its chirp loses
  digits: error 7e-13 against the exact sum, past the 1.0e-12 tolerance), so those
  types check 4097 against an impulse, whose spectrum is known. Round trips compare
  `ifft(fft(x))` with `x`.
  """
  use ExUnit.Case, async: false

  import NxCL.TestSupport.Compare

  alias NxCL.TestSupport.Gen

  @moduletag :opencl
  @moduletag timeout: 600_000

  @types [
    {:u, 8},
    {:u, 16},
    {:u, 32},
    {:u, 64},
    {:s, 8},
    {:s, 16},
    {:s, 32},
    {:s, 64},
    {:f, 16},
    {:bf, 16},
    {:f, 32},
    {:f, 64},
    {:c, 64},
    {:c, 128}
  ]

  @needs %{{:f, 64} => :fp64, {:c, 128} => :fp64, {:f, 16} => :fp16}

  defp integer?({kind, _}), do: kind in [:s, :u]
  defp complex?({kind, _}), do: kind == :c

  # Lengths a type takes: integers cannot scale their inputs, and the double
  # types leave 4097 to the impulse rows.
  defp lengths(type) do
    cond do
      integer?(type) -> [2, 8]
      type in [{:f, 64}, {:c, 128}] -> [2, 8, 100, 1024]
      true -> [2, 8, 100, 1024, 4097]
    end
  end

  defp input(type, shape, scale) do
    t = Gen.random(shape, type)
    if integer?(type), do: t, else: Nx.multiply(t, scale) |> Nx.as_type(type)
  end

  # The transform of `axis` as the reference computes it on the last axis.
  defp on_last_axis(x, axis, fun) do
    last = Nx.rank(x) - 1
    perm = for a <- 0..last, do: Map.get(%{axis => last, last => axis}, a, a)
    x |> Nx.transpose(axes: perm) |> fun.() |> Nx.transpose(axes: perm)
  end

  describe "fft and ifft" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for n <- lengths(type) do
          x = input(type, {2, n}, 1 / :math.sqrt(n))
          assert_same(&Nx.fft/1, [x])
          y = input(type, {n}, :math.sqrt(n))
          assert_same(&Nx.ifft/1, [y])
        end

        # Length 1: the reference returns an empty tensor, so state the answer.
        x = Gen.random({3, 1}, type)
        complex = Nx.as_type(x, Nx.Type.to_complex(type))
        assert_expected(&Nx.fft/1, [x], complex)
        assert_expected(&Nx.ifft/1, [x], complex)
      end

      @tag requires: Map.get(@needs, type)
      test "options on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = input(type, {3, 6, 4}, 0.4)

        # Pad, crop, a middle axis, and a power-of-two length.
        for opts <- [[axis: 1], [length: 3], [axis: 1, length: :power_of_two]] do
          assert_same(&Nx.fft(&1, opts), [x])
          assert_same(&Nx.ifft(&1, opts), [x])
        end

        # Axis 0 with a new length: the reference is wrong, so state the answer.
        for kind <- [:fft, :ifft] do
          expected =
            on_last_axis(x, 0, fn t -> apply(Nx, kind, [t, [axis: -1, length: 5]]) end)

          assert_expected(&apply(Nx, kind, [&1, [axis: 0, length: 5]]), [x], expected)
        end
      end
    end

    # An impulse at `p` transforms to `exp(-2 pi i p k / n)`, with the angle reduced in
    # integers first; `ifft` of it gives `exp(+2 pi i p k / n) / n`.
    for type <- [{:f, 64}, {:c, 128}] do
      @tag requires: :fp64
      test "an impulse at length 4097 on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        n = 4097
        positions = [0, 1234, 4096]
        k = Nx.iota({n}, type: :s64)

        spectrum = fn p, sign ->
          angle =
            Nx.remainder(Nx.multiply(k, p), n)
            |> Nx.as_type(:f64)
            |> Nx.multiply(Nx.tensor(sign * 2 * :math.pi() / n, type: :f64))

          Nx.complex(Nx.cos(angle), Nx.sin(angle))
        end

        impulses =
          for p <- positions do
            Nx.indexed_put(
              Nx.broadcast(Nx.tensor(0.0, type: type), {n}),
              Nx.tensor([[p]]),
              Nx.tensor([1.0], type: type)
            )
          end

        x = Nx.stack(impulses)
        fft = Nx.stack(for p <- positions, do: spectrum.(p, -1))

        ifft =
          Nx.stack(for p <- positions, do: Nx.divide(spectrum.(p, 1), Nx.tensor(n, type: :f64)))

        assert_expected(&Nx.fft/1, [x], fft)
        assert_expected(&Nx.ifft/1, [x], ifft)
      end
    end
  end

  describe "round trip" do
    for type <- [{:c, 64}, {:c, 128}] do
      @tag requires: Map.get(@needs, type)
      test "ifft(fft(x)) is x on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))

        for n <- [1, 2, 8, 100, 1024, 4097], shape <- [{n}, {3, n}] do
          x = Gen.random(shape, type)
          axis = tuple_size(shape) - 1
          round_trip = fn x -> x |> Nx.fft(axis: axis) |> Nx.ifft(axis: axis) end
          assert_expected(round_trip, [x], x)
        end
      end
    end
  end

  describe "fft2, ifft2, rfft and irfft" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        shapes = if integer?(type), do: [{4, 8}, {2, 3, 4}], else: [{8, 8}, {10, 12}, {2, 5, 7}]

        for shape <- shapes do
          x = input(type, shape, 0.2)
          assert_same(&Nx.fft2/1, [x])
          assert_same(&Nx.ifft2/1, [x])
          assert_same(&Nx.fft2(&1, lengths: [4, :power_of_two]), [x])

          if tuple_size(shape) == 2 do
            assert_same(&Nx.ifft2(&1, axes: [0, 1], lengths: [3, 5]), [x])
          else
            # Axes 0 and 1 of a rank 3 tensor: the reference is wrong, so state the answer.
            perm = [2, 0, 1]

            expected =
              x
              |> Nx.transpose(axes: perm)
              |> Nx.ifft2(lengths: [3, 5])
              |> Nx.transpose(axes: [1, 2, 0])

            assert_expected(
              &Nx.ifft2(&1, axes: [0, 1], lengths: [3, 5]),
              [x],
              expected
            )
          end
        end

        # `rfft` takes a real input; Nx rejects complex.
        unless complex?(type) do
          # Integer inputs cannot scale, so they stop at length 8 (see the moduledoc).
          for n <- if(integer?(type), do: [4, 7, 8], else: [4, 7, 8, 100]) do
            x = input(type, {2, n}, 0.3)
            assert_same(&Nx.rfft/1, [x])
            assert_same(&Nx.rfft(&1, length: 16), [x])
          end
        end

        # `irfft` mirrors the half spectrum: even and odd lengths.
        for m <- [3, 5, 9] do
          x = input(type, {2, m}, 0.3)
          assert_same(&Nx.irfft/1, [x])
          assert_same(&Nx.irfft(&1, length: 2 * m - 1), [x])
        end
      end
    end
  end
end

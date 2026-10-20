defmodule NxCL.ConvTest do
  @moduledoc """
  Differential rows for `conv`. The option grid runs on `f32`, `f64`, `c64` and
  `c128` over 1, 2 and 3 spatial axes: each option alone (strides, padding forms,
  dilations, feature groups, depthwise, batch groups, the three permutations),
  then combined. `f16` and `bf16` run a shorter grid; integers run the basic rows.

  Float and complex inputs hold small integers, so every sum is exact. A second
  set of rows uses random floats within a tolerance.
  """
  use ExUnit.Case, async: false

  import NxCL.TestSupport.Compare

  alias NxCL.TestSupport.Gen

  @moduletag :opencl
  @moduletag timeout: 900_000

  @needs %{{:f, 64} => :fp64, {:c, 128} => :fp64, {:f, 16} => :fp16}

  @int_types [{:u, 8}, {:u, 16}, {:u, 32}, {:u, 64}, {:s, 8}, {:s, 16}, {:s, 32}, {:s, 64}]

  defp exact({:c, size}, shape, seed) do
    part = {:f, div(size, 2)}
    re = exact(part, shape, seed)
    im = exact(part, shape, seed + 1)
    Nx.as_type(Nx.complex(Nx.as_type(re, :f32), Nx.as_type(im, :f32)), {:c, size})
  end

  defp exact({kind, _} = type, shape, seed) do
    key = Nx.Random.key(NxCL.TestSupport.Compare.seed() + seed)
    lo = if kind == :u, do: 0, else: -2
    {t, _} = Nx.Random.randint(key, lo, 3, shape: shape, type: :s32)
    Nx.as_type(t, type)
  end

  defp pair(type, xs, ys), do: [exact(type, xs, 1), exact(type, ys, 7)]

  defp check(type, xs, ks, opts) do
    assert_same(&Nx.conv(&1, &2, opts), pair(type, xs, ks))
  end

  # Shapes: input `{n, c, spatial...}`, kernel `{o, i, spatial...}`.
  defp base(1), do: {{2, 3, 11}, {4, 3, 3}}
  defp base(2), do: {{2, 3, 7, 6}, {4, 3, 3, 2}}
  defp base(3), do: {{1, 2, 5, 6, 4}, {3, 2, 2, 3, 2}}

  defp grid(rank) do
    [
      {"defaults", []},
      {"strides", [strides: Enum.take([2, 3, 2], rank)]},
      {"valid padding", [padding: :valid]},
      {"same padding", [padding: :same]},
      {"explicit padding", [padding: Enum.take([{1, 2}, {2, 0}, {0, 1}], rank)]},
      {"negative padding", [padding: Enum.take([{-1, 1}, {0, -1}, {1, 0}], rank)]},
      {"input dilation", [input_dilation: Enum.take([2, 1, 2], rank), padding: :valid]},
      {"kernel dilation", [kernel_dilation: Enum.take([2, 2, 1], rank), padding: :valid]},
      {"all options",
       [
         strides: Enum.take([2, 1, 2], rank),
         padding: Enum.take([{1, 1}, {0, 2}, {1, 0}], rank),
         input_dilation: Enum.take([2, 1, 1], rank),
         kernel_dilation: Enum.take([1, 2, 1], rank)
       ]}
    ]
  end

  defp short_grid(rank),
    do:
      Enum.filter(
        grid(rank),
        &(elem(&1, 0) in ["defaults", "strides", "explicit padding", "all options"])
      )

  defp run_grid(type, grid, ranks) do
    for rank <- ranks, {_name, opts} <- grid.(rank) do
      {xs, ks} = base(rank)
      check(type, xs, ks, opts)
    end
  end

  # Groups need their own channel counts.
  defp run_groups(type) do
    check(type, {2, 4, 7, 6}, {6, 2, 3, 2}, feature_group_size: 2)
    check(type, {2, 4, 9}, {4, 2, 3}, feature_group_size: 2, padding: :same)
    check(type, {2, 3, 7, 6}, {3, 1, 3, 2}, feature_group_size: 3)
    check(type, {2, 3, 7, 6}, {6, 1, 3, 2}, feature_group_size: 3, strides: [2, 1])
    check(type, {1, 3, 5, 6, 4}, {3, 1, 2, 3, 2}, feature_group_size: 3, padding: :same)
    check(type, {4, 3, 7, 6}, {4, 3, 3, 2}, batch_group_size: 2)
    check(type, {4, 2, 9}, {4, 2, 3}, batch_group_size: 2, padding: [{1, 1}])
  end

  # NHWC input, HWIO kernel, NHWC output, as TensorFlow lays them out.
  defp run_permutations(type) do
    opts = [
      input_permutation: [0, 3, 1, 2],
      kernel_permutation: [3, 2, 0, 1],
      output_permutation: [0, 3, 1, 2],
      padding: :same
    ]

    check(type, {2, 7, 6, 3}, {3, 2, 3, 4}, opts)
    check(type, {2, 7, 6, 4}, {3, 2, 2, 6}, [feature_group_size: 2] ++ opts)
    check(type, {2, 3, 7, 6}, {4, 3, 3, 2}, output_permutation: [3, 0, 1, 2])

    check(type, {2, 9, 3}, {3, 3, 4},
      input_permutation: [0, 2, 1],
      kernel_permutation: [2, 1, 0],
      output_permutation: [0, 2, 1]
    )
  end

  for type <- [{:f, 32}, {:f, 64}, {:c, 64}, {:c, 128}] do
    describe "conv on #{Nx.Type.to_string(type)}" do
      for rank <- 1..3 do
        @tag requires: Map.get(@needs, type)
        test "each option and all together, #{rank} spatial axes" do
          type = unquote(Macro.escape(type))
          rank = unquote(rank)
          run_grid(type, &grid/1, [rank])
        end
      end

      @tag requires: Map.get(@needs, type)
      test "feature groups, depthwise, batch groups" do
        run_groups(unquote(Macro.escape(type)))
      end

      @tag requires: Map.get(@needs, type)
      test "layout permutations" do
        run_permutations(unquote(Macro.escape(type)))
      end

      @tag requires: Map.get(@needs, type)
      test "channels and sizes that span several tiles, and a size-1 kernel" do
        type = unquote(Macro.escape(type))
        check(type, {2, 8, 20, 20}, {16, 8, 3, 3}, padding: :same)
        check(type, {1, 5, 33}, {9, 5, 1}, [])
        check(type, {3, 16, 12, 12}, {1, 16, 5, 5}, [])
      end
    end
  end

  for type <- [{:f, 16}, {:bf, 16}] do
    describe "conv on #{Nx.Type.to_string(type)}" do
      @tag requires: Map.get(@needs, type)
      test "a short option grid, groups and permutations" do
        type = unquote(Macro.escape(type))
        run_grid(type, &short_grid/1, [1, 2, 3])
        run_groups(type)
        run_permutations(type)
      end
    end
  end

  for type <- @int_types do
    describe "conv on #{Nx.Type.to_string(type)} input" do
      test "the output is floating, as Nx makes it" do
        type = unquote(Macro.escape(type))

        for opts <- [[], [padding: :same, strides: [2, 1]], [feature_group_size: 3]] do
          ks = if opts[:feature_group_size], do: {3, 1, 3, 2}, else: {4, 3, 3, 2}
          check(type, {2, 3, 7, 6}, ks, opts)
        end

        check(type, {2, 3, 11}, {4, 3, 3}, padding: :same)
      end
    end
  end

  describe "conv results" do
    test "random floats stay within the tolerance of the reference" do
      for {xs, ks, opts} <- [
            {{2, 3, 9, 8}, {4, 3, 3, 3}, [padding: :same]},
            {{1, 8, 20, 20}, {8, 8, 3, 3}, [strides: [2, 2]]}
          ] do
        assert_same(
          &Nx.conv(&1, &2, opts),
          [Gen.random(xs, {:f, 32}), Gen.random(ks, {:f, 32})],
          rtol: 1.0e-4,
          atol: 1.0e-4
        )
      end
    end

    test "operands of different types merge as Nx does" do
      x = Gen.random({2, 3, 7, 6}, {:s, 32})
      k = Gen.random({4, 3, 3, 2}, {:f, 32})
      assert_same(&Nx.conv/2, [x, k])
    end

    test "a small CNN forward pass matches the reference" do
      cnn = fn x, w, b, dense ->
        x
        |> Nx.conv(w, padding: :same)
        |> Nx.add(Nx.reshape(b, {1, :auto, 1, 1}))
        |> Nx.max(0)
        |> Nx.window_max({1, 1, 2, 2}, strides: [1, 1, 2, 2])
        |> Nx.reshape({2, :auto})
        |> Nx.dot(dense)
      end

      key = Nx.Random.key(NxCL.TestSupport.Compare.seed())
      {x, key} = Nx.Random.uniform(key, shape: {2, 1, 8, 8})
      {w, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {4, 1, 3, 3})
      {b, key} = Nx.Random.uniform(key, -0.1, 0.1, shape: {4})
      {dense, _} = Nx.Random.uniform(key, -0.5, 0.5, shape: {64, 10})

      assert_same(cnn, [x, w, b, dense], rtol: 1.0e-4, atol: 1.0e-4)
    end
  end
end

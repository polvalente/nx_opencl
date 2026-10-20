defmodule NxCL.ControlTest do
  @moduledoc """
  Differential rows for `cond` and `while` against `Nx.Defn.Evaluator`: one to
  three predicates, every predicate dtype, tuple, map and nested outputs and
  carries, trip counts 0, 1, 10 and 1000 (a counter against a tensor bound gives a
  dynamic loop; a literal bound gives a static one), nesting depths 1 to 3, and a
  body that uses `slice` and `put_slice` at the loop index.
  """
  use ExUnit.Case, async: false

  import Nx.Defn
  import NxCL.TestSupport.Compare

  alias NxCL.TestSupport.Gen

  @moduletag :opencl
  @moduletag timeout: 900_000

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

  @pred_types [
    {:u, 8},
    {:u, 16},
    {:u, 32},
    {:u, 64},
    {:s, 8},
    {:s, 16},
    {:s, 32},
    {:s, 64},
    {:bf, 16},
    {:f, 32},
    {:c, 64}
  ]

  defn cond1(p, x, y) do
    cond do
      p -> x + y
      true -> x - y
    end
  end

  defn cond2(p, q, x, y) do
    cond do
      p -> x + y
      q -> x - y
      true -> x + x
    end
  end

  defn cond3(p, q, r, x, y) do
    cond do
      p -> x + y
      q -> x - y
      r -> y - x
      true -> x + x
    end
  end

  defn cond_tuple(p, x, y) do
    cond do
      p -> {x + y, x - y, Nx.as_type(x, :s32)}
      true -> {x - y, x + y, Nx.as_type(y, :s32) + 1}
    end
  end

  defn cond_map(p, x, y) do
    cond do
      p -> %{a: x + y, b: {x, y}}
      true -> %{a: x - y, b: {y, x}}
    end
  end

  defn cond_inner_pred(a, x, y) do
    cond do
      Nx.sum(a) > 3 -> x + y
      Nx.sum(a) < -3 -> x - y
      true -> x + x
    end
  end

  defn cond_nested(p, q, x, y) do
    cond do
      p ->
        cond do
          q -> x + y
          true -> x - y
        end

      true ->
        x + x
    end
  end

  defp data(type, shape), do: Gen.random(shape, type)

  # A loop adds a value many times. `bf16` rounds differently on the device (to
  # nearest) and in `BinaryBackend` (it truncates), and the gap grows with each add.
  # Whole numbers add exactly in `bf16` and `f16`, so the loop rows compare the loop.
  defp loop_data(type, shape) when type in [{:bf, 16}, {:f, 16}] do
    shape |> Gen.random(type) |> Nx.round()
  end

  defp loop_data(type, shape), do: Gen.random(shape, type)

  defp pred(type, value), do: Nx.tensor(value, type: type)

  describe "cond" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "one, two and three predicates on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = data(type, {5})
        y = data(type, {5})
        [yes, no] = [pred({:s, 32}, 1), pred({:s, 32}, 0)]

        for p <- [yes, no] do
          assert_same(&cond1/3, [p, x, y])

          for q <- [yes, no] do
            assert_same(&cond2/4, [p, q, x, y])

            for r <- [yes, no], do: assert_same(&cond3/5, [p, q, r, x, y])
          end
        end
      end

      @tag requires: Map.get(@needs, type)
      test "tuple, map and nested results on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = data(type, {3, 2})
        y = data(type, {3, 2})

        for p <- [pred({:u, 8}, 1), pred({:u, 8}, 0)] do
          assert_same(&cond_tuple/3, [p, x, y])
          assert_same(&cond_map/3, [p, x, y])

          for q <- [pred({:u, 8}, 1), pred({:u, 8}, 0)],
              do: assert_same(&cond_nested/4, [p, q, x, y])
        end

        a = data({:f, 32}, {4})

        for a <- [a, Nx.negate(Nx.abs(a)) |> Nx.multiply(3), Nx.abs(a) |> Nx.multiply(3)] do
          assert_same(&cond_inner_pred/3, [a, x, y])
        end
      end
    end

    for type <- @pred_types do
      test "predicate of #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = data({:f, 32}, {4})
        y = data({:f, 32}, {4})

        for value <- [0, 1, 3] do
          assert_same(&cond2/4, [pred(type, value), pred(type, 1), x, y])
          assert_same(&cond2/4, [pred(type, 0), pred(type, value), x, y])
        end
      end
    end

    test "a float predicate that is zero, NaN or negative zero" do
      x = data({:f, 32}, {4})
      y = data({:f, 32}, {4})

      for value <- [0.0, -0.0, :nan, 0.5, :infinity] do
        assert_same(&cond1/3, [Nx.tensor(value, type: :f32), x, y])
      end
    end
  end

  defn while_scalar(x, n) do
    {_i, acc, _x, _n} =
      while {i = 0, acc = x, x, n}, i < n do
        {i + 1, acc + x, x, n}
      end

    acc
  end

  defn while_tuple(x, y, n) do
    {_i, a, b, _x, _y, _n} =
      while {i = 0, a = x, b = Nx.as_type(y, :s32), x, y, n}, i < n do
        {i + 1, a + x, b + 1, x, y, n}
      end

    {a, b}
  end

  defn while_map(x, n) do
    {_i, state, _x, _n} =
      while {i = 0, state = %{a: x, b: {x, Nx.as_type(n, :s64)}}, x, n}, i < n do
        {i + 1, %{a: state.a + x, b: {elem(state.b, 0) - x, elem(state.b, 1) + 1}}, x, n}
      end

    state
  end

  defn while_nested_carry(x, n) do
    {_i, {{a, b}, c}, _x, _n} =
      while {i = 0, {{a = x, b = x}, c = Nx.as_type(n, :s32)}, x, n}, i < n do
        {i + 1, {{a + x, b - x}, c + 2}, x, n}
      end

    {a, b, c}
  end

  defn nest2(x, n) do
    {_i, acc, _n} =
      while {i = 0, acc = x, n}, i < n do
        {_j, acc, _n} =
          while {j = 0, acc, n}, j < n do
            {j + 1, acc + 1, n}
          end

        {i + 1, acc, n}
      end

    acc
  end

  defn nest3(x, n) do
    {_i, acc, _n} =
      while {i = 0, acc = x, n}, i < n do
        {_j, acc, _n} =
          while {j = 0, acc, n}, j < n do
            {_k, acc, _n} =
              while {k = 0, acc, n}, k < n do
                {k + 1, acc + 1, n}
              end

            {j + 1, acc, n}
          end

        {i + 1, acc, n}
      end

    acc
  end

  defn while_static(x) do
    {_i, acc, _x} =
      while {i = 0, acc = x, x}, i < 10 do
        {i + 1, acc + x, x}
      end

    acc
  end

  defn while_range(x) do
    while acc = x, _i <- 0..9 do
      acc + 1
    end
  end

  defn while_range_step(x) do
    while acc = x, _i <- 2..11//3 do
      acc * 2
    end
  end

  defn while_down(x) do
    {_i, acc, _x} =
      while {i = 7, acc = x, x}, i > 0 do
        {i - 2, acc + x, x}
      end

    acc
  end

  defn fill(x, n) do
    {_i, acc, _n} =
      while {i = 0, acc = x, n}, i < n do
        v = Nx.slice(acc, [i], [1]) + 1
        {i + 1, Nx.put_slice(acc, [i], v), n}
      end

    acc
  end

  defn while_with_cond(x, n) do
    {_i, acc, _n} =
      while {i = 0, acc = x, n}, i < n do
        acc =
          cond do
            rem(i, 2) == 0 -> acc + 1
            true -> acc - 2
          end

        {i + 1, acc, n}
      end

    acc
  end

  defn cond_with_while(p, x, n) do
    cond do
      p ->
        {_i, acc, _n} =
          while {i = 0, acc = x, n}, i < n do
            {i + 1, acc + 1, n}
          end

        acc

      true ->
        x
    end
  end

  defn while_data_dependent(x) do
    {acc, _n} =
      while {acc = x, n = 0}, Nx.sum(acc) < 20 and n < 50 do
        {acc + 1, n + 1}
      end

    acc
  end

  describe "while, carries" do
    for type <- @types do
      @tag requires: Map.get(@needs, type)
      test "scalar, tuple, map and nested carries on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = loop_data(type, {4})
        s = loop_data(type, {})

        for trips <- [0, 1, 10] do
          n = pred({:s, 32}, trips)
          assert_same(&while_scalar/2, [x, n])
          assert_same(&while_scalar/2, [s, n])
          assert_same(&while_tuple/3, [x, loop_data(type, {2, 2}), n])
          assert_same(&while_map/2, [x, n])
          assert_same(&while_nested_carry/2, [x, n])
        end
      end

      @tag requires: Map.get(@needs, type)
      test "static loops and a loop that fills a slice on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = loop_data(type, {3})

        assert_same(&while_static/1, [x])
        assert_same(&while_range/1, [x])
        assert_same(&while_range_step/1, [x])
        assert_same(&while_down/1, [x])

        for trips <- [0, 3, 8] do
          assert_same(&fill/2, [loop_data(type, {8}), pred({:s, 32}, trips)])
        end
      end

      @tag requires: Map.get(@needs, type)
      test "control flow inside control flow on #{Nx.Type.to_string(type)}" do
        type = unquote(Macro.escape(type))
        x = loop_data(type, {3})

        for trips <- [0, 1, 4] do
          n = pred({:s, 32}, trips)
          assert_same(&while_with_cond/2, [x, n])
          assert_same(&nest2/2, [x, n])
          assert_same(&cond_with_while/3, [pred({:u, 8}, 1), x, n])
          assert_same(&cond_with_while/3, [pred({:u, 8}, 0), x, n])
        end

        assert_same(&nest3/2, [x, pred({:s, 32}, 3)])
      end
    end

    test "trip counts 0, 1, 10 and 1000" do
      for type <- [{:s, 32}, {:f, 32}, {:u, 8}], trips <- [0, 1, 10, 1000] do
        assert_same(&while_scalar/2, [loop_data(type, {3}), pred({:s, 32}, trips)])
        assert_same(&fill/2, [loop_data(type, {8}), pred({:s, 32}, min(trips, 8))])
      end
    end

    test "nesting depths 1, 2 and 3" do
      x = loop_data({:f, 32}, {3})

      assert_same(&while_scalar/2, [x, pred({:s, 32}, 4)])
      assert_same(&nest2/2, [x, pred({:s, 32}, 4)])
      assert_same(&nest3/2, [x, pred({:s, 32}, 4)])
      assert_same(&nest3/2, [x, pred({:s, 32}, 0)])
    end

    test "a condition that reads the carry" do
      for start <- [0, 3, 30] do
        assert_same(&while_data_dependent/1, [Nx.tensor([start, 1], type: :f32)])
      end
    end
  end
end

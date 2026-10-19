defmodule NxCL.DeviceArgsTest do
  @moduledoc """
  A compiled function that takes tensors already on the device uses their buffers
  as they are.
  """
  use ExUnit.Case, async: false

  import Nx.Defn
  import NxCL.TestSupport, only: [run_jit: 1]

  @moduletag :opencl

  defp on_device(t), do: Nx.backend_copy(t, NxCL.Backend)

  defp dense(x, w, b), do: x |> Nx.dot(w) |> Nx.add(b) |> Nx.max(0)

  defn counting(x) do
    {_i, acc} =
      while {i = 0, acc = x}, i < 5 do
        {i + 1, acc * 2.0 + 1.0}
      end

    acc
  end

  test "device tensors give the same result as host tensors" do
    x = Nx.iota({4, 3}, type: :f32)
    w = Nx.iota({3, 5}, type: :f32) |> Nx.divide(7)
    b = Nx.iota({5}, type: :f32)

    host = run_jit(&dense/3).(x, w, b)
    device = run_jit(&dense/3).(on_device(x), on_device(w), on_device(b))

    assert Nx.to_binary(host) == Nx.to_binary(device)
  end

  test "a program with a loop reads device tensors too" do
    x = Nx.iota({6}, type: :f32)

    assert Nx.to_binary(run_jit(&counting/1).(x)) ==
             Nx.to_binary(run_jit(&counting/1).(on_device(x)))
  end

  test "the input tensor stays readable after the call" do
    x = on_device(Nx.iota({8}, type: :f32))
    expected = Nx.to_binary(x)

    for _ <- 1..5, do: run_jit(&Nx.exp/1).(x)

    assert Nx.to_binary(x) == expected
  end
end

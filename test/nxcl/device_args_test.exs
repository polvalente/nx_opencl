defmodule NxCL.DeviceArgsTest do
  @moduledoc """
  A compiled function that takes tensors already on the device uses their buffers
  as they are. Run on one GPU, alone: two suites at once flake.
  """
  use ExUnit.Case, async: false

  import Nx.Defn

  alias NxCL.Devices
  alias NxCL.Native
  alias NxCL.ProgramCache

  @moduletag :opencl

  defp run_jit(fun), do: Nx.Defn.jit(fun, compiler: NxCL.Compiler)
  defp on_device(t), do: Nx.backend_copy(t, NxCL.Backend)

  defp settled_buffers(previous \\ nil) do
    :erlang.garbage_collect()
    Process.sleep(30)
    count = Native.live_buffer_count(Devices.ctx())
    if count == previous, do: count, else: settled_buffers(count)
  end

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

  test "repeat calls on device tensors leave live_buffer_count unchanged" do
    x = on_device(Nx.iota({256}, type: :f32))
    w = on_device(Nx.iota({256}, type: :f32))
    fun = run_jit(fn a, b -> a |> Nx.multiply(b) |> Nx.sum() end)
    fun.(x, w)

    ProgramCache.clear()
    fun.(x, w)
    before = settled_buffers()

    for _ <- 1..200, do: fun.(x, w)

    assert settled_buffers() == before
    # The arguments are still in use here, so they were not collected above.
    assert Nx.size(x) + Nx.size(w) == 512
  end
end

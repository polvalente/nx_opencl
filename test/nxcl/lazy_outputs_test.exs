defmodule NxCL.LazyOutputsTest do
  @moduledoc """
  A run on the default queue returns before the device is done. These tests check
  that a later read, and a later run on another queue, still see finished data, and
  that a program with several outputs reads them all with one wait.
  Run on one GPU, alone: two suites at once flake.
  """
  use ExUnit.Case, async: false

  alias NxCL.Devices

  @moduletag :opencl

  defp heavy(x), do: x |> Nx.exp() |> Nx.sin() |> Nx.exp() |> Nx.cos()

  defp input(n), do: Nx.iota({n}, type: :f32) |> Nx.divide(n)

  test "a read right after a compiled run sees the result" do
    x = input(1_000_000)
    expected = Nx.Defn.jit(&heavy/1, compiler: Nx.Defn.Evaluator).(x)
    got = Nx.Defn.jit(&heavy/1, compiler: NxCL.Compiler).(x)

    assert Nx.all_close(got, expected, atol: 1.0e-5) |> Nx.to_number() == 1
  end

  test "a run on another queue waits for output still queued on the default queue" do
    {:ok, queue} = Devices.new_queue()
    x = input(1_000_000)
    expected = x |> heavy() |> Nx.add(1)

    for _ <- 1..20 do
      queued = x |> Nx.backend_copy(NxCL.Backend) |> heavy()

      got =
        Nx.Defn.jit(&Nx.add(&1, 1), compiler: NxCL.Compiler, command_queue: queue).(queued)

      assert Nx.all_close(got, expected, atol: 1.0e-5) |> Nx.to_number() == 1
    end
  end

  test "a program with several outputs of mixed types returns each one" do
    fun = fn x ->
      {Nx.sum(x), Nx.multiply(x, 2), Nx.as_type(x, :s32), %{max: Nx.reduce_max(x)}}
    end

    x = Nx.iota({5}, type: :f32)
    expected = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator).(x)
    got = Nx.Defn.jit(fun, compiler: NxCL.Compiler).(x)

    assert got == expected
  end
end

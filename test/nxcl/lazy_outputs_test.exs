defmodule NxCL.LazyOutputsTest do
  @moduledoc """
  A run on the default queue may return before the device is done. These tests check
  values: a later read, and a later run on another queue, see finished data, and a
  program with several outputs returns each one. They cannot tell a lazy run from a
  synchronous one; they fail if a read or a later run sees unfinished data.
  """
  use ExUnit.Case, async: false

  alias NxCL.Devices

  @moduletag :opencl

  defp heavy(x), do: x |> Nx.exp() |> Nx.sin() |> Nx.exp() |> Nx.cos()

  defp input(n), do: Nx.iota({n}, type: :f32) |> Nx.divide(n)

  test "a read right after a compiled run gives the right values" do
    x = input(100_000)
    expected = Nx.Defn.jit(&heavy/1, compiler: Nx.Defn.Evaluator).(x)
    got = Nx.Defn.jit(&heavy/1, compiler: NxCL.Compiler).(x)

    assert Nx.all_close(got, expected, atol: 1.0e-5) |> Nx.to_number() == 1
  end

  test "a run on another queue reads the right values from a default-queue run" do
    {:ok, queue} = Devices.new_queue()
    x = input(100_000)
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

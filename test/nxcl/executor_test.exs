defmodule NxCL.ExecutorTest do
  @moduledoc """
  Argument checks, constant encoding and capture reuse in `NxCL.Compiler.Executor`.
  Run on one GPU, alone: two suites at once flake.
  """
  use ExUnit.Case, async: false

  alias NxCL.Compiler.Executor
  alias NxCL.Compiler.Executor.Wire
  alias NxCL.Compiler.Lower

  @moduletag :opencl

  defp negate_program do
    expr = Nx.Defn.debug_expr(&Nx.negate/1).(Nx.template({2}, :f32))
    Lower.lower(expr, 1)
  end

  test "a wrong argument count raises CompilerError with both counts" do
    error = assert_raise NxCL.CompilerError, fn -> Executor.run(negate_program(), [], []) end
    assert error.message == "expected 1 arguments, got 0"

    two = [fn -> Nx.tensor([1.0, 2.0]) end, fn -> Nx.tensor([1.0, 2.0]) end]
    error = assert_raise NxCL.CompilerError, fn -> Executor.run(negate_program(), two, []) end
    assert error.message == "expected 1 arguments, got 2"
  end

  test "a constant that cannot be encoded says what is supported" do
    error = assert_raise NxCL.CompilerError, fn -> Wire.const({"one", {:f, 32}}) end
    assert error.message =~ "\"one\""
    assert error.message =~ ":nan"
    assert error.message =~ "Complex"

    assert Wire.const({1.5, {:f, 32}}) == {{:f, 32}, 1.5}
    assert Wire.const({:infinity, {:f, 32}}) == {{:f, 32}, :infinity}
    assert Wire.const({Complex.new(1.0, :nan), {:c, 64}}) == {{:c, 64}, {1.0, :nan}}
  end

  test "captured tensors move to the device once and are reused" do
    capture = Nx.tensor([10.0, 20.0])
    fun = fn x -> Nx.add(x, capture) end
    expr = Nx.Defn.debug_expr(fun).(Nx.template({2}, :f32))
    program = Lower.lower(expr, 1)
    assert [%Nx.Tensor{data: %Nx.BinaryBackend{}}] = program.captures

    resident = Executor.with_resident_captures(program)
    assert [%Nx.Tensor{data: %NxCL.Backend{ref: ref}}] = resident.captures

    # A second call keeps the buffers it has.
    assert Executor.with_resident_captures(resident).captures == resident.captures

    args = [fn -> Nx.tensor([1.0, 2.0]) end]
    assert Executor.run(resident, args, []) |> Nx.to_flat_list() == [11.0, 22.0]
    assert Executor.run(resident, args, []) |> Nx.to_flat_list() == [11.0, 22.0]
    assert [%Nx.Tensor{data: %NxCL.Backend{ref: ^ref}}] = resident.captures
  end
end

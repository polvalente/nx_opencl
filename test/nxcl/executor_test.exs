defmodule NxCL.ExecutorTest do
  @moduledoc """
  Argument checks, constant encoding and capture reuse in `NxCL.Compiler.Executor`.
  """
  use ExUnit.Case, async: false

  alias NxCL.Compiler.Executor
  alias NxCL.Compiler.Executor.Wire
  alias NxCL.Compiler.Lower
  alias NxCL.TestSupport.Calls

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

    assert Executor.with_resident_captures(resident).captures == resident.captures

    args = [fn -> Nx.tensor([1.0, 2.0]) end]
    assert Executor.run(resident, args, []) |> Nx.to_flat_list() == [11.0, 22.0]
    assert Executor.run(resident, args, []) |> Nx.to_flat_list() == [11.0, 22.0]
    assert [%Nx.Tensor{data: %NxCL.Backend{ref: ^ref}}] = resident.captures
  end

  describe "captures of a jitted function" do
    @write {NxCL.Native, :buffer_write, 2}

    test "go to the device on the first call and are not uploaded again" do
      capture = Nx.tensor([10.0, 20.0])
      jitted = Nx.Defn.jit(fn x -> Nx.add(x, capture) end, compiler: NxCL.Compiler)
      # `x` is on the device already, so the only upload is the capture.
      x = Nx.backend_copy(Nx.tensor([1.0, 2.0]), NxCL.Backend)

      {first, %{@write => first_writes}} = Calls.count([@write], fn -> jitted.(x) end)
      {second, %{@write => second_writes}} = Calls.count([@write], fn -> jitted.(x) end)

      assert Nx.to_flat_list(first) == [11.0, 22.0]
      assert Nx.to_flat_list(second) == [11.0, 22.0]
      assert first_writes == 1
      assert second_writes == 0
    end

    test "go to the context the function runs on, once" do
      other = {0, 0}
      capture = Nx.tensor([10.0, 20.0])

      jitted =
        Nx.Defn.jit(fn x -> Nx.add(x, capture) end, compiler: NxCL.Compiler, device: other)

      x = Nx.tensor([1.0, 2.0], backend: {NxCL.Backend, device: other})

      {first, %{@write => first_writes}} = Calls.count([@write], fn -> jitted.(x) end)
      {second, %{@write => second_writes}} = Calls.count([@write], fn -> jitted.(x) end)

      assert Nx.to_flat_list(first) == [11.0, 22.0]
      assert Nx.to_flat_list(second) == [11.0, 22.0]
      assert first_writes == 1
      assert second_writes == 0
    end

    # Nx does not let a closure capture a tensor on `NxCL.Backend`, so a capture that
    # already sits on one context reaches the executor only through `with_resident_captures/2`.
    test "that sit on another context are uploaded again for the run device" do
      other = NxCL.Devices.fetch!(device: {0, 0}).ctx
      capture = Nx.tensor([10.0, 20.0])
      expr = Nx.Defn.debug_expr(fn x -> Nx.add(x, capture) end).(Nx.template({2}, :f32))

      on_default = Executor.with_resident_captures(Lower.lower(expr, 1))
      assert [%Nx.Tensor{data: %NxCL.Backend{ctx: default_ctx}}] = on_default.captures
      assert default_ctx == NxCL.Devices.ctx()

      on_other = Executor.with_resident_captures(on_default, device: {0, 0})
      assert [%Nx.Tensor{data: %NxCL.Backend{ctx: ^other}} = moved] = on_other.captures
      assert Nx.to_flat_list(moved) == [10.0, 20.0]
      assert other != default_ctx

      args = [fn -> Nx.tensor([1.0, 2.0], backend: {NxCL.Backend, device: {0, 0}}) end]
      assert Executor.run(on_other, args, device: {0, 0}) |> Nx.to_flat_list() == [11.0, 22.0]
    end
  end
end

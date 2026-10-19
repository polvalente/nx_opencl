defmodule NxCL.MultiDeviceTest do
  @moduledoc """
  The eager backend runs on the device its operands live on. `device: {0, 0}`
  opens a second context on the same GPU, so the checks need no second device.
  """
  use ExUnit.Case, async: false

  @moduletag :opencl

  @other {NxCL.Backend, device: {0, 0}}

  defp ctx_of(%Nx.Tensor{data: %NxCL.Backend{ctx: ctx}}), do: ctx

  setup_all do
    other = NxCL.Devices.fetch!(device: {0, 0}).ctx
    assert other != NxCL.Devices.ctx()
    %{other: other}
  end

  describe "creation ops take the device option" do
    test "constant, scalar and shaped", %{other: other} do
      assert ctx_of(Nx.tensor(2.0, backend: @other)) == other
      assert ctx_of(Nx.broadcast(Nx.tensor(2.0, backend: @other), {3})) == other

      # A shaped constant goes through the backend callback with the option.
      out = %Nx.Tensor{shape: {2}, type: {:f, 32}, names: [nil], data: nil}
      shaped = NxCL.Backend.constant(out, 3.0, device: {0, 0})
      assert ctx_of(shaped) == other
      assert Nx.to_flat_list(shaped) == [3.0, 3.0]
    end

    test "iota and eye", %{other: other} do
      iota = Nx.iota({4}, backend: @other)
      assert ctx_of(iota) == other
      assert Nx.to_flat_list(iota) == [0, 1, 2, 3]

      eye = Nx.eye(3, backend: @other)
      assert ctx_of(eye) == other
      assert Nx.to_flat_list(eye) == [1, 0, 0, 0, 1, 0, 0, 0, 1]
    end

    test "the default stays on device 0" do
      assert ctx_of(Nx.iota({2}, backend: NxCL.Backend)) == NxCL.Devices.ctx()
    end
  end

  describe "operands on one device" do
    test "a result stays on the operands' device", %{other: other} do
      a = Nx.tensor([1.0, 2.0, 3.0], backend: @other)
      result = Nx.add(a, 1.0)
      assert ctx_of(result) == other
      assert Nx.to_flat_list(result) == [2.0, 3.0, 4.0]
    end

    test "a host tensor goes up to the other operand's device", %{other: other} do
      a = Nx.tensor([1.0, 2.0], backend: @other)
      result = Nx.add(a, Nx.tensor([10.0, 20.0], backend: Nx.BinaryBackend))
      assert ctx_of(result) == other
      assert Nx.to_flat_list(result) == [11.0, 22.0]
    end

    test "integer slice starts go up to the operand's device", %{other: other} do
      a = Nx.iota({5}, backend: @other)
      result = Nx.slice(a, [1], [3])
      assert ctx_of(result) == other
      assert Nx.to_flat_list(result) == [1, 2, 3]
    end

    test "a block's default builds its constants on the operand's device", %{other: other} do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: @other)
      result = Nx.LinAlg.determinant(a)
      assert ctx_of(result) == other
      assert_in_delta Nx.to_number(result), -2.0, 1.0e-4
    end

    test "batches go back to the tensor's device", %{other: other} do
      a = Nx.iota({4, 2}, backend: @other)
      assert [first, _] = a |> Nx.to_batched(2) |> Enum.to_list()
      assert ctx_of(first) == other
    end
  end

  describe "operands on different contexts" do
    test "raise ArgumentError that names the fix" do
      a = Nx.tensor([1.0, 2.0], backend: NxCL.Backend)
      b = Nx.tensor([1.0, 2.0], backend: @other)

      error = assert_raise ArgumentError, fn -> Nx.add(a, b) end
      assert error.message =~ "different OpenCL contexts"
      assert error.message =~ "Nx.backend_transfer"
    end

    test "raise for a slice start on another context" do
      a = Nx.iota({5}, backend: NxCL.Backend)
      start = Nx.tensor(1, backend: @other)

      assert_raise ArgumentError, ~r/different OpenCL contexts/, fn ->
        Nx.slice(a, [start], [2])
      end
    end

    test "moving one operand makes the call work", %{other: other} do
      a = Nx.tensor([1.0, 2.0], backend: NxCL.Backend)
      b = Nx.tensor([3.0, 4.0], backend: @other)
      moved = Nx.backend_transfer(a, @other)

      result = Nx.add(moved, b)
      assert ctx_of(result) == other
      assert Nx.to_flat_list(result) == [4.0, 6.0]
    end
  end

  describe "NxCL.Compiler.Evaluator.device_opts/2" do
    test "an operand on the device named by the option is accepted" do
      a = Nx.tensor([1.0], backend: @other)
      assert NxCL.Compiler.Evaluator.device_opts([a], device: {0, 0}) == [device: {0, 0}]
    end

    test "an operand on another device than the option names raises" do
      a = Nx.tensor([1.0], backend: @other)

      assert_raise ArgumentError, ~r/not on the requested device/, fn ->
        NxCL.Compiler.Evaluator.device_opts([a], device: 0)
      end
    end
  end
end

defmodule NxCL.DeviceCtxTest do
  @moduledoc """
  `NxCL.Native` device context creation and device info.
  """

  use ExUnit.Case, async: false

  @moduletag :opencl

  describe "device_ctx_create/1" do
    test "creates a device context for device 0" do
      ctx = NxCL.Native.device_ctx_create(0)
      assert is_reference(ctx)
    end

    test "returns error tuple for invalid device index" do
      assert {:error, msg} = NxCL.Native.device_ctx_create(999)
      assert msg =~ "out of range"
    end
  end

  describe "device_info/1" do
    test "returns device information" do
      ctx = NxCL.Native.device_ctx_create(0)
      info = NxCL.Native.device_info(ctx)

      info_map = Map.new(info)

      assert Map.has_key?(info_map, "name")
      assert Map.has_key?(info_map, "vendor")
      assert Map.has_key?(info_map, "version")
      assert Map.has_key?(info_map, "driver_version")
      assert info_map["device_type"] in ["GPU", "CPU", "ACCELERATOR", "OTHER"]
      assert Map.has_key?(info_map, "max_workgroup_size")
      assert Map.has_key?(info_map, "max_compute_units")
      assert Map.has_key?(info_map, "global_mem_size")

      assert byte_size(info_map["name"]) > 0
    end
  end
end

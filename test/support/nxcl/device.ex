defmodule NxCL.TestSupport.Device do
  @moduledoc """
  What device 0 can do. Tests tagged `requires: :fp64 | :fp16 | :int64_atomics`
  are excluded when the device lacks the extension.
  """

  @capabilities %{
    fp64: "cl_khr_fp64",
    fp16: "cl_khr_fp16",
    int64_atomics: "cl_khr_int64_base_atomics"
  }

  @doc """
  Extension string of device 0. Tests override it with
  `Application.put_env(:nx_opencl, :test_extensions, "...")`.
  """
  @spec extensions() :: String.t()
  def extensions do
    case Application.fetch_env(:nx_opencl, :test_extensions) do
      {:ok, ext} -> ext
      :error -> device_info()["extensions"] || ""
    end
  end

  @doc "True when the device has the extension for `capability` (`:fp64`, `:fp16` or `:int64_atomics`)."
  @spec capability?(:fp64 | :fp16 | :int64_atomics) :: boolean()
  def capability?(capability) do
    extensions()
    |> String.split(" ", trim: true)
    |> Enum.member?(@capabilities[capability])
  end

  @doc "Message for a skipped capability, or `nil` when the device has it."
  @spec skip_reason(:fp64 | :fp16 | :int64_atomics) :: String.t() | nil
  def skip_reason(capability) do
    unless capability?(capability) do
      "skipped: device #{name()} lacks #{@capabilities[capability]}"
    end
  end

  @doc "ExUnit exclude filters for every capability the device lacks."
  @spec unsupported_filters() :: [{:requires, :fp64 | :fp16 | :int64_atomics}]
  def unsupported_filters do
    for cap <- Map.keys(@capabilities), skip_reason(cap), do: {:requires, cap}
  end

  @doc "Name of device 0, or `\"unknown\"` when no device opens."
  @spec name() :: String.t()
  def name, do: device_info()["name"] || "unknown"

  defp device_info do
    case :persistent_term.get({__MODULE__, :device_info}, nil) do
      nil ->
        info = fetch_device_info()
        :persistent_term.put({__MODULE__, :device_info}, info)
        info

      info ->
        info
    end
  end

  defp fetch_device_info do
    case NxCL.Native.device_ctx_create(0) do
      {:error, _} -> %{}
      ctx -> ctx |> NxCL.Native.device_info() |> Map.new()
    end
  rescue
    _ -> %{}
  end
end

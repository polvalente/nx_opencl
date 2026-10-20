defmodule NxCL.Application do
  @moduledoc false
  # Owns the device registry and the caches. Starts with the `:nx_opencl` app.

  use Application

  @impl true
  def start(_type, _args) do
    children = [NxCL.Devices, NxCL.Compiler.ProgramCache, NxCL.Compiler.CompileCache]

    # `:rest_for_one`: when `Devices` restarts, its table is gone and it opens new
    # contexts. The caches hold programs and buffers of the old ones, so they restart
    # after it. Tensors made before the crash keep their old context but no longer map
    # to a device.
    Supervisor.start_link(children, strategy: :rest_for_one, name: NxCL.Supervisor)
  end
end

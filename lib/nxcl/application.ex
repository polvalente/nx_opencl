defmodule NxCL.Application do
  @moduledoc false
  # Owns the device registry and the caches. Starts with the `:nx_opencl` app.

  use Application

  @impl true
  def start(_type, _args) do
    children = [NxCL.Devices, NxCL.ProgramCache, NxCL.CompileCache]
    Supervisor.start_link(children, strategy: :one_for_one, name: NxCL.Supervisor)
  end
end

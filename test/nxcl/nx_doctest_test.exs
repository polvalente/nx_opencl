defmodule NxCL.NxDoctestTest do
  @moduledoc """
  Nx's own doctests with `NxCL.Backend` as the default backend. Every test is
  traced for calls into `Nx.BinaryBackend`: none may run. Functions left out are in
  `NxCL.DoctestSkips`.
  """
  use ExUnit.Case, async: false

  alias NxCL.TestSupport.NoFallback

  @moduletag :opencl
  @moduletag timeout: 600_000

  setup do
    Nx.default_backend(NxCL.Backend)
    collector = NoFallback.begin_trace()
    pid = self()

    on_exit(fn ->
      assert %{binary: []} = NoFallback.end_trace(collector, pid)
    end)

    :ok
  end

  doctest Nx, except: [:moduledoc | NxCL.DoctestSkips.except(Nx)]
end

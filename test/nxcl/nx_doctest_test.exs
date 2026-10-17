defmodule NxCL.NxDoctestTest do
  @moduledoc """
  Nx's own doctests with `NxCL.Backend` as the default backend, after
  `emlx/test/emlx/nx_doctest_test.exs`. Every ran test is traced for calls into
  `Nx.BinaryBackend`: none may run. The functions left out are in
  `NxCL.DoctestSkips`, each with a reason.
  """
  use ExUnit.Case, async: false

  alias NxCL.Parity.Tripwire

  @moduletag :opencl
  @moduletag timeout: 600_000

  setup do
    Nx.default_backend(NxCL.Backend)
    collector = Tripwire.begin_trace()
    pid = self()

    on_exit(fn ->
      assert %{binary: []} = Tripwire.end_trace(collector, pid)
    end)

    :ok
  end

  doctest Nx, except: [:moduledoc | NxCL.DoctestSkips.except(Nx)]
end

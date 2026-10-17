defmodule NxCL.CompilerDoctestTest do
  @moduledoc """
  Doctests of the Nx modules whose functions are `defn`, run through
  `NxCL.Compiler`: `Nx.Defn.default_options/1` names it, so each `defn` call
  compiles and runs on the device. The eager ops inside run on `NxCL.Backend`.
  The functions left out are in `NxCL.DoctestSkips`. No trace here: the default
  compiler also reaches the inner calls of a host function (`svd`), which run on
  `Nx.BinaryBackend` by design and make the trace very slow. `NxCL.NoFallbackTest` checks
  that the compiler runs ops on the device.
  """
  use ExUnit.Case, async: false

  @moduletag :opencl
  @moduletag timeout: 600_000

  setup do
    Nx.default_backend(NxCL.Backend)
    Nx.Defn.default_options(compiler: NxCL.Compiler)
    :ok
  end

  doctest Nx.LinAlg, except: [:moduledoc | NxCL.DoctestSkips.except(Nx.LinAlg)]
  doctest Nx.Random, except: [:moduledoc | NxCL.DoctestSkips.except(Nx.Random)]
  doctest Nx.Constants, except: [:moduledoc | NxCL.DoctestSkips.except(Nx.Constants)]
end

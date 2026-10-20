defmodule NxCL.CompilerDoctestTest do
  @moduledoc """
  Doctests of the Nx modules whose functions are `defn`, run through
  `NxCL.Compiler` (`Nx.Defn.default_options/1` names it). Functions left out are in
  `NxCL.DoctestSkips`. There is no trace here: the default compiler also reaches
  the inner calls of a host function (`svd`), which run on `Nx.BinaryBackend` and
  make the trace very slow. `NxCL.NoFallbackTest` checks that ops run on the device.
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

defmodule NxOpencl do
  @moduledoc """
  Nx backend and `defn` compiler for OpenCL GPUs.

  NxCL runs Nx on OpenCL 1.2+ devices, through two interfaces:

    * `NxCL.Backend` - an eager `Nx.Backend`. Each op runs as a one-op program.
    * `NxCL.Compiler` - an `Nx.Defn.Compiler`. A `defn` runs as one program.

  ## Eager backend

      Nx.default_backend(NxCL.Backend)

      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      b = Nx.tensor([[5.0, 6.0], [7.0, 8.0]])
      Nx.dot(a, b)

  ## Defn compiler

      jitted = Nx.Defn.jit(&MyModel.predict/5, compiler: NxCL.Compiler)
      result = jitted.(input, w1, b1, w2, b2)

  ## Transferring tensors

      gpu_tensor = Nx.backend_transfer(cpu_tensor, NxCL.Backend)
      cpu_tensor = Nx.backend_transfer(gpu_tensor, Nx.BinaryBackend)

  The README lists what works and what does not.
  """
end

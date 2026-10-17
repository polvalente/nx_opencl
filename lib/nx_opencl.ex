defmodule NxOpencl do
  @moduledoc """
  Nx backend and `defn` compiler for OpenCL GPUs.

  NxCL targets OpenCL 1.2+ devices for running inference workloads.
  It provides two complementary interfaces:

    * `NxCL.Backend` — an eager `Nx.Backend` that runs each tensor op as a
      one-op program on the device. An op with no lowering raises; nothing
      runs on `Nx.BinaryBackend` behind your back.

    * `NxCL.Compiler` — an `Nx.Defn.Compiler` that traces a `defn`
      function into a program and runs the whole graph on the device, with
      elementwise fusion and compile caches.

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

  The README lists every op, with its dtypes and what is out of scope. See
  `NxCL.Backend` and `NxCL.Compiler` for details of each path.
  """
end

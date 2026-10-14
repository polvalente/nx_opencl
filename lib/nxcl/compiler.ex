defmodule NxCL.Compiler do
  @moduledoc """
  `Nx.Defn.Compiler` implementation for OpenCL.

  Receives a computation graph from `defn`, lowers it to an `NxCL.Program`
  (`NxCL.Lower`), and runs the program on the GPU.

  Unlike the eager backend (`NxCL.Backend`) which dispatches one op at
  a time, the compiler sees the entire computation graph up front. This
  enables future optimisations like elementwise fusion, GEMM+bias+activation
  fusion, and buffer reuse via liveness analysis.

  ## Usage

  Use with `Nx.Defn.jit/2`:

      defn predict(x, w, b) do
        x |> Nx.dot(w) |> Nx.add(b) |> Nx.sigmoid()
      end

      jitted = Nx.Defn.jit(&predict/3, compiler: NxCL.Compiler)
      result = jitted.(input, weights, bias)

  Or set as the default compiler:

      Nx.Defn.default_options(compiler: NxCL.Compiler)

  Arguments and results can be containers (tuples, maps, nested).

  ## Unsupported operations

  The compiler never falls back to the CPU. An op that `NxCL.Lower` cannot
  lower raises `ArgumentError` ("does not yet lower op ...") when the function
  is compiled. A dtype the device cannot run (`f64` without `cl_khr_fp64`, or a type
  with no OpenCL mapping yet) raises `NxCL.CompilerError` when the function runs.
  """

  @behaviour Nx.Defn.Compiler

  alias Nx.Defn.Composite
  alias NxCL.Executor
  alias NxCL.Lower

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl true
  def __compile__(_key, vars, fun, _opts) do
    expr = fun.(vars)
    num_inputs = vars |> Composite.flatten_list() |> length()
    program = Lower.lower(expr, num_inputs)

    fn [args] ->
      [Executor.run(program, args)]
    end
  end

  @impl true
  def __partitions_options__(opts) do
    [opts]
  end

  @impl true
  def __to_backend__(_opts) do
    {NxCL.Backend, []}
  end
end

defmodule NxCL.CompilerError do
  defexception [:message]
end

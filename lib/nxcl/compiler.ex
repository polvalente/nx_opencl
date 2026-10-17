defmodule NxCL.Compiler do
  @moduledoc """
  `Nx.Defn.Compiler` implementation for OpenCL.

  Receives a computation graph from `defn`, lowers it to an `NxCL.Program`
  (`NxCL.Lower`), and runs the program on the GPU.

  Unlike the eager backend (`NxCL.Backend`) which dispatches one op at
  a time, the compiler sees the entire computation graph up front. This
  enables optimisations like buffer reuse via liveness analysis.

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

  ## Options

    * `:device` - the device index (or `{platform, index}`) to run on. See
      `NxCL.Devices`. The default is 0.
    * `:command_queue` - a queue from `NxCL.Devices.new_queue/1`. The default is
      the device's own queue. `__partitions_options__/1` gives each partition
      its own.
    * `:max_concurrency` - the partition count `__partitions_options__/1` reads.
    * `:hooks` - must be empty.

  ## Unsupported operations

  The compiler never falls back to the CPU. An op that `NxCL.Lower` cannot
  lower raises `ArgumentError` ("does not yet lower op ...") when the function
  is compiled. A dtype the device cannot run (`f64` without `cl_khr_fp64`, or a type
  with no OpenCL mapping yet) raises `NxCL.CompilerError` when the function runs.

  ## Host callbacks

  `Nx.runtime_call/4`, `Nx.io_call/2` and `Nx.Defn.Kernel.print_value/2` run
  Elixir code in the calling process, between device steps. The executor reads
  back only the tensors the callback takes. The `:hooks` option raises `ArgumentError`.

  ## Caches

  `NxCL.CompileCache` keeps the compiled closure of each function, keyed by the
  function, its input templates and the device. A repeat call does not trace or
  lower. `NxCL.ProgramCache` keeps compiled programs by structure, so two
  functions that lower alike share one.

  An unknown option raises `ArgumentError` naming the key.
  """

  @behaviour Nx.Defn.Compiler

  alias Nx.Defn.Composite
  alias NxCL.CompileCache
  alias NxCL.Compiler.Options
  alias NxCL.Devices
  alias NxCL.Executor
  alias NxCL.Lower
  alias NxCL.Program

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl true
  def __compile__(key, vars, fun, opts) do
    opts = Options.validate!(opts)
    run_opts = Keyword.take(opts, [:device, :command_queue])

    cache_key =
      {key, Enum.map(vars, &Composite.traverse(&1, fn t -> Nx.to_template(t) end)), run_opts}

    case CompileCache.fetch(cache_key) do
      {:ok, closure} ->
        closure

      {:error, :cache_miss} ->
        CompileCache.put(cache_key, build_closure(vars, fun, run_opts))
    end
  end

  defp build_closure(vars, fun, run_opts) do
    # Fail on a bad device when the function compiles, not when it first runs.
    Devices.fetch!(run_opts)

    expr = fun.(vars)
    num_inputs = vars |> Composite.flatten_list() |> length()
    program = Lower.lower(expr, num_inputs)
    program = %Program{program | signature: Program.signature(program)}

    fn [args] ->
      [Executor.run(program, args, run_opts)]
    end
  end

  @impl true
  def __partitions_options__(opts) do
    opts = Options.validate!(opts)
    {n, opts} = Keyword.pop(opts, :max_concurrency, 1)

    for _ <- 1..n//1 do
      {:ok, queue} = new_queue(opts)
      Keyword.put(opts, :command_queue, queue)
    end
  end

  defp new_queue(opts) do
    case Devices.new_queue(opts) do
      {:ok, queue} -> {:ok, queue}
      {:error, reason} -> raise NxCL.CompilerError, message: "cannot open a queue: #{reason}"
    end
  end

  @impl true
  def __to_backend__(opts) do
    {NxCL.Backend, Keyword.take(opts, [:device])}
  end
end

defmodule NxCL.CompilerError do
  defexception [:message]
end

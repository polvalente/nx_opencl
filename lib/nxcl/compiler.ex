defmodule NxCL.Compiler do
  @moduledoc """
  `Nx.Defn.Compiler` for OpenCL.

  It lowers the whole `defn` graph to an `NxCL.Compiler.Program` and runs it on
  the device. Results stay on the device between ops, and each buffer is freed
  after its last reader. `NxCL.Backend` runs one op at a time instead.

  ## Usage

      defn predict(x, w, b) do
        x |> Nx.dot(w) |> Nx.add(b) |> Nx.sigmoid()
      end

      jitted = Nx.Defn.jit(&predict/3, compiler: NxCL.Compiler)
      result = jitted.(input, weights, bias)

  Or set it as the default: `Nx.Defn.default_options(compiler: NxCL.Compiler)`.
  Arguments and results can be containers (tuples, maps).

  ## Options

    * `:device` - device index or `{platform, index}` (see `NxCL.Devices`).
      Default 0.
    * `:command_queue` - a queue from `NxCL.Devices.new_queue/1`. Default: the
      device's own queue.
    * `:max_concurrency` - how many queues `__partitions_options__/1` opens.
    * `:hooks` - must be empty.

  An unknown option raises `ArgumentError` that names the key.

  ## Unsupported operations

  The compiler never falls back to the CPU. An op it cannot lower raises
  `ArgumentError` ("does not yet lower op ...") at compile time. A dtype the
  device cannot run (`f64` without `cl_khr_fp64`, or a type with no OpenCL
  mapping) raises `NxCL.CompilerError` when the function runs.

  ## Host callbacks

  `Nx.runtime_call/4`, `Nx.io_call/2` and `Nx.Defn.Kernel.print_value/2` run
  Elixir code in the calling process, between device steps. Only the tensors the
  callback takes are read back.

  ## Caches

  `NxCL.Compiler.CompileCache` keeps the compiled closure per function, input
  templates and device, so a repeat call does not trace or lower.
  `NxCL.Compiler.ProgramCache` keeps programs by structure, so functions that
  lower alike share one.
  """

  @behaviour Nx.Defn.Compiler

  alias Nx.Defn.Composite
  alias NxCL.Compiler.CompileCache
  alias NxCL.Compiler.Executor
  alias NxCL.Compiler.Lower
  alias NxCL.Compiler.Options
  alias NxCL.Compiler.Program
  alias NxCL.Devices

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
    # Fail on a bad device at compile time.
    Devices.fetch!(run_opts)

    expr = fun.(vars)
    num_inputs = vars |> Composite.flatten_list() |> length()
    program = Lower.lower(expr, num_inputs)
    program = %Program{program | signature: Program.signature(program)}
    # Captured tensors upload once; each run reuses the buffers.
    program = Executor.with_resident_captures(program, run_opts)

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

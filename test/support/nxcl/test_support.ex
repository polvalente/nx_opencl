defmodule NxCL.TestSupport do
  @moduledoc """
  Small helpers that several test files share.

  Device tests run on one GPU, alone: two test runs at once share the device and flake.
  """

  alias NxCL.Compiler.Lower

  @doc "Lowers `fun` applied to `args` into a `NxCL.Compiler.Program`."
  @spec lower(function(), list()) :: NxCL.Compiler.Program.t()
  def lower(fun, args) do
    flat = args |> Nx.Defn.Composite.flatten_list() |> length()
    fun |> Nx.Defn.debug_expr_apply(args) |> Lower.lower(flat)
  end

  @doc "The op of each top-level instruction of `program`, in order."
  @spec ops(NxCL.Compiler.Program.t()) :: [atom()]
  def ops(program), do: Enum.map(program.instructions, & &1.op)

  @doc "`fun` jitted with `compiler` (default `NxCL.Compiler`)."
  @spec run_jit(function(), module()) :: function()
  def run_jit(fun, compiler \\ NxCL.Compiler), do: Nx.Defn.jit(fun, compiler: compiler)

  @doc "Polls `fun` every few milliseconds until it returns a truthy value. Raises after `timeout` ms."
  @spec wait_until((-> as_boolean(term())), non_neg_integer()) :: :ok
  def wait_until(fun, timeout \\ 5000) do
    deadline = System.monotonic_time(:millisecond) + timeout
    do_wait(fun, deadline)
  end

  defp do_wait(fun, deadline) do
    cond do
      fun.() ->
        :ok

      System.monotonic_time(:millisecond) > deadline ->
        raise "condition not met before the timeout"

      true ->
        Process.sleep(10)
        do_wait(fun, deadline)
    end
  end
end

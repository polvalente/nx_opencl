defmodule NxCL.TestSupport.Collector do
  @moduledoc """
  The collector process behind the call tracers (`NxCL.TestSupport.Trace`,
  `NxCL.TestSupport.Calls`, `NxCL.TestSupport.NoFallback`). A process does not
  trace itself, so a second process receives the trace messages.

  `start/3` spawns it. `stop/2` ends the trace, waits until the VM has handed over
  every trace message, and returns what the collector built. `discard/2` ends it
  without waiting: call it in an `after` so a raise in the traced code leaves no
  trace and no collector behind. A process has one tracer: do not nest tracers.
  """

  @doc """
  Spawns a collector. `handle.(message, state)` returns the next state for each
  trace message; `finish.(state)` makes the result `stop/2` returns.
  """
  @spec start(state, (tuple(), state -> state), (state -> result)) :: pid()
        when state: term(), result: term()
  def start(state, handle, finish \\ & &1) do
    spawn(fn -> loop(state, handle, finish) end)
  end

  @doc "Traces the calling process with calls sent to `collector`."
  @spec trace_self(pid()) :: :ok
  def trace_self(collector) do
    :erlang.trace(self(), true, [:call, {:tracer, collector}])
    :ok
  end

  @doc "Ends the trace of `pid` and returns the collector's result."
  @spec stop(pid(), pid()) :: term()
  def stop(collector, pid \\ self()) do
    trace_off(pid)
    # Wait until the VM has handed every trace message to the collector.
    delivered = :erlang.trace_delivered(pid)

    receive do
      {:trace_delivered, ^pid, ^delivered} -> :ok
    after
      5000 -> raise "trace messages were not delivered"
    end

    ref = make_ref()
    send(collector, {:done, self(), ref})

    receive do
      {^ref, result} -> result
    after
      5000 -> raise "trace collector did not answer"
    end
  end

  @doc "Ends the trace of `pid` and kills the collector, whatever state they are in."
  @spec discard(pid(), pid()) :: :ok
  def discard(collector, pid \\ self()) do
    trace_off(pid)
    Process.exit(collector, :kill)
    :ok
  end

  # A process that has exited has no trace left to stop.
  defp trace_off(pid) do
    :erlang.trace(pid, false, [:call, :silent])
    :ok
  rescue
    ArgumentError -> :ok
  end

  defp loop(state, handle, finish) do
    receive do
      {:trace, _pid, _kind, _mfa, _caller} = message ->
        loop(handle.(message, state), handle, finish)

      {:trace, _pid, _kind, _mfa} = message ->
        loop(handle.(message, state), handle, finish)

      {:done, from, ref} ->
        send(from, {ref, finish.(state)})
    end
  end
end

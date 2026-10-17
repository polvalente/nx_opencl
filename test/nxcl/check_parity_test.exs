defmodule NxCL.CheckParityTest do
  @moduledoc """
  `test/support/check_parity.exs` on made-up results: it must fail when the only
  device is a CPU device, when a GPU misses a cell its extensions allow, and when
  a cell that needs `cl_khr_fp64` passed on a device without it.
  """
  use ExUnit.Case, async: true

  @moduletag :tmp_dir

  @script Path.expand("../support/check_parity.exs", __DIR__)
  @all_ext "cl_khr_fp64 cl_khr_int64_base_atomics"

  defp matrix, do: NxCL.Parity.ReadmeTable.matrix()

  defp counted do
    for %{kind: :cell, scope: scope} = c <- matrix(),
        scope == :in or match?({:host, _}, scope),
        do: c
  end

  defp device(name, type, extensions) do
    %{
      status: :device,
      device: name,
      vendor: "test",
      driver_version: "1",
      version: "OpenCL 1.2",
      device_type: type,
      extensions: extensions
    }
  end

  defp passes(name, cells),
    do:
      for(c <- cells, do: %{status: :pass, device: name, op: c.op, dtype: c.dtype, path: c.path})

  defp run(dir, entries, args \\ []) do
    path = Path.join(dir, "results.exs")
    File.write!(path, inspect(entries, limit: :infinity, printable_limit: :infinity))
    System.cmd("elixir", [@script] ++ args ++ [path], stderr_to_stdout: true)
  end

  test "passes when a GPU and a CPU device cover every cell", %{tmp_dir: dir} do
    needs_ext = fn c ->
      c.dtype in [{:f, 64}, {:c, 128}] or
        (c.op == :indexed_add and c.dtype in [{:s, 64}, {:u, 64}]) or
        (c.op == :block_all_close and elem(c.dtype, 0) in [:s, :u])
    end

    gpu_cells = Enum.reject(counted(), needs_ext)
    cpu_cells = counted()

    entries =
      [device("gpu", "GPU", "")] ++
        passes("gpu", gpu_cells) ++
        [device("cpu", "CPU", @all_ext)] ++ passes("cpu", cpu_cells)

    assert {out, 0} = run(dir, entries)
    assert out =~ "parity check: ok"
  end

  test "fails when no device is a GPU", %{tmp_dir: dir} do
    entries = [device("cpu", "CPU", @all_ext)] ++ passes("cpu", counted())

    assert {out, 1} = run(dir, entries)
    assert out =~ "no result comes from a GPU device"
    assert {_, 0} = run(dir, entries, ["--cpu-only"])
  end

  test "fails when a GPU misses a cell its extensions allow", %{tmp_dir: dir} do
    cells = Enum.reject(counted(), &(&1.op == :abs and &1.dtype == {:f, 32}))
    entries = [device("gpu", "GPU", @all_ext)] ++ passes("gpu", cells)

    assert {out, 1} = run(dir, entries)
    assert out =~ "missed: abs {:f, 32}"
  end

  test "fails when an f64 cell passed only on a device without cl_khr_fp64", %{tmp_dir: dir} do
    entries = [device("gpu", "GPU", "")] ++ passes("gpu", counted())

    assert {out, 1} = run(dir, entries)
    assert out =~ "passed only on a device without the extension"
  end

  test "fails on a host call outside qr, eigh, svd, lu and callbacks", %{tmp_dir: dir} do
    entries =
      [device("cpu", "CPU", @all_ext), %{status: :host_calls, device: "cpu", ops: %{sort: 1}}] ++
        passes("cpu", counted())

    assert {out, 1} = run(dir, entries, ["--cpu-only"])
    assert out =~ "host call of a kind that may not run on the host: sort"
  end
end

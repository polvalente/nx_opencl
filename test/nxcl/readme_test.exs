defmodule NxCL.ReadmeTest do
  @moduledoc """
  The README's op table and "Out of scope" table match the parity matrix, and
  the README holds no claim the code dropped. Regenerate the tables with
  `MIX_ENV=test mix run -e 'NxCL.Parity.ReadmeTable.update()'`.
  """
  use ExUnit.Case, async: true

  alias NxCL.Parity.ReadmeTable

  @readme File.read!("README.md")

  test "the tables are generated from the matrix" do
    assert ReadmeTable.render(@readme) == @readme
  end

  test "every out-of-scope reason in the matrix is a README row" do
    matrix = ReadmeTable.matrix()
    reasons = for %{kind: :cut, reason: reason} <- matrix, do: reason
    cells = for %{kind: :cell, scope: {:out, reason}} <- matrix, uniq: true, do: reason

    assert cells -- reasons == []
    for reason <- reasons, do: assert(@readme =~ reason)
  end

  test "the f16 cut is a README row" do
    assert @readme =~
             "| `f16` on devices without `cl_khr_fp16` | No device in the test set has the extension, so the cells cannot run. The f16 kernels stay in the code and are tested by Rust source tests. Decided by the user 2026-10-16. |"
  end

  test "no stale claim" do
    refute @readme =~ "TODO"
    refute @readme =~ ~r/not supported/i
    refute @readme =~ ~r/fall(s)? back/i
    refute @readme =~ "GPU-Accelerated"
    refute @readme =~ "flat op sequence"
  end

  test "host calls are the four linear algebra blocks" do
    hosts =
      for %{kind: :cell, scope: {:host, _}, op: op} <- ReadmeTable.matrix(), uniq: true, do: op

    assert Enum.sort(hosts) ==
             [:block_lin_alg_eigh, :block_lin_alg_lu, :block_lin_alg_qr, :block_lin_alg_svd]
  end
end

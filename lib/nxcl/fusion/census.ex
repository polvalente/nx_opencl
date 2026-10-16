defmodule NxCL.Fusion.Census do
  @moduledoc """
  Counts the instructions of a program, by op and by class, and estimates how
  many kernel launches it makes.

  Classes: `:elementwise`, `:shape` (broadcast and data movement), `:reduce`,
  `:dot` (`batched_gemm` and `conv`), `:fused`, `:control` and `:other`. The
  count includes the sub-programs of a `cond` or `while`, once each. A
  `reshape`, `squeeze` or `bitcast` shares its operand's buffer and launches
  nothing. A reduction, scan or sort can launch more than one kernel, so the
  launch count is a lower bound; `NxCL.Native.kernel_cache_stats/1` has the
  measured number.

  Turn it on with the compiler option `census: true`, which prints
  `format/1` when the function compiles.
  """

  alias NxCL.Program
  alias NxCL.Program.Builder
  alias NxCL.Program.Instruction

  @alias_ops [:reshape, :squeeze, :bitcast]
  @control [:cond, :while, :host_call]
  @shape ~w(broadcast transpose reverse slice put_slice pad concatenate stack iota eye)a
  @reduce ~w(sum product reduce_max reduce_min all any argmax argmin reduce cumulative_sum
             cumulative_product cumulative_min cumulative_max sort argsort window_sum
             window_product window_max window_min window_scatter_max window_scatter_min)a
  @dot [:batched_gemm, :conv]
  @elementwise Builder.unary_ops() ++ Builder.binary_ops() ++ [:as_type, :select, :clip]

  @type t :: %{
          instructions: non_neg_integer(),
          launches: non_neg_integer(),
          histogram: %{atom() => non_neg_integer()},
          classes: %{atom() => non_neg_integer()}
        }

  @doc "The census of `program`."
  @spec count(Program.t()) :: t()
  def count(%Program{} = program) do
    instrs = collect(program.instructions)
    histogram = Enum.frequencies_by(instrs, & &1.op)

    %{
      instructions: length(instrs),
      launches: Enum.count(instrs, &(&1.op not in @alias_ops and &1.op not in @control)),
      histogram: histogram,
      classes: Enum.frequencies_by(instrs, &class/1)
    }
  end

  @doc "The class of an instruction."
  @spec class(Instruction.t()) :: atom()
  def class(%Instruction{op: op}) do
    cond do
      op in @elementwise -> :elementwise
      op == :fused -> :fused
      op in @shape or op in @alias_ops -> :shape
      op in @reduce -> :reduce
      op in @dot -> :dot
      op in @control -> :control
      true -> :other
    end
  end

  @doc "A plain-text table of `count/1`."
  @spec format(t(), String.t()) :: String.t()
  def format(census, title \\ "census") do
    ops =
      census.histogram
      |> Enum.sort_by(fn {op, n} -> {-n, op} end)
      |> Enum.map_join(", ", fn {op, n} -> "#{op}=#{n}" end)

    classes =
      [:elementwise, :fused, :shape, :reduce, :dot, :control, :other]
      |> Enum.map_join(" ", &"#{&1}=#{Map.get(census.classes, &1, 0)}")

    "#{title}: #{census.instructions} instructions, #{census.launches} launches\n" <>
      "  classes: #{classes}\n  ops: #{ops}"
  end

  defp collect(instructions) do
    Enum.flat_map(instructions, fn instr ->
      [instr | instr.subprograms |> Enum.flat_map(&collect(&1.instructions))]
    end)
  end
end

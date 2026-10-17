defmodule NxCL.Parity.ReadmeTable do
  @moduledoc """
  The README's op table and "Out of scope" table, generated from the parity
  matrix (`test/support/parity_matrix.exs`).

  The tables sit between marker comments in `README.md`:

      <!-- parity:ops:start -->
      ...
      <!-- parity:ops:end -->

  `update/1` rewrites them in place; the README test checks that they match.
  Run it after `.work/gen_parity_matrix.exs` writes a new matrix:

      MIX_ENV=test mix run -e 'NxCL.Parity.ReadmeTable.update()'
  """

  @matrix_path Path.expand("../../parity_matrix.exs", __DIR__)

  @doc "The matrix entries."
  @spec matrix() :: [map()]
  def matrix, do: @matrix_path |> Code.eval_file() |> elem(0)

  @doc "The markdown table of ops: one row per op, with the status of each path."
  @spec ops_table([map()]) :: String.t()
  def ops_table(matrix \\ matrix()) do
    cells = Enum.filter(matrix, &(&1.kind == :cell))

    full =
      cells |> Enum.filter(&in_scope?/1) |> Enum.map(& &1.dtype) |> Enum.uniq() |> Enum.sort()

    rows =
      cells
      |> Enum.group_by(& &1.op)
      |> Enum.sort_by(fn {op, _} -> op end)
      |> Enum.map(fn {op, op_cells} ->
        paths = Enum.group_by(op_cells, & &1.path)
        dtypes = op_cells |> Enum.filter(&in_scope?/1) |> Enum.map(& &1.dtype) |> Enum.uniq()

        "| `#{op}` | #{status(paths[:compiler])} | #{status(paths[:backend])} | #{dtype_text(dtypes, full)} |"
      end)

    Enum.join(
      ["| Op | `NxCL.Compiler` | `NxCL.Backend` | Dtypes |", "| --- | --- | --- | --- |" | rows],
      "\n"
    )
  end

  @doc "The markdown table of scope cuts, one row per README \"Out of scope\" entry."
  @spec cuts_table([map()]) :: String.t()
  def cuts_table(matrix \\ matrix()) do
    rows = for %{kind: :cut, cut: cut, reason: reason} <- matrix, do: "| #{cut} | #{reason} |"
    Enum.join(["| Cut | Reason |", "| --- | --- |" | rows], "\n")
  end

  @doc "The README text with both tables replaced between their markers."
  @spec render(String.t(), [map()]) :: String.t()
  def render(readme, matrix \\ matrix()) do
    readme
    |> replace("ops", ops_table(matrix))
    |> replace("cuts", cuts_table(matrix))
  end

  @doc "Rewrites the tables in `path` (default `README.md`)."
  @spec update(Path.t()) :: :ok
  def update(path \\ "README.md"), do: File.write!(path, render(File.read!(path)))

  defp replace(text, name, table) do
    start = "<!-- parity:#{name}:start -->"
    stop = "<!-- parity:#{name}:end -->"
    pattern = ~r/#{Regex.escape(start)}.*?#{Regex.escape(stop)}/s
    replacement = start <> "\n" <> table <> "\n" <> stop

    unless Regex.match?(pattern, text), do: raise("README has no #{start} ... #{stop} markers")
    Regex.replace(pattern, text, fn _ -> replacement end)
  end

  defp in_scope?(%{scope: :in}), do: true
  defp in_scope?(%{scope: {:host, _}}), do: true
  defp in_scope?(_), do: false

  # "device" when any cell runs on the device, "host call" for the G5 blocks, else "out".
  defp status(nil), do: "out of scope"

  defp status(cells) do
    cond do
      Enum.any?(cells, &match?(%{scope: {:host, _}}, &1)) -> "host call"
      Enum.any?(cells, &(&1.scope == :in)) -> "device"
      true -> "out of scope"
    end
  end

  defp dtype_text([], _full), do: "none"
  defp dtype_text(dtypes, full) when length(dtypes) == length(full), do: "all"

  defp dtype_text(dtypes, _full) do
    dtypes |> Enum.sort() |> Enum.map_join(" ", &"`#{Nx.Type.to_string(&1)}`")
  end
end

defmodule NxCL.Parity.Recorder do
  @moduledoc """
  Collects passing `{op, dtype, path}` cells and writes them to
  `.work/parity_results.exs` when `NXCL_PARITY_REPORT=1`.

  The file is a list of maps: `%{op, dtype, path, device, status: :pass}` for a
  passing cell and `%{requires, device, status: :skipped}` for a skipped
  capability. A write replaces the entries of this device and keeps the
  entries of other devices, so a run on each device adds up.
  """

  @table __MODULE__
  @default_path ".work/parity_results.exs"

  @spec start() :: :ok
  def start do
    if :ets.whereis(@table) == :undefined do
      :ets.new(@table, [:named_table, :public, :set])
    end

    :ok
  end

  @spec record(atom(), Nx.Type.t(), :compiler | :backend) :: :ok
  def record(op, dtype, path) do
    start()
    :ets.insert(@table, {{op, dtype, path}})
    :ok
  end

  @spec recorded() :: [{atom(), Nx.Type.t(), atom()}]
  def recorded do
    start()
    @table |> :ets.tab2list() |> Enum.map(&elem(&1, 0)) |> Enum.sort()
  end

  @doc "Writes the file when `env` is `\"1\"`; returns `:written` or `:disabled`."
  @spec write_if_enabled(String.t() | nil, Path.t(), [atom()]) :: :written | :disabled
  def write_if_enabled(
        env \\ System.get_env("NXCL_PARITY_REPORT"),
        path \\ @default_path,
        skipped \\ []
      )

  def write_if_enabled("1", path, skipped) do
    write(path, skipped)
    :written
  end

  def write_if_enabled(_env, _path, _skipped), do: :disabled

  @spec write(Path.t(), [atom()]) :: :ok
  def write(path, skipped \\ []) do
    device = NxCL.Parity.device_name()

    passed =
      for {op, dtype, cell_path} <- recorded() do
        %{op: op, dtype: dtype, path: cell_path, device: device, status: :pass}
      end

    skips = for cap <- skipped, do: %{requires: cap, device: device, status: :skipped}
    mine = passed ++ skips

    others = path |> read_existing() |> Enum.reject(&(&1.device == device))

    body =
      inspect(Enum.sort_by(others ++ mine, &Map.to_list/1),
        limit: :infinity,
        printable_limit: :infinity,
        pretty: true
      )

    File.write!(path, body <> "\n")
  end

  defp read_existing(path) do
    case File.read(path) do
      {:ok, text} -> text |> Code.eval_string() |> elem(0)
      {:error, _} -> []
    end
  end
end

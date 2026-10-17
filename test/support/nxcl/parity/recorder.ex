defmodule NxCL.Parity.Recorder do
  @moduledoc """
  Collects passing `{op, dtype, path}` cells and writes them to
  `.work/parity_results.exs` when `NXCL_PARITY_REPORT=1`. The
  `NXCL_PARITY_RESULTS` variable names another file; with no `.work/`
  directory (CI) the default is `tmp/parity_results.exs`.

  The file is a list of maps: `%{op, dtype, path, device, status: :pass}` for a
  passing cell, `%{requires, device, status: :skipped}` for a skipped
  capability, and one `%{device, status: :device, vendor, driver_version,
  version, device_type, extensions}` for the device itself, and one
  `%{device, status: :host_calls, ops: %{kind => count}}` with the host calls
  by kind (`NxCL.Parity.Tripwire`). A write replaces
  the entries of this device and keeps the entries of other devices, so a run
  on each device adds up.
  """

  @table __MODULE__

  @doc "Where results go: `NXCL_PARITY_RESULTS`, else `.work/` when it exists, else `tmp/`."
  @spec default_path() :: Path.t()
  def default_path do
    cond do
      path = System.get_env("NXCL_PARITY_RESULTS") -> path
      File.dir?(".work") -> ".work/parity_results.exs"
      true -> "tmp/parity_results.exs"
    end
  end

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
        path \\ default_path(),
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
    host = %{status: :host_calls, device: device, ops: NxCL.Parity.Tripwire.host_calls()}
    mine = [NxCL.Parity.device_identity(), host | passed] ++ skips

    others = path |> read_existing() |> Enum.reject(&(&1.device == device))

    body =
      inspect(Enum.sort_by(others ++ mine, &Map.to_list/1),
        limit: :infinity,
        printable_limit: :infinity,
        pretty: true
      )

    File.mkdir_p!(Path.dirname(path))
    File.write!(path, body <> "\n")
  end

  defp read_existing(path) do
    case File.read(path) do
      {:ok, text} -> text |> Code.eval_string() |> elem(0)
      {:error, _} -> []
    end
  end
end

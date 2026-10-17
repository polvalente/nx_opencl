defmodule NxCL.Compiler.Options do
  @moduledoc false
  # Checks the options `NxCL.Compiler` takes. An unknown key raises
  # `ArgumentError` and names the key.

  @keys [:device, :command_queue, :max_concurrency, :hooks]

  @doc "Validates `opts` and returns them. Raises on an unknown key or a non-empty `:hooks`."
  @spec validate!(keyword()) :: keyword()
  def validate!(opts) do
    opts = Keyword.validate!(opts, @keys)
    check_hooks!(Keyword.get(opts, :hooks, %{}))
    opts
  end

  # The named-override map is not wired through.
  defp check_hooks!(hooks) when hooks == %{} or hooks == [], do: :ok

  defp check_hooks!(hooks) do
    raise ArgumentError,
          "NxCL.Compiler does not support the :hooks option (got callbacks for " <>
            "#{inspect(hooks_keys(hooks))}); io_call nodes run their own callback"
  end

  defp hooks_keys(hooks) when is_map(hooks), do: Map.keys(hooks)
  defp hooks_keys(hooks), do: hooks
end

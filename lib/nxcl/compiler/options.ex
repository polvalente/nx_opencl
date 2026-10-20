defmodule NxCL.Compiler.Options do
  @moduledoc false
  # Checks the options `NxCL.Compiler` takes.

  @keys [:device, :command_queue, :max_concurrency]

  @doc "Validates `opts` and returns them. Raises on an unknown key or a non-empty `:hooks`."
  @spec validate!(keyword()) :: keyword()
  def validate!(opts) do
    # Nx passes an empty `:hooks` to every compiler.
    {hooks, opts} = Keyword.pop(opts, :hooks, %{})

    if hooks not in [%{}, []] do
      named = if is_map(hooks), do: Map.keys(hooks), else: hooks

      raise ArgumentError,
            "NxCL.Compiler does not support the :hooks option (got callbacks for " <>
              "#{inspect(named)}); io_call nodes run their own callback"
    end

    Keyword.validate!(opts, @keys)
  end
end

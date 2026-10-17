defmodule NxCL.Executor.Host do
  @moduledoc false
  # Runs one `:host_call` step: read the operands the callback takes, call it
  # in the calling process, check the result, upload it.
  #
  # The callback gets tensors on the default backend (`Nx.from_binary/2`), in
  # the container shape of the node's input. Buffers it does not take stay on the
  # device. A side-effect call (`:io_call`) returns its input buffers as its
  # outputs, with no copy and no upload.

  alias Nx.Defn.Composite

  @doc """
  Runs `entry` (a `:host_calls` table entry) on `buffers`, the device buffers of
  its operands, and returns one buffer per output leaf.
  """
  @spec run(map(), reference(), [reference()]) :: [reference()]
  def run(%{kind: :io_call, fun: nil}, _ctx, buffers), do: buffers

  def run(%{kind: :io_call, fun: fun} = entry, _ctx, buffers) do
    fun.(read(entry.in_template, buffers))
    buffers
  end

  def run(%{kind: :runtime_call} = entry, ctx, buffers) do
    result = entry.fun.(read(entry.in_template, buffers), entry.opts)

    if not Nx.compatible?(entry.out_template, result) do
      raise ArgumentError,
            "expected the runtime_call function to match the given output template " <>
              "#{inspect(entry.out_template)}, got: #{inspect(actual_template(result))}"
    end

    leaves = Composite.flatten_list([result])
    Enum.map(leaves, &NxCL.Native.buffer_write(ctx, &1 |> Nx.to_tensor() |> Nx.to_binary()))
  end

  # Rebuilds the input container with host tensors read from the buffers.
  defp read(in_template, buffers) do
    {container, []} =
      Composite.traverse(in_template, buffers, fn template, [buf | rest] ->
        size = Nx.byte_size(template)

        tensor =
          buf
          |> NxCL.Native.buffer_read(size)
          |> Nx.from_binary(template.type)
          |> Nx.reshape(template.shape, names: template.names)

        {tensor, rest}
      end)

    container
  end

  # The templates of what the callback returned, for the error text.
  defp actual_template(result) do
    Composite.traverse(result, &(&1 |> Nx.to_tensor() |> Nx.to_template()))
  rescue
    _ -> result
  end
end

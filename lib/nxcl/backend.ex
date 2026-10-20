defmodule NxCL.Backend do
  @moduledoc """
  Nx backend for OpenCL GPUs.

  Tensor data lives on the device. Each callback builds a one-op program and
  runs it with the kernels `NxCL.Compiler` uses. The program is cached by op,
  operand types, shapes and attributes.

  No op runs on `Nx.BinaryBackend`. An op with no lowering raises
  `ArgumentError` ("does not yet lower op ..."). `from_pointer/5` and
  `to_pointer/2` raise, because OpenCL has no portable handle to share.

  ## Memory

  `Nx.backend_deallocate/1` frees the device buffer at once. Otherwise it goes
  when the BEAM collects the tensor. A `reshape` or `squeeze` result shares its
  operand's memory, which goes when both are freed.

  ## Usage

      Nx.default_backend(NxCL.Backend)

      gpu_tensor = Nx.backend_transfer(cpu_tensor, NxCL.Backend)

  ## Device selection

  The default is device 0. Pass `device:` to `Nx.default_backend/1`,
  `Nx.backend_transfer` or `Nx.Defn.jit/2` to pick another (see
  `NxCL.Devices`).
  """

  @behaviour Nx.Backend

  alias NxCL.Compiler.Evaluator
  alias NxCL.Compiler.Ops

  defstruct [:ref, :ctx]
  @impl true
  def init(opts) do
    opts
  end

  @impl true
  def from_binary(%Nx.Tensor{} = tensor, binary, backend_opts) do
    ctx = NxCL.Devices.fetch!(Keyword.take(backend_opts, [:device])).ctx
    ref = ctx |> NxCL.Native.buffer_write(binary) |> NxCL.Native.ok!("could not upload a tensor")
    put_in(tensor.data, %__MODULE__{ref: ref, ctx: ctx})
  end

  # `limit` counts elements, vectorized axes included.
  @impl true
  def to_binary(%Nx.Tensor{data: %__MODULE__{ref: ref}, type: {_, bits}} = tensor, limit) do
    ref
    |> NxCL.Native.buffer_read(div(bits * min(limit, Nx.flat_size(tensor)), 8))
    |> NxCL.Native.ok!("could not read a tensor")
  end

  @impl true
  def inspect(%Nx.Tensor{} = tensor, opts) do
    binary = to_binary(tensor, Nx.flat_size(tensor))
    Nx.Backend.inspect(tensor, binary, opts)
  end

  @impl true
  def backend_deallocate(%Nx.Tensor{data: %__MODULE__{ref: ref}}) do
    if NxCL.Native.buffer_free(ref), do: :ok, else: :already_deallocated
  end

  @impl true
  def backend_copy(tensor, Nx.BinaryBackend, _opts) do
    binary = to_binary(tensor, Nx.flat_size(tensor))

    Nx.BinaryBackend.from_binary(
      %{tensor | data: %Nx.BinaryBackend{}},
      binary,
      []
    )
  end

  def backend_copy(tensor, backend, opts) do
    binary = to_binary(tensor, Nx.flat_size(tensor))
    backend.from_binary(%{tensor | data: %{__struct__: backend}}, binary, opts)
  end

  @impl true
  def backend_transfer(tensor, backend, opts) do
    result = backend_copy(tensor, backend, opts)
    backend_deallocate(tensor)
    result
  end

  # Batches are cut from the host copy and uploaded again.
  @impl true
  def to_batched(out, %Nx.Tensor{} = tensor, opts) do
    batch_size = elem(out.shape, 0)
    full = div(elem(tensor.shape, 0), batch_size)
    repeat? = rem(elem(tensor.shape, 0), batch_size) != 0 and opts[:leftover] == :repeat
    range = if repeat?, do: 0..full, else: 0..(full - 1)//1
    binary = to_binary(tensor, Nx.flat_size(tensor))
    bytes = Nx.byte_size(out)
    device_opts = Evaluator.device_opts([tensor])

    Stream.map(range, fn
      ^full ->
        before = full * bytes
        available = byte_size(binary) - before
        rest = binary_part(binary, before, available) <> binary_part(binary, 0, bytes - available)
        from_binary(out, rest, device_opts)

      i ->
        from_binary(out, binary_part(binary, i * bytes, bytes), device_opts)
    end)
  end

  @impl true
  def constant(%Nx.Tensor{} = out, number, backend_opts),
    do: Evaluator.constant(out, number, Keyword.take(backend_opts, [:device]))

  @impl true
  def eye(%Nx.Tensor{} = out, backend_opts),
    do: Evaluator.eye(out, Keyword.take(backend_opts, [:device]))

  @impl true
  def iota(%Nx.Tensor{} = out, axis, backend_opts),
    do: Evaluator.iota(out, axis, Keyword.take(backend_opts, [:device]))

  # `logical_not` and `phase` reach the backend as blocks, not callbacks.
  for op <- Ops.unary_ops() -- [:logical_not, :phase] do
    @impl true
    def unquote(op)(out, tensor), do: Evaluator.unary(unquote(op), out, tensor)
  end

  for op <- Ops.binary_ops() do
    @impl true
    def unquote(op)(out, left, right), do: Evaluator.binary(unquote(op), out, left, right)
  end

  @impl true
  def as_type(out, tensor), do: Evaluator.as_type(out, tensor)

  @impl true
  def bitcast(out, tensor), do: Evaluator.bitcast(out, tensor)

  @impl true
  def select(out, pred, on_true, on_false), do: Evaluator.select(out, pred, on_true, on_false)

  @impl true
  def clip(out, tensor, min, max), do: Evaluator.clip(out, tensor, min, max)
  @impl true
  def dot(out, left, left_contract, left_batch, right, right_contract, right_batch) do
    Evaluator.dot(out, left, left_contract, left_batch, right, right_contract, right_batch)
  end

  @impl true
  def conv(out, tensor, kernel, opts), do: Evaluator.conv(out, tensor, kernel, opts)
  @impl true
  def reshape(out, tensor), do: Evaluator.alias_op(:reshape, out, tensor)

  @impl true
  def squeeze(out, tensor, _axes), do: Evaluator.alias_op(:squeeze, out, tensor)

  @impl true
  def broadcast(out, tensor, _shape, axes), do: Evaluator.broadcast(out, tensor, axes)

  @impl true
  def transpose(out, tensor, axes), do: Evaluator.transpose(out, tensor, axes)

  @impl true
  def reverse(out, tensor, axes), do: Evaluator.reverse(out, tensor, axes)

  @impl true
  def pad(out, tensor, pad_value, padding_config),
    do: Evaluator.pad(out, tensor, pad_value, padding_config)

  @impl true
  def slice(out, tensor, starts, lengths, strides),
    do: Evaluator.slice(out, tensor, starts, lengths, strides)

  @impl true
  def put_slice(out, tensor, starts, slice), do: Evaluator.put_slice(out, tensor, starts, slice)

  @impl true
  def concatenate(out, tensors, axis), do: Evaluator.join(:concatenate, out, tensors, axis)

  @impl true
  def stack(out, tensors, axis), do: Evaluator.join(:stack, out, tensors, axis)

  for op <- [:sum, :product, :reduce_max, :reduce_min, :all, :any] do
    @impl true
    def unquote(op)(out, tensor, opts), do: Evaluator.reduce(unquote(op), out, tensor, opts)
  end

  for op <- [:argmax, :argmin] do
    @impl true
    def unquote(op)(out, tensor, opts), do: Evaluator.arg_reduce(unquote(op), out, tensor, opts)
  end

  @impl true
  def reduce(out, tensor, acc, opts, fun), do: Evaluator.reduce_fun(out, tensor, acc, opts, fun)

  for op <- [:window_sum, :window_product, :window_max, :window_min] do
    @impl true
    def unquote(op)(out, tensor, shape, opts),
      do: Evaluator.window(unquote(op), out, tensor, shape, opts)
  end

  @impl true
  def window_reduce(out, tensor, acc, shape, opts, fun),
    do: Evaluator.window_reduce(out, tensor, acc, shape, opts, fun)

  for op <- [:window_scatter_max, :window_scatter_min] do
    @impl true
    def unquote(op)(out, tensor, source, init, shape, opts),
      do: Evaluator.window_scatter(unquote(op), out, tensor, source, init, shape, opts)
  end

  for op <- [:sort, :argsort] do
    @impl true
    def unquote(op)(out, tensor, opts), do: Evaluator.sort(unquote(op), out, tensor, opts)
  end

  @impl true
  def gather(out, input, indices, opts), do: Evaluator.gather(out, input, indices, opts)

  for op <- [:indexed_add, :indexed_put] do
    @impl true
    def unquote(op)(out, tensor, indices, updates, opts),
      do: Evaluator.indexed(unquote(op), out, tensor, indices, updates, opts)
  end

  @impl true
  def triangular_solve(out, a, b, opts), do: Evaluator.triangular_solve(out, a, b, opts)

  @impl true
  def fft(out, tensor, opts), do: Evaluator.fft(:fft, out, tensor, opts)

  @impl true
  def ifft(out, tensor, opts), do: Evaluator.fft(:ifft, out, tensor, opts)

  @impl true
  def block(struct, out, args, fun), do: Evaluator.block(struct, out, args, fun)
  @impl true
  def from_pointer(_out, _pointer, _backend_opts, _offset, _byte_size) do
    raise ArgumentError,
          "NxCL.Backend does not support from_pointer: OpenCL has no portable handle to import"
  end

  @impl true
  def to_pointer(_tensor, _opts) do
    raise ArgumentError,
          "NxCL.Backend does not support to_pointer: OpenCL has no portable handle to export"
  end
end

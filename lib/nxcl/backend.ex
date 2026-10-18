defmodule NxCL.Backend do
  @moduledoc """
  Nx backend for OpenCL GPUs.

  Tensor data lives on the GPU as opaque NIF resource references. Each compute
  callback builds a one-op `NxCL.Program` and runs it through `NxCL.Executor`,
  the same executor and kernels that `NxCL.Compiler` uses (`NxCL.Eager`). The
  executor caches the program by op, operand types, shapes and attributes, so a
  repeat call builds no kernel.

  ## No CPU copy

  The backend never runs an op on `Nx.BinaryBackend`. An op with no lowering
  raises `ArgumentError` ("does not yet lower op ..."). Each `Nx.Block` runs as a
  native instruction or as its default expression (every op in it lowers).

  `from_pointer/5` and `to_pointer/2` raise: OpenCL has no portable handle to
  import or export.

  ## Memory

  `Nx.backend_deallocate/1` frees the device buffer at once. Without it, the
  buffer goes when the BEAM collects the tensor. A `reshape` or `squeeze` result
  shares its operand's device memory; the memory goes when both are freed.

  ## Usage

      # Set as default backend
      Nx.default_backend(NxCL.Backend)

      # Or transfer individual tensors
      gpu_tensor = Nx.backend_transfer(cpu_tensor, NxCL.Backend)

  ## Device selection

  The default is the first available GPU device (index 0). Pass `device:` to
  `Nx.default_backend/1`, `Nx.backend_transfer/3` or `Nx.Defn.jit/2` to pick
  another one (see `NxCL.Devices`). `NxCL.Devices` holds one OpenCL context per
  device and shares it across all processes.
  """

  @behaviour Nx.Backend

  alias NxCL.Eager
  alias NxCL.Program.Builder

  defstruct [:ref, :ctx]

  # ── Lifecycle ──────────────────────────────────────

  @impl true
  def init(opts) do
    opts
  end

  # ── Data Transfer ──────────────────────────────────

  @impl true
  def from_binary(%Nx.Tensor{} = tensor, binary, backend_opts) do
    ctx = NxCL.Devices.fetch!(Keyword.take(backend_opts, [:device])).ctx
    ref = NxCL.Native.buffer_write(ctx, binary)
    put_in(tensor.data, %__MODULE__{ref: ref, ctx: ctx})
  end

  # `limit` counts elements (vectorized axes included), as in `Nx.BinaryBackend`.
  @impl true
  def to_binary(%Nx.Tensor{data: %__MODULE__{ref: ref}, type: {_, bits}} = tensor, limit) do
    NxCL.Native.buffer_read(ref, div(bits * min(limit, Nx.flat_size(tensor)), 8))
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

  # Batches are cut from the host copy, as `Nx.BinaryBackend` does; each goes back up.
  @impl true
  def to_batched(out, %Nx.Tensor{} = tensor, opts) do
    batch_size = elem(out.shape, 0)
    full = div(elem(tensor.shape, 0), batch_size)
    repeat? = rem(elem(tensor.shape, 0), batch_size) != 0 and opts[:leftover] == :repeat
    range = if repeat?, do: 0..full, else: 0..(full - 1)//1
    binary = to_binary(tensor, Nx.flat_size(tensor))
    bytes = Nx.byte_size(out)

    Stream.map(range, fn
      ^full ->
        before = full * bytes
        available = byte_size(binary) - before
        rest = binary_part(binary, before, available) <> binary_part(binary, 0, bytes - available)
        from_binary(out, rest, [])

      i ->
        from_binary(out, binary_part(binary, i * bytes, bytes), [])
    end)
  end

  # ── Creation ───────────────────────────────────────

  @impl true
  def constant(%Nx.Tensor{} = out, number, _backend_opts), do: Eager.constant(out, number)

  @impl true
  def eye(%Nx.Tensor{} = out, _backend_opts), do: Eager.eye(out)

  @impl true
  def iota(%Nx.Tensor{} = out, axis, _backend_opts), do: Eager.iota(out, axis)

  # ── Elementwise ────────────────────────────────────

  # `logical_not` and `phase` reach the backend as blocks, not callbacks.
  for op <- Builder.unary_ops() -- [:logical_not, :phase] do
    @impl true
    def unquote(op)(out, tensor), do: Eager.unary(unquote(op), out, tensor)
  end

  for op <- Builder.binary_ops() do
    @impl true
    def unquote(op)(out, left, right), do: Eager.binary(unquote(op), out, left, right)
  end

  @impl true
  def as_type(out, tensor), do: Eager.as_type(out, tensor)

  @impl true
  def bitcast(out, tensor), do: Eager.bitcast(out, tensor)

  @impl true
  def select(out, pred, on_true, on_false), do: Eager.select(out, pred, on_true, on_false)

  @impl true
  def clip(out, tensor, min, max), do: Eager.clip(out, tensor, min, max)

  # ── Dot / Conv ─────────────────────────────────────

  @impl true
  def dot(out, left, left_contract, left_batch, right, right_contract, right_batch) do
    Eager.dot(out, left, left_contract, left_batch, right, right_contract, right_batch)
  end

  @impl true
  def conv(out, tensor, kernel, opts), do: Eager.conv(out, tensor, kernel, opts)

  # ── Shape ──────────────────────────────────────────

  @impl true
  def reshape(out, tensor), do: Eager.alias_op(:reshape, out, tensor)

  @impl true
  def squeeze(out, tensor, _axes), do: Eager.alias_op(:squeeze, out, tensor)

  @impl true
  def broadcast(out, tensor, _shape, axes), do: Eager.broadcast(out, tensor, axes)

  @impl true
  def transpose(out, tensor, axes), do: Eager.transpose(out, tensor, axes)

  @impl true
  def reverse(out, tensor, axes), do: Eager.reverse(out, tensor, axes)

  @impl true
  def pad(out, tensor, pad_value, padding_config),
    do: Eager.pad(out, tensor, pad_value, padding_config)

  @impl true
  def slice(out, tensor, starts, lengths, strides),
    do: Eager.slice(out, tensor, starts, lengths, strides)

  @impl true
  def put_slice(out, tensor, starts, slice), do: Eager.put_slice(out, tensor, starts, slice)

  @impl true
  def concatenate(out, tensors, axis), do: Eager.join(:concatenate, out, tensors, axis)

  @impl true
  def stack(out, tensors, axis), do: Eager.join(:stack, out, tensors, axis)

  # ── Reductions, windows, sort ──────────────────────

  for op <- [:sum, :product, :reduce_max, :reduce_min, :all, :any] do
    @impl true
    def unquote(op)(out, tensor, opts), do: Eager.reduce(unquote(op), out, tensor, opts)
  end

  for op <- [:argmax, :argmin] do
    @impl true
    def unquote(op)(out, tensor, opts), do: Eager.arg_reduce(unquote(op), out, tensor, opts)
  end

  @impl true
  def reduce(out, tensor, acc, opts, fun), do: Eager.reduce_fun(out, tensor, acc, opts, fun)

  for op <- [:window_sum, :window_product, :window_max, :window_min] do
    @impl true
    def unquote(op)(out, tensor, shape, opts),
      do: Eager.window(unquote(op), out, tensor, shape, opts)
  end

  @impl true
  def window_reduce(out, tensor, acc, shape, opts, fun),
    do: Eager.window_reduce(out, tensor, acc, shape, opts, fun)

  for op <- [:window_scatter_max, :window_scatter_min] do
    @impl true
    def unquote(op)(out, tensor, source, init, shape, opts),
      do: Eager.window_scatter(unquote(op), out, tensor, source, init, shape, opts)
  end

  for op <- [:sort, :argsort] do
    @impl true
    def unquote(op)(out, tensor, opts), do: Eager.sort(unquote(op), out, tensor, opts)
  end

  # ── Indexing ───────────────────────────────────────

  @impl true
  def gather(out, input, indices, opts), do: Eager.gather(out, input, indices, opts)

  for op <- [:indexed_add, :indexed_put] do
    @impl true
    def unquote(op)(out, tensor, indices, updates, opts),
      do: Eager.indexed(unquote(op), out, tensor, indices, updates, opts)
  end

  # ── Linear algebra, FFT, blocks ────────────────────

  @impl true
  def triangular_solve(out, a, b, opts), do: Eager.triangular_solve(out, a, b, opts)

  @impl true
  def fft(out, tensor, opts), do: Eager.fft(:fft, out, tensor, opts)

  @impl true
  def ifft(out, tensor, opts), do: Eager.fft(:ifft, out, tensor, opts)

  @impl true
  def block(struct, out, args, fun), do: Eager.block(struct, out, args, fun)

  # ── Pointers ───────────────────────────────────────

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

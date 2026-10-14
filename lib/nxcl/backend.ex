defmodule NxCL.Backend do
  @moduledoc """
  Nx backend for OpenCL GPUs.

  Tensor data lives on the GPU as opaque NIF resource references.
  For ops not yet implemented on GPU, data is transferred to the CPU
  via `Nx.BinaryBackend`, computed there, and transferred back.

  ## GPU-accelerated operations

  The following operations run on the GPU for `f32` tensors with matching
  shapes (no broadcasting required):

    * Binary elementwise: `add`, `subtract`, `multiply`, `divide`
    * Unary elementwise: `negate`, `exp`, `log`, `tanh`, `sigmoid`, `abs`
    * Linear algebra: `dot` (2D matrix multiply)
    * Reductions: `sum`, `reduce_max`, `reduce_min` (full reduction)
    * Shape: `reshape`, `squeeze`, `bitcast` (zero-copy, metadata only)

  All other operations fall back to `Nx.BinaryBackend` automatically.

  ## Usage

      # Set as default backend
      Nx.default_backend(NxCL.Backend)

      # Or transfer individual tensors
      gpu_tensor = Nx.backend_transfer(cpu_tensor, NxCL.Backend)

  ## Device selection

  Currently uses the first available GPU device (index 0). A single
  OpenCL context is shared across all processes via `persistent_term`.
  """

  @behaviour Nx.Backend

  defstruct [:ref, :ctx]

  # Shared device context across all processes via persistent_term
  @ctx_key :nxcl_device_ctx

  defp get_ctx() do
    case :persistent_term.get(@ctx_key, nil) do
      nil ->
        ctx = NxCL.Native.device_ctx_create(0)
        :persistent_term.put(@ctx_key, ctx)
        ctx

      ctx ->
        ctx
    end
  end

  defp to_gpu(%Nx.Tensor{data: %__MODULE__{}} = tensor), do: tensor

  defp to_gpu(%Nx.Tensor{} = tensor) do
    from_binary(tensor, Nx.to_binary(tensor), [])
  end

  defp gpu_ref(%Nx.Tensor{data: %__MODULE__{ref: ref}}), do: ref

  defp flat_size(%Nx.Tensor{} = t), do: Nx.size(t)

  # Fallback: run on CPU and bring result back to GPU
  defp fallback(callback, args) do
    # Convert GPU tensors to BinaryBackend tensors
    cpu_args = Enum.map(args, &to_cpu_arg/1)

    # Run on CPU
    result = apply(Nx.BinaryBackend, callback, cpu_args)

    # Convert result back to GPU
    case result do
      %Nx.Tensor{} = t -> to_gpu(t)
      other -> other
    end
  end

  defp to_cpu_arg(%Nx.Tensor{data: %__MODULE__{}} = t) do
    bin = to_binary(t, Nx.byte_size(t))
    Nx.from_binary(bin, Nx.type(t)) |> Nx.reshape(Nx.shape(t))
  end

  defp to_cpu_arg(list) when is_list(list) do
    Enum.map(list, &to_cpu_arg/1)
  end

  defp to_cpu_arg(other), do: other

  # ── Lifecycle ──────────────────────────────────────

  @impl true
  def init(opts) do
    opts
  end

  # ── Data Transfer ──────────────────────────────────

  @impl true
  def from_binary(%Nx.Tensor{} = tensor, binary, _backend_opts) do
    ctx = get_ctx()
    ref = NxCL.Native.buffer_write(ctx, binary)
    put_in(tensor.data, %__MODULE__{ref: ref, ctx: ctx})
  end

  @impl true
  def to_binary(%Nx.Tensor{data: %__MODULE__{ref: ref}} = tensor, _limit) do
    NxCL.Native.buffer_read(ref, Nx.byte_size(tensor))
  end

  @impl true
  def inspect(%Nx.Tensor{} = tensor, opts) do
    binary = to_binary(tensor, Nx.byte_size(tensor))
    Nx.Backend.inspect(tensor, binary, opts)
  end

  @impl true
  def backend_deallocate(%Nx.Tensor{data: %__MODULE__{}}), do: :ok

  @impl true
  def backend_copy(tensor, Nx.BinaryBackend, _opts) do
    binary = to_binary(tensor, Nx.byte_size(tensor))

    Nx.BinaryBackend.from_binary(
      %{tensor | data: %Nx.BinaryBackend{}},
      binary,
      []
    )
  end

  def backend_copy(tensor, backend, opts) do
    binary = to_binary(tensor, Nx.byte_size(tensor))
    backend.from_binary(%{tensor | data: %{__struct__: backend}}, binary, opts)
  end

  @impl true
  def backend_transfer(tensor, backend, opts) do
    result = backend_copy(tensor, backend, opts)
    backend_deallocate(tensor)
    result
  end

  # ── Constant / Eye / Iota ──────────────────────────

  @impl true
  def constant(%Nx.Tensor{} = out, number, backend_opts) do
    binary = Nx.BinaryBackend.constant(%{out | data: %Nx.BinaryBackend{}}, number, [])
    bin = Nx.to_binary(binary)
    from_binary(out, bin, backend_opts)
  end

  @impl true
  def eye(%Nx.Tensor{} = out, backend_opts) do
    cpu = Nx.BinaryBackend.eye(%{out | data: %Nx.BinaryBackend{}}, [])
    bin = Nx.to_binary(cpu)
    from_binary(out, bin, backend_opts)
  end

  @impl true
  def iota(%Nx.Tensor{} = out, axis, backend_opts) do
    cpu = Nx.BinaryBackend.iota(%{out | data: %Nx.BinaryBackend{}}, axis, [])
    bin = Nx.to_binary(cpu)
    from_binary(out, bin, backend_opts)
  end

  # ── Shape operations (many are zero-copy metadata changes) ──

  @impl true
  def reshape(%Nx.Tensor{} = out, %Nx.Tensor{data: %__MODULE__{} = data} = _tensor) do
    # Reshape is metadata-only for contiguous tensors
    put_in(out.data, data)
  end

  @impl true
  def squeeze(out, tensor, _axes) do
    put_in(out.data, tensor.data)
  end

  @impl true
  def as_type(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor) do
    if Nx.type(out) == Nx.type(tensor) do
      put_in(out.data, tensor.data)
    else
      fallback(:as_type, [out, tensor])
    end
  end

  @impl true
  def bitcast(%Nx.Tensor{} = out, %Nx.Tensor{data: %__MODULE__{} = data}) do
    put_in(out.data, data)
  end

  # ── Elementwise Binary Ops ─────────────────────────

  @gpu_binary_ops %{
    add: "add",
    subtract: "subtract",
    multiply: "multiply",
    divide: "divide"
  }

  for {op, kernel_name} <- @gpu_binary_ops do
    @impl true
    def unquote(op)(%Nx.Tensor{} = out, %Nx.Tensor{} = left, %Nx.Tensor{} = right) do
      # If shapes match (no broadcasting needed), use GPU kernel directly
      if Nx.shape(left) == Nx.shape(out) and Nx.shape(right) == Nx.shape(out) and
           Nx.type(left) == {:f, 32} and Nx.type(right) == {:f, 32} do
        left = to_gpu(left)
        right = to_gpu(right)
        ctx = get_ctx()
        n = flat_size(out)

        ref =
          NxCL.Native.elementwise_binary_op(
            ctx,
            unquote(kernel_name),
            gpu_ref(left),
            gpu_ref(right),
            n
          )

        put_in(out.data, %NxCL.Backend{ref: ref, ctx: ctx})
      else
        # Broadcasting needed - fallback to CPU
        fallback(unquote(op), [out, left, right])
      end
    end
  end

  # ── Elementwise Unary Ops ──────────────────────────

  @gpu_unary_ops %{
    negate: "negate",
    exp: "exp",
    log: "log",
    tanh: "tanh",
    sigmoid: "sigmoid",
    abs: "abs"
  }

  for {op, kernel_name} <- @gpu_unary_ops do
    @impl true
    def unquote(op)(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor) do
      if Nx.type(tensor) == {:f, 32} do
        tensor = to_gpu(tensor)
        ctx = get_ctx()
        n = flat_size(out)

        ref =
          NxCL.Native.elementwise_unary_op(
            ctx,
            unquote(kernel_name),
            gpu_ref(tensor),
            n
          )

        put_in(out.data, %NxCL.Backend{ref: ref, ctx: ctx})
      else
        fallback(unquote(op), [out, tensor])
      end
    end
  end

  # ── Dot / Matmul ───────────────────────────────────

  @impl true
  def dot(
        %Nx.Tensor{} = out,
        %Nx.Tensor{} = left,
        [left_contract_axis],
        [],
        %Nx.Tensor{} = right,
        [right_contract_axis],
        []
      ) do
    left = to_gpu(left)
    right = to_gpu(right)

    left_shape = Nx.shape(left)
    right_shape = Nx.shape(right)

    # Only handle 2D matmul for now
    if tuple_size(left_shape) == 2 and tuple_size(right_shape) == 2 and
         left_contract_axis == 1 and right_contract_axis == 0 do
      {m, k} = left_shape
      {_k, n} = right_shape
      ctx = get_ctx()

      ref = NxCL.Native.matmul_op(ctx, gpu_ref(left), gpu_ref(right), m, n, k)
      put_in(out.data, %NxCL.Backend{ref: ref, ctx: ctx})
    else
      fallback(:dot, [out, left, [left_contract_axis], [], right, [right_contract_axis], []])
    end
  end

  def dot(out, left, left_contract, left_batch, right, right_contract, right_batch) do
    fallback(:dot, [out, left, left_contract, left_batch, right, right_contract, right_batch])
  end

  # ── Reductions ─────────────────────────────────────

  @impl true
  def sum(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    axes = opts[:axes]

    if axes == nil or axes == Nx.axes(tensor) do
      # Full reduction
      tensor = to_gpu(tensor)
      ctx = get_ctx()
      n = flat_size(tensor)

      ref = NxCL.Native.reduce_op(ctx, "sum", gpu_ref(tensor), n)
      put_in(out.data, %NxCL.Backend{ref: ref, ctx: ctx})
    else
      fallback(:sum, [out, tensor, opts])
    end
  end

  @impl true
  def reduce_max(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    axes = opts[:axes]

    if axes == nil or axes == Nx.axes(tensor) do
      tensor = to_gpu(tensor)
      ctx = get_ctx()
      n = flat_size(tensor)

      ref = NxCL.Native.reduce_op(ctx, "max", gpu_ref(tensor), n)
      put_in(out.data, %NxCL.Backend{ref: ref, ctx: ctx})
    else
      fallback(:reduce_max, [out, tensor, opts])
    end
  end

  @impl true
  def reduce_min(%Nx.Tensor{} = out, %Nx.Tensor{} = tensor, opts) do
    axes = opts[:axes]

    if axes == nil or axes == Nx.axes(tensor) do
      tensor = to_gpu(tensor)
      ctx = get_ctx()
      n = flat_size(tensor)

      ref = NxCL.Native.reduce_op(ctx, "min", gpu_ref(tensor), n)
      put_in(out.data, %NxCL.Backend{ref: ref, ctx: ctx})
    else
      fallback(:reduce_min, [out, tensor, opts])
    end
  end

  # ── To Batched ─────────────────────────────────────

  @impl true
  def to_batched(out, tensor, opts) do
    fallback(:to_batched, [out, tensor, opts])
  end

  # ── Fallback for everything else ───────────────────
  # All required callbacks that we haven't implemented on GPU
  # fall back to CPU computation.

  @fallback_binary_ops [
    :pow,
    :remainder,
    :atan2,
    :min,
    :max,
    :quotient,
    :bitwise_and,
    :bitwise_or,
    :bitwise_xor,
    :left_shift,
    :right_shift,
    :equal,
    :not_equal,
    :greater,
    :less,
    :greater_equal,
    :less_equal,
    :logical_and,
    :logical_or,
    :logical_xor
  ]

  for op <- @fallback_binary_ops do
    @impl true
    def unquote(op)(out, left, right) do
      fallback(unquote(op), [out, left, right])
    end
  end

  @all_unary_ops Enum.map(Nx.Shared.unary_math_funs(), &elem(&1, 0)) ++
                   [
                     :bitwise_not,
                     :ceil,
                     :conjugate,
                     :floor,
                     :round,
                     :sign,
                     :count_leading_zeros,
                     :population_count,
                     :real,
                     :imag,
                     :is_nan,
                     :is_infinity
                   ]

  @fallback_unary_ops @all_unary_ops -- Map.keys(@gpu_unary_ops)

  for op <- @fallback_unary_ops do
    @impl true
    def unquote(op)(out, tensor) do
      fallback(unquote(op), [out, tensor])
    end
  end

  @impl true
  def broadcast(out, tensor, shape, axes) do
    fallback(:broadcast, [out, tensor, shape, axes])
  end

  @impl true
  def transpose(out, tensor, axes) do
    fallback(:transpose, [out, tensor, axes])
  end

  @impl true
  def pad(out, tensor, pad_value, padding_config) do
    fallback(:pad, [out, tensor, pad_value, padding_config])
  end

  @impl true
  def reverse(out, tensor, axes) do
    fallback(:reverse, [out, tensor, axes])
  end

  @impl true
  def clip(out, tensor, min, max) do
    fallback(:clip, [out, tensor, min, max])
  end

  @impl true
  def slice(out, tensor, starts, lengths, strides) do
    fallback(:slice, [out, tensor, starts, lengths, strides])
  end

  @impl true
  def put_slice(out, tensor, start_tensor, starts) do
    fallback(:put_slice, [out, tensor, start_tensor, starts])
  end

  @impl true
  def gather(out, input, indices, opts) do
    fallback(:gather, [out, input, indices, opts])
  end

  @impl true
  def concatenate(out, tensors, axis) do
    fallback(:concatenate, [out, tensors, axis])
  end

  @impl true
  def stack(out, tensors, axis) do
    fallback(:stack, [out, tensors, axis])
  end

  @impl true
  def select(out, pred, on_true, on_false) do
    fallback(:select, [out, pred, on_true, on_false])
  end

  @impl true
  def conv(out, tensor, kernel, opts) do
    fallback(:conv, [out, tensor, kernel, opts])
  end

  @impl true
  def all(out, tensor, opts), do: fallback(:all, [out, tensor, opts])

  @impl true
  def any(out, tensor, opts), do: fallback(:any, [out, tensor, opts])

  @impl true
  def product(out, tensor, opts), do: fallback(:product, [out, tensor, opts])

  @impl true
  def argmax(out, tensor, opts), do: fallback(:argmax, [out, tensor, opts])

  @impl true
  def argmin(out, tensor, opts), do: fallback(:argmin, [out, tensor, opts])

  @impl true
  def reduce(out, tensor, acc, opts, fun) do
    fallback(:reduce, [out, tensor, acc, opts, fun])
  end

  @impl true
  def window_reduce(out, tensor, acc, shape, opts, fun) do
    fallback(:window_reduce, [out, tensor, acc, shape, opts, fun])
  end

  @impl true
  def window_sum(out, tensor, shape, opts) do
    fallback(:window_sum, [out, tensor, shape, opts])
  end

  @impl true
  def window_product(out, tensor, shape, opts) do
    fallback(:window_product, [out, tensor, shape, opts])
  end

  @impl true
  def window_max(out, tensor, shape, opts) do
    fallback(:window_max, [out, tensor, shape, opts])
  end

  @impl true
  def window_min(out, tensor, shape, opts) do
    fallback(:window_min, [out, tensor, shape, opts])
  end

  @impl true
  def sort(out, tensor, opts), do: fallback(:sort, [out, tensor, opts])

  @impl true
  def argsort(out, tensor, opts), do: fallback(:argsort, [out, tensor, opts])

  @impl true
  def window_scatter_max(out, tensor, source, init, shape, opts) do
    fallback(:window_scatter_max, [out, tensor, source, init, shape, opts])
  end

  @impl true
  def window_scatter_min(out, tensor, source, init, shape, opts) do
    fallback(:window_scatter_min, [out, tensor, source, init, shape, opts])
  end

  @impl true
  def indexed_add(out, tensor, indices, updates, opts) do
    fallback(:indexed_add, [out, tensor, indices, updates, opts])
  end

  @impl true
  def indexed_put(out, tensor, indices, updates, opts) do
    fallback(:indexed_put, [out, tensor, indices, updates, opts])
  end

  @impl true
  def fft(out, tensor, opts), do: fallback(:fft, [out, tensor, opts])

  @impl true
  def ifft(out, tensor, opts), do: fallback(:ifft, [out, tensor, opts])

  @impl true
  def triangular_solve(out, a, b, opts) do
    fallback(:triangular_solve, [out, a, b, opts])
  end

  # Nx 1.0 callback: run the block's default implementation.
  @impl true
  def block(struct, _out, args, fun), do: apply(fun, [struct | args])

  @impl true
  def from_pointer(_out, _pointer, _backend_opts, _offset, _byte_size) do
    raise "NxCL does not support from_pointer"
  end

  @impl true
  def to_pointer(_tensor, _opts) do
    raise "NxCL does not support to_pointer"
  end
end

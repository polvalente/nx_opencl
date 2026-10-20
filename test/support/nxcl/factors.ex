defmodule NxCL.TestSupport.Factors do
  @moduledoc """
  Checks for `qr`, `eigh`, `svd` and `lu`, whose factors are not unique: a
  column of `Q`, an eigenvector or a singular vector may flip sign. The device
  runs the block's default implementation, which can pick other signs than
  `Nx.BinaryBackend`, so the checks compare what does not change:

    * `qr`: `Q R == A`, `Q^H Q == I`, `R` upper triangular.
    * `eigh`: `A V == V diag(w)`, `V^H V == I`, `w` equal (sorted) to the `w` of
      the same function on the input widened to `f64` or `c128`.
    * `svd`: `U diag(S) V^H == A`, `U` and `V` orthonormal, `S` in descending
      order and equal (sorted) to the `S` on the widened input.
    * `lu`: `P L U == A`, `P` a permutation matrix (entries 0 or 1, `P^T P == I`),
      `L` unit lower triangular, `U` upper triangular.

  Every result must also have the reference's types and shapes. Each check
  measures the largest absolute error of one identity (the residual).

    * `f32`, `f64`, `c64`, `c128`: the tolerance is the one in
      `NxCL.TestSupport.Compare`, scaled by `max(1, max |A|)`. The device
      residual may exceed the reference residual by at most that.
    * `f16`, `bf16`: the tolerance is `8 * eps * max(m, n) * max(1, max |A|)` and
      ignores the reference residual: a half type rounds at every step, so the
      reference residual is large and random.

  The reference truncates a half type and the device rounds to nearest, so `w`
  and `S` are measured against the widened input's, not the reference's. A check
  computes in `f64` or `c128` on `Nx.BinaryBackend`, so it adds no rounding.

  Each check runs through the compiler and, as `Compare` does, eagerly.
  """

  import ExUnit.Assertions

  @tol %{{:f, 64} => 1.0e-12, {:f, 32} => 1.0e-5}

  @half_epsilon %{{:f, 16} => :math.pow(2, -10), {:bf, 16} => :math.pow(2, -7)}

  # The factor 8 is about twice the largest residual over 400 seeds of each kind on
  # 3 x 3 to 4 x 5 `bf16` inputs. A `bf16` `R` of a 5-column input has entries of 0.3
  # below the diagonal, so a tolerance tied to the reference residual fails at some seeds.
  @half_factor 8

  # The default `svd` stops at a tolerance of `1.19e-7` for every type, so `U S V^H == A`
  # is off by up to about `1.4e-4 * max |A|` in `f32` and `f64` (over 1600 `f32` and
  # 400 `f64` cases), also on the reference: `Nx.BinaryBackend` misses it by 2e-4 in
  # `f32`. In `f64` the device and the reference take the same steps. In `f32` they
  # round differently and take different steps: the reference can sit at 5e-6 where the
  # device sits at 6e-4 (seed 77310, a 4 x 3 input). So for `f32` that identity also
  # passes below `4e-4 * max |A|`, about three times the largest residual seen.
  @svd_f32_floor 4.0e-4

  @type kind :: :qr | :eigh | :svd | :lu

  @doc """
  Runs `fun` on `x` and checks the factors of `kind`. `fun` is
  `&Nx.LinAlg.qr/1`, `&Nx.LinAlg.qr(&1, mode: :complete)`, and so on.

  The option `:reference_fun` (default `fun`) is the function the widened input and the
  reference run. A test that checks the check itself passes a wrong `fun` and the right
  `:reference_fun`, so the wrong factor shows only in the compiler and eager runs.
  """
  @spec assert_factors(kind(), function(), Nx.Tensor.t(), keyword()) :: :ok
  def assert_factors(kind, fun, x, opts \\ []) do
    ref_fun = Keyword.get(opts, :reference_fun, fun)
    wide = widen(x)
    truth = apply(Nx.Defn.jit(ref_fun, compiler: Nx.Defn.Evaluator), [wide])

    reference = apply(Nx.Defn.jit(ref_fun, compiler: Nx.Defn.Evaluator), [to_binary(x)])
    actual = apply(Nx.Defn.jit(fun, compiler: NxCL.Compiler), [x])
    check(kind, x, {reference, truth}, actual, "compiler")

    eager =
      Nx.with_default_backend(NxCL.Backend, fn ->
        apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), [Nx.backend_copy(x, NxCL.Backend)])
      end)

    check(kind, x, {reference, truth}, eager, "backend")
  end

  defp check(kind, x, {reference, truth}, actual, path) do
    actual = actual |> Tuple.to_list() |> Enum.map(&to_binary/1)
    expected = reference |> Tuple.to_list() |> Enum.map(&to_binary/1)

    for {e, a} <- Enum.zip(expected, actual) do
      assert {a.type, a.shape} == {e.type, e.shape},
             "#{kind} (#{path}): got #{inspect({a.type, a.shape})}, expected #{inspect({e.type, e.shape})}"
    end

    type = x |> Nx.type() |> Nx.Type.to_floating()
    a = widen(x)
    max_a = max(1.0, scalar(Nx.reduce_max(Nx.abs(a))))
    tol = tolerance(type, Nx.shape(x), max_a)
    truth = truth |> Tuple.to_list() |> Enum.map(&widen/1)
    ours = residuals(kind, a, Enum.map(actual, &widen/1), truth)
    theirs = residuals(kind, a, Enum.map(expected, &widen/1), truth)

    for {{label, err}, {_, ref_err}} <- Enum.zip(ours, theirs) do
      allowed = allowed(kind, label, type, tol, ref_err, max_a)

      assert err <= allowed,
             "#{kind} (#{path}): #{label}: max error #{err}, reference #{ref_err}, tolerance #{allowed} (seed #{NxCL.TestSupport.Compare.seed()})"
    end

    :ok
  end

  defp residuals(:qr, a, [q, r], _truth) do
    [
      {"Q R == A", err(mm(q, r), a)},
      {"Q^H Q == I", err(mm(ct(q), q), identity(q, cols(q)))},
      {"R upper triangular", err(Nx.multiply(r, below_diagonal(r)), zeros(r))}
    ]
  end

  defp residuals(:eigh, a, [w, v], [tw, _]) do
    [
      {"A V == V diag(w)", err(mm(a, v), Nx.multiply(v, Nx.new_axis(w, -2)))},
      {"V^H V == I", err(mm(ct(v), v), identity(v, cols(v)))},
      {"w == w of the widened input",
       err(Nx.sort(Nx.real(w), axis: -1), Nx.sort(Nx.real(tw), axis: -1))}
    ]
  end

  defp residuals(:svd, a, [u, s, vt], [_, ts, _]) do
    k = Nx.axis_size(s, -1)
    uk = Nx.slice_along_axis(u, 0, k, axis: -1)
    vk = Nx.slice_along_axis(vt, 0, k, axis: -2)

    [
      {"U S V^H == A", err(mm(Nx.multiply(uk, Nx.new_axis(s, -2)), vk), a)},
      {"U^H U == I", err(mm(ct(u), u), identity(u, cols(u)))},
      {"V^H V == I", err(mm(vt, ct(vt)), identity(vt, cols(vt)))},
      {"S == S of the widened input", err(Nx.sort(s, axis: -1), Nx.sort(ts, axis: -1))},
      {"S descending", ascent(s)}
    ]
  end

  defp residuals(:lu, a, [p, l, u], _truth) do
    [
      {"P L U == A", err(mm(mm(p, l), u), a)},
      {"P entries are 0 or 1", err(Nx.multiply(p, Nx.subtract(p, 1)), zeros(p))},
      {"P^T P == I", err(mm(ct(p), p), identity(p, cols(p)))},
      {"L unit lower triangular",
       err(Nx.multiply(l, above_or_on_diagonal(l)), identity(l, cols(l)))},
      {"U upper triangular", err(Nx.multiply(u, below_diagonal(u)), zeros(u))}
    ]
  end

  defp err(actual, expected),
    do: actual |> Nx.subtract(expected) |> Nx.abs() |> Nx.reduce_max() |> scalar()

  # The largest step up between neighbors of `s`: 0 when `s` is in descending order.
  defp ascent(s) do
    k = Nx.axis_size(s, -1)

    if k < 2 do
      0.0
    else
      s
      |> Nx.slice_along_axis(1, k - 1, axis: -1)
      |> Nx.subtract(Nx.slice_along_axis(s, 0, k - 1, axis: -1))
      |> Nx.max(0)
      |> Nx.reduce_max()
      |> scalar()
    end
  end

  defp scalar(t), do: t |> Nx.real() |> Nx.to_number()

  # Half types use the tolerance alone; the others add the reference's error.
  defp allowed(kind, label, type, tol, ref_err, max_a) do
    cond do
      Map.has_key?(@half_epsilon, type) ->
        tol

      kind == :svd and type == {:f, 32} and label == "U S V^H == A" ->
        max(tol + ref_err, @svd_f32_floor * max_a)

      true ->
        tol + ref_err
    end
  end

  # Loose for half types: about 20% of max |A| at n = 3 to 5 (8 * eps * n, eps = 2^-7).
  # A `bf16` check catches only gross defects (a wrong sign, a missing row).
  defp tolerance(type, shape, max_a) do
    case @half_epsilon do
      %{^type => eps} ->
        n = max(elem(shape, tuple_size(shape) - 1), elem(shape, tuple_size(shape) - 2))
        @half_factor * eps * n * max_a

      _ ->
        tol(type) * max_a
    end
  end

  defp tol({:c, size}), do: Map.fetch!(@tol, {:f, div(size, 2)})
  defp tol(type), do: Map.fetch!(@tol, type)

  defp to_binary(t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  defp widen(t) do
    t = to_binary(t)
    Nx.as_type(t, if(Nx.Type.complex?(Nx.type(t)), do: {:c, 128}, else: {:f, 64}))
  end

  defp rank(t), do: tuple_size(Nx.shape(t))
  defp cols(t), do: Nx.axis_size(t, -1)

  defp mm(a, b) do
    batch = Enum.to_list(0..(rank(a) - 3)//1)
    Nx.dot(a, [rank(a) - 1], batch, b, [rank(b) - 2], batch)
  end

  defp ct(t) do
    axes = Enum.to_list(0..(rank(t) - 3)//1) ++ [rank(t) - 1, rank(t) - 2]
    t = Nx.transpose(t, axes: axes)
    if Nx.Type.complex?(Nx.type(t)), do: Nx.conjugate(t), else: t
  end

  defp identity(t, n) do
    shape = t |> Nx.shape() |> Tuple.delete_at(rank(t) - 1) |> Tuple.delete_at(rank(t) - 2)
    eye = Nx.eye(n, type: Nx.type(t))
    Nx.broadcast(eye, List.to_tuple(Tuple.to_list(shape) ++ [n, n]))
  end

  defp zeros(t), do: Nx.multiply(t, 0)

  defp below_diagonal(t), do: mask(t, &Nx.greater/2)
  defp above_or_on_diagonal(t), do: mask(t, &Nx.less_equal/2)

  defp mask(t, cmp) do
    shape = Nx.shape(t)
    r = Nx.iota(shape, axis: rank(t) - 2)
    c = Nx.iota(shape, axis: rank(t) - 1)
    cmp.(r, c) |> Nx.as_type(Nx.type(t))
  end
end

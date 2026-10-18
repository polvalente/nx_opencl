defmodule NxCL.DoctestSkips do
  @moduledoc """
  The doctests of Nx that the suite leaves out, one reason per entry.

  `@skips` lists functions that every device skips; `@needs` lists functions
  that a device skips only when it lacks the extension (their examples use
  `f64`, or an `s64` scatter). A device with the extension runs them. Each
  entry is `{{function, arity}, reason}`; `@needs` entries carry the capability
  first. Doctests skip a whole function, since `ExUnit.DocTest` cannot skip one
  example.

  The reasons fall in four groups:

    * the expected text prints more digits than a device result keeps (the last
      digit of an `f64`, rounding noise in a `c64` transform). The differential
      rows for the op compare within a tolerance;
    * the reference text is not the true value (`erf_inv` on `f64`);
    * the example needs a dtype that is out of scope (see the README);
    * the example needs `f64` or 64-bit atomics, and the device has neither.

  An entry never loosens a tolerance: it means the example does not run.
  """

  @f64_digits "The f64 example prints 16 digits of a libm result; the device differs in the last digit. The differential rows for the op compare within a tolerance."
  @c64_noise "The c64 transform prints rounding noise near 1e-7 where the text shows exact zeros (the reference computes in f64 and rounds). The FFT rows compare within a tolerance."
  @f32_digit "The f32 example prints the last digit of a device libm result, which differs from the text. The differential rows compare within a tolerance."

  @skips %{
    Nx => [
      {{:as_type, 2}, "An example casts to f16; no device in the test set has cl_khr_fp16."},
      {{:asin, 1}, @f64_digits},
      {{:atan, 1}, @f64_digits},
      {{:erf, 1}, @f64_digits},
      {{:erfc, 1}, @f64_digits},
      {{:exp, 1}, @f64_digits},
      {{:expm1, 1}, @f64_digits},
      {{:sin, 1}, @f64_digits},
      {{:tan, 1}, @f64_digits},
      {{:atan2, 2}, @f32_digit},
      {{:log, 2}, @f32_digit},
      {{:logsumexp, 2}, @f32_digit},
      {{:erf_inv, 1},
       "The text shows an f64 result that is only right to single precision (3e-8 off); the device gives the true value, which the elementwise rows check."},
      {{:fft, 2}, @c64_noise},
      {{:ifft, 2}, @c64_noise},
      {{:fft2, 2}, @c64_noise},
      {{:ifft2, 2}, @c64_noise},
      {{:rfft, 2}, @c64_noise},
      {{:irfft, 2}, @c64_noise},
      {{:bit_size, 1}, "An example builds an s4 tensor; sub-byte types are out of scope."}
    ],
    Nx.LinAlg => [
      {{:cholesky, 1}, @f32_digit},
      {{:least_squares, 3}, @f32_digit},
      {{:lu, 1}, @f32_digit},
      {{:matrix_power, 2}, @f32_digit},
      {{:pinv, 2}, @f32_digit},
      {{:solve, 2}, @f32_digit},
      {{:triangular_solve, 3}, @f32_digit}
    ],
    Nx.Random => [
      {{:gumbel, 2}, @f32_digit},
      {{:multivariate_normal, 4}, @f32_digit <> " An example also uses f16."},
      {{:normal, 4}, @f32_digit <> " An example also uses f16."},
      {{:uniform, 4}, "An example uses f16; no device in the test set has cl_khr_fp16."}
    ]
  }

  @needs %{
    Nx => [
      {:fp64, {:acos, 1}, "The examples use f64."},
      {:fp64, {:acosh, 1}, "The examples use f64."},
      {:fp64, {:all_close, 3}, "An example uses f64."},
      {:fp64, {:asinh, 1}, "The examples use f64."},
      {:fp64, {:atanh, 1}, "The examples use f64."},
      {:fp64, {:cbrt, 1}, "The examples use f64."},
      {:fp64, {:cos, 1}, "The examples use f64."},
      {:fp64, {:cosh, 1}, "The examples use f64."},
      {:fp64, {:iota, 2}, "An example uses f64."},
      {:fp64, {:log, 1}, "The examples use f64."},
      {:fp64, {:log1p, 1}, "The examples use f64."},
      {:fp64, {:rsqrt, 1}, "The examples use f64."},
      {:fp64, {:sigmoid, 1}, "The examples use f64."},
      {:fp64, {:sinh, 1}, "The examples use f64."},
      {:fp64, {:sqrt, 1}, "The examples use f64."},
      {:fp64, {:tanh, 1}, "The examples use f64."},
      {:int64_atomics, {:indexed_add, 4}, "An example adds into an s64 tensor."}
    ]
  }

  @doc """
  The `except:` list for `doctest mod`: the skips of every device, and the ones
  whose capability this device lacks.
  """
  @spec except(module()) :: [{atom(), non_neg_integer()}]
  def except(mod) do
    always = for {fun, _reason} <- Map.get(@skips, mod, []), do: fun

    gated =
      for {cap, fun, _reason} <- Map.get(@needs, mod, []),
          not NxCL.TestSupport.Device.capability?(cap),
          do: fun

    always ++ gated
  end

  @doc "Every entry as `{module, {name, arity}, requires, reason}`; `requires` is `nil` for a skip on every device."
  @spec all() :: [{module(), {atom(), non_neg_integer()}, atom() | nil, String.t()}]
  def all do
    always = for {mod, entries} <- @skips, {fun, reason} <- entries, do: {mod, fun, nil, reason}

    gated =
      for {mod, entries} <- @needs, {cap, fun, reason} <- entries, do: {mod, fun, cap, reason}

    always ++ gated
  end
end

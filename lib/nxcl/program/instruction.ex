defmodule NxCL.Program.Instruction do
  @moduledoc """
  One step of an `NxCL.Program`.

  Fields:

    * `:op` - the op name, an atom such as `:add` or `:batched_gemm`.
    * `:operands` - the `t:ref/0` values the op reads.
    * `:out_type` - the Nx type of the result, for example `{:f, 32}`.
    * `:out_shape` - the shape of the result, a tuple.
    * `:attrs` - op-specific extras (`t:attr/0`). A `:launch` instruction carries
      kernel source here as a binary (see `NxCL.Program.Builder.launch/6`).
    * `:subprograms` - `NxCL.Program.SubProgram` values for `cond` and `while`.
      A `while` has `[cond, body]`; a `cond` has `[pred_1, body_1, ..., else]`
      (see `NxCL.Lower.Control`). Empty for every other op.

  A `host_call` runs an Elixir function on host copies of its operands. Its one
  attr indexes `NxCL.Program`'s `:host_calls` table (see `NxCL.Lower.HostCall`).

  A `cond`, `while` or `host_call` has one result per output. `:out_type` and `:out_shape` are
  lists, and the k-th result is `{:result, i, k}`. A `while` with the attrs
  `[:static, n]` runs its body `n` times without reading the condition.

  A ref points at a value that already exists:

    * `{:input, i}` - the i-th flattened argument.
    * `{:capture, i}` - the i-th literal tensor of the program.
    * `{:const, i}` - the i-th scalar constant of the program.
    * `{:result, i}` - the result of the i-th instruction.
    * `{:result, i, k}` - the k-th result of the i-th instruction, a `cond`, `while` or `host_call`.
    * `{:carry, i}` - the i-th loop-carry slot (inside a `while` or `cond` sub-program).

  An attr is an integer, an atom, or a binary.

  `reshape` and `squeeze` carry the attr `:alias`. Their result shares the
  operand's buffer, so a runtime must not free it twice.
  """

  @enforce_keys [:op, :operands, :out_type, :out_shape]
  defstruct [:op, :operands, :out_type, :out_shape, attrs: [], subprograms: []]

  @type ref ::
          {:input | :capture | :const | :result | :carry, non_neg_integer()}
          | {:result, non_neg_integer(), non_neg_integer()}

  @type attr :: integer() | atom() | binary()

  @type t :: %__MODULE__{
          op: atom(),
          operands: [ref()],
          out_type: Nx.Type.t() | [Nx.Type.t()],
          out_shape: tuple() | [tuple()],
          attrs: [attr()],
          subprograms: [NxCL.Program.SubProgram.t()]
        }
end

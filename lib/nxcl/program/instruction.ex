defmodule NxCL.Program.Instruction do
  @moduledoc """
  One step of an `NxCL.Program`.

  Fields:

    * `:op` - the op name, an atom such as `:add` or `:dot`.
    * `:operands` - the `t:ref/0` values the op reads.
    * `:out_type` - the Nx type of the result, for example `{:f, 32}`.
    * `:out_shape` - the shape of the result, a tuple.
    * `:attrs` - op-specific extras (`t:attr/0`). Stage 03 adds kernel source
      here as a binary.
    * `:subprograms` - `NxCL.Program.SubProgram` values for `cond` and `while`.
      Empty until stage 10.

  A ref points at a value that already exists:

    * `{:input, i}` - the i-th flattened argument.
    * `{:capture, i}` - the i-th literal tensor of the program.
    * `{:const, i}` - the i-th scalar constant of the program.
    * `{:result, i}` - the result of the i-th instruction.
    * `{:carry, i}` - the i-th loop-carry slot (inside a `while` sub-program).

  An attr is an integer, an atom, or a binary.

  `reshape` and `squeeze` carry the attr `:alias`. Their result shares the
  operand's buffer, so a runtime must not free it twice.
  """

  @enforce_keys [:op, :operands, :out_type, :out_shape]
  defstruct [:op, :operands, :out_type, :out_shape, attrs: [], subprograms: []]

  @type ref ::
          {:input | :capture | :const | :result | :carry, non_neg_integer()}

  @type attr :: integer() | atom() | binary()

  @type t :: %__MODULE__{
          op: atom(),
          operands: [ref()],
          out_type: Nx.Type.t(),
          out_shape: tuple(),
          attrs: [attr()],
          subprograms: [NxCL.Program.SubProgram.t()]
        }
end

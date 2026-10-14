defmodule NxCL.Program.SubProgram do
  @moduledoc """
  The condition or body of a `cond` or `while` instruction, as its own flat
  instruction list.

  `{:result, i}` refs are local to the sub-program. `{:carry, i}` refs read the
  loop-carry slots of a `while`, or the operands of a `cond`. `{:capture, i}` and
  `{:const, i}` use the parent's tables. A sub-program reads nothing else from
  its parent.
  """

  @enforce_keys [:instructions, :outputs]
  defstruct [:instructions, :outputs]

  @type t :: %__MODULE__{
          instructions: [NxCL.Program.Instruction.t()],
          outputs: [NxCL.Program.Instruction.ref()]
        }
end

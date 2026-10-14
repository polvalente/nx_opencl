defmodule NxCL.Program.SubProgram do
  @moduledoc """
  The condition or body of a `cond` or `while` instruction, as its own flat
  instruction list. Unused until stage 10.

  `{:result, i}` refs are local to the sub-program. `{:carry, i}` refs read the
  loop-carry slots. `{:capture, i}` and `{:const, i}` use the parent's tables.
  """

  @enforce_keys [:instructions, :outputs]
  defstruct [:instructions, :outputs]

  @type t :: %__MODULE__{
          instructions: [NxCL.Program.Instruction.t()],
          outputs: [NxCL.Program.Instruction.ref()]
        }
end

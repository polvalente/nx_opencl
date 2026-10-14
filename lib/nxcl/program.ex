defmodule NxCL.Program do
  @moduledoc """
  A lowered `defn`: a flat list of typed instructions.

  `NxCL.Lower.lower/2` builds it from an `Nx.Defn.Expr` graph. A runtime
  reads it without looking at the graph again.

  Fields:

    * `:inputs` - one entry per flattened argument, in argument order. An entry
      is `{type, shape}`, or `nil` when the function never reads that argument.
    * `:captures` - literal tensors the graph holds, read as `{:capture, i}`.
    * `:constants` - scalar constants as `{value, type}`, read as `{:const, i}`.
    * `:instructions` - `NxCL.Program.Instruction` values in run order.
    * `:outputs` - one ref per flattened output leaf.
    * `:output_template` - the output container with template tensors, used to
      rebuild tuples and maps from `:outputs`.
    * `:ref_counts` - how often each ref is read, by instructions or as an
      output. Stage 03 uses it to free buffers.

  See `NxCL.Program.Instruction` for the ref and attr types.
  """

  alias NxCL.Program.Instruction

  @enforce_keys [:inputs, :instructions, :outputs, :output_template]
  defstruct [
    :inputs,
    :instructions,
    :outputs,
    :output_template,
    captures: [],
    constants: [],
    ref_counts: %{}
  ]

  @type t :: %__MODULE__{
          inputs: [{Nx.Type.t(), tuple()} | nil],
          captures: [Nx.Tensor.t()],
          constants: [{number() | atom(), Nx.Type.t()}],
          instructions: [Instruction.t()],
          outputs: [Instruction.ref()],
          output_template: Nx.Container.t(),
          ref_counts: %{Instruction.ref() => non_neg_integer()}
        }
end

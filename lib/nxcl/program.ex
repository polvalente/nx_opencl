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
    * `:host_calls` - side table for `:host_call` instructions (callbacks and
      templates). An instruction names its entry by index in its attrs.
    * `:outputs` - one ref per flattened output leaf.
    * `:output_template` - the output container with template tensors, used to
      rebuild tuples and maps from `:outputs`.
    * `:ref_counts` - how often each ref is read, by instructions or as an
      output. The native executor derives the same counts from the encoded
      program to free buffers at their last reader.
    * `:signature` - the value `signature/1` returns, kept by the compiler so a
      cached program is not hashed again on each call.

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
    host_calls: [],
    ref_counts: %{},
    signature: nil
  ]

  @type t :: %__MODULE__{
          inputs: [{Nx.Type.t(), tuple()} | nil],
          captures: [Nx.Tensor.t()],
          constants: [{number() | atom(), Nx.Type.t()}],
          host_calls: [map()],
          instructions: [Instruction.t()],
          outputs: [Instruction.ref()],
          output_template: Nx.Container.t(),
          ref_counts: %{Instruction.ref() => non_neg_integer()},
          signature: binary() | nil
        }

  @doc """
  A hash of what decides the compiled program: the input and capture types and
  shapes, the constants, and every instruction with its sub-programs. It holds no
  expression ids and no tensor values, so two functions that lower to the same
  program have the same signature.
  """
  @spec signature(t()) :: binary()
  def signature(%__MODULE__{signature: signature}) when is_binary(signature), do: signature

  def signature(%__MODULE__{} = program) do
    term =
      {program.inputs, Enum.map(program.captures, &{&1.type, &1.shape}), program.constants,
       Enum.map(program.instructions, &instruction_signature/1), program.outputs}

    :erlang.md5(:erlang.term_to_binary(term))
  end

  defp instruction_signature(%Instruction{} = instr) do
    subs =
      Enum.map(instr.subprograms, fn sub ->
        {Enum.map(sub.instructions, &instruction_signature/1), sub.outputs}
      end)

    {instr.op, instr.operands, instr.out_type, instr.out_shape, instr.attrs, subs}
  end
end

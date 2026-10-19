defmodule NxCL.Compiler.Instruction do
  @moduledoc """
  One step of an `NxCL.Compiler.Program`.

  Fields:

    * `:op` - the op name, an atom such as `:add` or `:batched_gemm`.
    * `:operands` - the `t:ref/0` values the op reads.
    * `:out_type` - the Nx type of the result, for example `{:f, 32}`.
    * `:out_shape` - the shape of the result, a tuple.
    * `:attrs` - op-specific extras (`t:attr/0`). Mostly integers and atoms that
      the builders in `NxCL.Compiler.Ops` document per op. The raw `:launch` op
      carries kernel source as a binary (`NxCL.Compiler.Ops.launch/6`).

  A `host_call` runs an Elixir function on host copies of its operands. Its one
  attr indexes `NxCL.Compiler.Program`'s `:host_calls` table (see
  `NxCL.Compiler.Lower.HostCall`).

  A `cond`, `while` or `host_call` has one result per output. `:out_type` and `:out_shape` are
  lists, and the k-th result is `{:result, i, k}`.

  ## Nested instructions

  A `cond` and a `while` hold their bodies as instruction lists in `:attrs`. A
  body is a `t:body/0`: `{instructions, outputs}`, where `outputs` are the refs
  the body returns. Inside a body `{:result, i}` is local to the body,
  `{:carry, i}` reads the i-th operand of the holding instruction (the loop-carry
  slot of a `while`), and `{:capture, i}` and `{:const, i}` use the program's
  tables. A body reads nothing else from its parent.

    * `while`: `[cond: body, body: body]`, plus `static: n` when the trip count
      is known, so the body runs `n` times without reading the condition.
    * `cond`: `[clauses: [{pred_body, body}, ...], else: body]`.

  A ref points at a value that already exists:

    * `{:input, i}` - the i-th flattened argument.
    * `{:capture, i}` - the i-th literal tensor of the program.
    * `{:const, i}` - the i-th scalar constant of the program.
    * `{:result, i}` - the result of the i-th instruction.
    * `{:result, i, k}` - the k-th result of the i-th instruction, a `cond`, `while` or `host_call`.
    * `{:carry, i}` - the i-th operand of the enclosing `while` or `cond`, inside its bodies.

  `reshape` and `squeeze` carry the attr `:alias`. Their result shares the
  operand's buffer, so a runtime must not free it twice.
  """

  @enforce_keys [:op, :operands, :out_type, :out_shape]
  defstruct [:op, :operands, :out_type, :out_shape, attrs: []]

  @type ref ::
          {:input | :capture | :const | :result | :carry, non_neg_integer()}
          | {:result, non_neg_integer(), non_neg_integer()}

  @type body :: {[t()], [ref()]}

  @type attr ::
          integer()
          | atom()
          | binary()
          | {:cond | :body | :else, body()}
          | {:clauses, [{body(), body()}]}
          | {:static, non_neg_integer()}

  @type t :: %__MODULE__{
          op: atom(),
          operands: [ref()],
          out_type: Nx.Type.t() | [Nx.Type.t()],
          out_shape: tuple() | [tuple()],
          attrs: [attr()]
        }
end

defmodule NxCL.Compiler.Program do
  @moduledoc """
  A lowered `defn`: a flat list of typed instructions.

  `NxCL.Compiler.Lower.lower/2` builds it from an `Nx.Defn.Expr` graph. A runtime
  reads it without looking at the graph again.

  Fields:

    * `:inputs` - one entry per flattened argument, in argument order. An entry
      is `{type, shape}`, or `nil` when the function never reads that argument.
    * `:captures` - literal tensors the graph holds, read as `{:capture, i}`.
    * `:constants` - scalar constants as `{value, type}`, read as `{:const, i}`.
    * `:instructions` - `NxCL.Compiler.Instruction` values in run order.
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

  See `NxCL.Compiler.Instruction` for the ref and attr types.
  """

  alias NxCL.Compiler.Instruction

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

  @typedoc """
  A `:host_calls` entry; see `NxCL.Compiler.Lower.HostCall`. `:fun` is `nil`
  for an `:io_call` with no callback.
  """
  @type host_call :: %{
          kind: :io_call | :runtime_call,
          fun: function() | nil,
          opts: keyword(),
          in_template: Nx.Container.t(),
          out_template: Nx.Container.t()
        }

  @type t :: %__MODULE__{
          inputs: [{Nx.Type.t(), tuple()} | nil],
          captures: [Nx.Tensor.t()],
          constants: [{number() | :nan | :infinity | :neg_infinity | Complex.t(), Nx.Type.t()}],
          host_calls: [host_call()],
          instructions: [Instruction.t()],
          outputs: [Instruction.ref()],
          output_template: Nx.Container.t(),
          ref_counts: %{Instruction.ref() => non_neg_integer()},
          signature: binary() | nil
        }

  @doc """
  A hash of what decides the compiled program: the input and capture types and
  shapes, the constants, and every instruction with its nested bodies. It holds no
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

  defp instruction_signature(%Instruction{} = instr),
    do: {instr.op, instr.operands, instr.out_type, instr.out_shape, instr.attrs}
end

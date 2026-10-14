defmodule NxCL.Compiler.FlatOp do
  @moduledoc false

  defstruct [
    :id,
    :op,
    :args,
    :shape,
    :type,
    :consumers,
    # For :parameter ops
    :param_index,
    # For :constant ops
    :value,
    # For :tensor ops (literal tensors)
    :tensor,
    # For :dot ops
    :contract_axes,
    :batch_axes
  ]

  @type t :: %__MODULE__{}
end

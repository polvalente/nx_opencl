defmodule NxCL.CompilerError do
  @moduledoc """
  Raised when a program cannot be compiled or run on the device.
  """

  defexception [:message]
end

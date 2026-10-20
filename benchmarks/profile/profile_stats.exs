# Reads the counter file that shim.c writes.

defmodule ProfStats do
  @moduledoc false
  @slots 512
  @slot_bytes 72

  def read(path) do
    bin = File.read!(path)
    <<h::binary-size(128), rest::binary>> = bin
    [up, upn, rd, rdn, al, aln, bl, bln, en, enn, fi, fin | _] = for <<v::64-little <- h>>, do: v

    kernels =
      for <<name::binary-size(56), count::64-little,
            ns::64-little <- binary_part(rest, 0, @slots * @slot_bytes)>>,
          count > 0,
          into: %{} do
        [n | _] = String.split(name, <<0>>)
        {n, {count, ns}}
      end

    %{
      up: {up, upn},
      read: {rd, rdn},
      alloc: {al, aln},
      build: {bl, bln},
      enq: {en, enn},
      fin: {fi, fin},
      kernels: kernels
    }
  end
end

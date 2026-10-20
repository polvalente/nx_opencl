defmodule NxCL.E2E.LinAlgTest do
  @moduledoc """
  QR and SVD in composition with other Nx ops.
  """
  use ExUnit.Case, async: false
  import NxCL.TestHelpers

  @moduletag :opencl

  describe "QR decomposition" do
    test "Q * R reconstructs original matrix" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

      gpu_a = Nx.backend_transfer(a, NxCL.Backend)
      {q, r} = Nx.LinAlg.qr(gpu_a)

      q_cpu = Nx.backend_transfer(q, Nx.BinaryBackend)
      r_cpu = Nx.backend_transfer(r, Nx.BinaryBackend)

      reconstructed = Nx.dot(q_cpu, r_cpu)
      assert_tensors_close(reconstructed, a, 1.0e-3)
    end

    test "Q is orthogonal (Q^T * Q ≈ I)" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

      gpu_a = Nx.backend_transfer(a, NxCL.Backend)
      {q, _r} = Nx.LinAlg.qr(gpu_a)

      q_cpu = Nx.backend_transfer(q, Nx.BinaryBackend)
      qtq = Nx.dot(Nx.transpose(q_cpu), q_cpu)
      eye = Nx.eye(elem(Nx.shape(qtq), 0))

      assert_tensors_close(qtq, eye, 1.0e-3)
    end

    test "R is upper triangular" do
      a = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])

      gpu_a = Nx.backend_transfer(a, NxCL.Backend)
      {_q, r} = Nx.LinAlg.qr(gpu_a)

      r_cpu = Nx.backend_transfer(r, Nx.BinaryBackend)

      {rows, cols} = Nx.shape(r_cpu)

      for i <- 0..(rows - 1), j <- 0..(cols - 1), i > j do
        val = r_cpu[i][j] |> Nx.to_number()
        assert abs(val) < 1.0e-3, "R[#{i}][#{j}] = #{val}, expected ~0"
      end
    end

    test "QR matches CPU backend" do
      key = Nx.Random.key(42)
      {a, _key} = Nx.Random.uniform(key, shape: {4, 3}, type: :f32)

      {q_cpu, r_cpu} = Nx.LinAlg.qr(a)
      cpu_reconstructed = Nx.dot(q_cpu, r_cpu)

      gpu_a = Nx.backend_transfer(a, NxCL.Backend)
      {q_gpu, r_gpu} = Nx.LinAlg.qr(gpu_a)

      gpu_reconstructed =
        Nx.dot(
          Nx.backend_transfer(q_gpu, Nx.BinaryBackend),
          Nx.backend_transfer(r_gpu, Nx.BinaryBackend)
        )

      assert_tensors_close(cpu_reconstructed, a, 1.0e-3)
      assert_tensors_close(gpu_reconstructed, a, 1.0e-3)
    end

    test "QR on square matrix" do
      a = Nx.tensor([[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]])

      gpu_a = Nx.backend_transfer(a, NxCL.Backend)
      {q, r} = Nx.LinAlg.qr(gpu_a)

      q_cpu = Nx.backend_transfer(q, Nx.BinaryBackend)
      r_cpu = Nx.backend_transfer(r, Nx.BinaryBackend)

      reconstructed = Nx.dot(q_cpu, r_cpu)
      assert_tensors_close(reconstructed, a, 1.0e-2)
    end
  end

  describe "SVD decomposition" do
    test "U * S * V^T reconstructs original matrix" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])

      gpu_a = Nx.backend_transfer(a, NxCL.Backend)
      {u, s, vt} = Nx.LinAlg.svd(gpu_a)

      u_cpu = Nx.backend_transfer(u, Nx.BinaryBackend)
      s_cpu = Nx.backend_transfer(s, Nx.BinaryBackend)
      vt_cpu = Nx.backend_transfer(vt, Nx.BinaryBackend)

      s_diag = Nx.make_diagonal(s_cpu)
      reconstructed = u_cpu |> Nx.dot(s_diag) |> Nx.dot(vt_cpu)

      assert_tensors_close(reconstructed, a, 1.0e-3)
    end

    test "singular values are non-negative and sorted descending" do
      key = Nx.Random.key(42)
      {a, _key} = Nx.Random.uniform(key, shape: {3, 3}, type: :f32)

      gpu_a = Nx.backend_transfer(a, NxCL.Backend)
      {_u, s, _vt} = Nx.LinAlg.svd(gpu_a)

      s_cpu = Nx.backend_transfer(s, Nx.BinaryBackend)
      s_list = Nx.to_list(s_cpu)

      for val <- s_list do
        assert val >= -1.0e-5, "Singular value #{val} is negative"
      end

      assert s_list == Enum.sort(s_list, :desc)
    end

    test "U is orthogonal" do
      a = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

      gpu_a = Nx.backend_transfer(a, NxCL.Backend)
      {u, _s, _vt} = Nx.LinAlg.svd(gpu_a)

      u_cpu = Nx.backend_transfer(u, Nx.BinaryBackend)
      utu = Nx.dot(Nx.transpose(u_cpu), u_cpu)
      n = elem(Nx.shape(utu), 0)
      eye = Nx.eye(n)

      assert_tensors_close(utu, eye, 1.0e-2)
    end

    test "SVD matches CPU backend" do
      key = Nx.Random.key(99)
      {a, _key} = Nx.Random.uniform(key, shape: {3, 3}, type: :f32)

      {u_cpu, s_cpu, vt_cpu} = Nx.LinAlg.svd(a)
      s_diag_cpu = Nx.make_diagonal(s_cpu)
      cpu_reconstructed = u_cpu |> Nx.dot(s_diag_cpu) |> Nx.dot(vt_cpu)

      gpu_a = Nx.backend_transfer(a, NxCL.Backend)
      {u_gpu, s_gpu, vt_gpu} = Nx.LinAlg.svd(gpu_a)

      u_g = Nx.backend_transfer(u_gpu, Nx.BinaryBackend)
      s_g = Nx.backend_transfer(s_gpu, Nx.BinaryBackend)
      vt_g = Nx.backend_transfer(vt_gpu, Nx.BinaryBackend)
      s_diag_g = Nx.make_diagonal(s_g)
      gpu_reconstructed = u_g |> Nx.dot(s_diag_g) |> Nx.dot(vt_g)

      assert_tensors_close(cpu_reconstructed, a, 1.0e-3)
      assert_tensors_close(gpu_reconstructed, a, 1.0e-3)
    end
  end
end

# Whole-program benchmarks: MLP forward and training step, CNN forward, transformer
# block. Columns compare the compiler, the eager backend, the Evaluator on
# BinaryBackend, and EXLA when it is installed.
#
# Usage: mix run benchmarks/compiler_vs_eager.exs

Code.require_file("bench_helper.exs", __DIR__)
Code.require_file("models.exs", __DIR__)

BenchHelper.run_cases("Models", Models.cases())

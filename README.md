# High-Performance Matrix Multiplication in C++ & CUDA ⚡

A comprehensive exploration of General Matrix Multiplication (GEMM) algorithms, progressing from foundational CPU and naive GPU implementations to high-throughput CUDA kernels, cuBLAS benchmarks, and architectural analyses.

---

## 🚀 Key Highlights

- **Progressive CUDA Kernels**: Step-by-step optimization roadmap from uncoalesced naive memory access to advanced shared memory tiling and software pipelining.
- **C++ vs. Python Benchmarks**: Rigorous performance comparisons against PyTorch, cuBLAS, Numba, and NumPy across matrix dimensions up to $4096 \times 4096$.
- **Architectural Deep-Dive**: In-depth Notion notes breaking down CPU vs. GPU memory hierarchies, warp scheduling, bank conflicts, and the Roofline model.
- **Foundations of Backpropagation**: Accompanying matrix-calculus notes and an interactive spreadsheet validating manual backprop derivations.

---

## 🛠️ Kernel Optimization Progression

| Kernel | Technique | Key Mechanism | Bottleneck Solved |
| :--- | :--- | :--- | :--- |
| **Kernel 1** | **Naive GEMM** | 1 thread per output element | Baseline implementation |
| **Kernel 2** | **Coalesced Access** | Consecutive threads access contiguous global memory | Memory transaction serialization |
| **Kernel 3** | **Shared Memory Tiling** | $32 \times 32$ cache blocking via `__shared__` memory | High global memory latency & redundant loads |
| **Kernel 4** | **1D Block / Register Tiling** | Thread-level register accumulation ($BM=128$) | Shared memory bandwidth pressure |
| **Kernel 5** | **Double Buffering** | Multi-stage software pipelining (`cp.async` / ping-pong) | Memory load stalls; overlaps compute and fetch |

---

## 📊 Benchmark Summary

- **cuBLAS (C++) vs. PyTorch (Python)**:
  Both invoke NVIDIA's tuned SGEMM backend with **<1% language gap** at scale ($N \ge 1000$), reaching **~4,300 GFLOPS** (~53–57% of FP32 theoretical peak).
- **C++ CUDA vs. Numba CUDA (Naive)**:
  C++ delivers **2×–4× lower kernel launch and runtime overhead** compared to Python-dispatched kernels.
- **CUDA Kernel 5 vs. NumPy (CPU BLAS)**:
  Massive GPU throughput scaling over multi-threaded AVX2/AVX-512 CPU execution as matrix dimensions grow.

---

## 📁 Repository Structure

```text
├── 4_Kernels_for_matmul_in_cpp_using_cuda.ipynb   # Progressive CUDA Kernels 1-4
├── cuda_cublas_benchmark.ipynb                    # Comprehensive C++ vs Python benchmarks
├── Matrix-multiplication notion notes/            # Detailed architectural notes & diagrams
├── GFLOPS vs Matrix Size.png                      # cuBLAS vs PyTorch throughput plot
├── naive.png                                      # Naive C++ vs Python CUDA scaling plot
├── Delta Backpropagation.xlsx                     # Interactive numerical backpropagation sheet
└── delta_backprop_notation_image.jpeg             # Layer matrix dimension & interaction guide
```

---

## 💻 Quick Start

### Prerequisites
- NVIDIA GPU with Compute Capability $\ge 7.0$
- CUDA Toolkit ($\ge 11.0$) & `nvcc` compiler
- Python 3.10+ (PyTorch, NumPy, Matplotlib, Numba)

### Running the Notebooks
Launch either notebook in Jupyter or Google Colab:
```bash
jupyter notebook 4_Kernels_for_matmul_in_cpp_using_cuda.ipynb
```
or compile individual CUDA kernels directly:
```bash
nvcc -O3 -arch=sm_80 kernel1.cu -o kernel1 && ./kernel1
```

---
*Developed as part of a deep-dive research into hardware-aware machine learning acceleration.*

#include "gpu_gemm.hpp"
#include <iostream>
#include <random>
#include <typeinfo>
#include <vector>

// ===============================
// 生成随机矩阵
// ===============================
template <typename T> void fill_random(std::vector<T> &mat) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  for (auto &x : mat) {
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>) {
      x = T(dist(gen), dist(gen));
    } else {
      x = T(dist(gen));
    }
  }
}

// ===============================
// CPU 参考结果（简单 O(N^3)）
// ===============================
template <typename T>
void cpu_gemm(int N, const std::vector<T> &A, const std::vector<T> &B,
              std::vector<T> &C) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      T sum = T(0);
      for (int k = 0; k < N; k++)
        sum += A[i * N + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
}

// ===============================
// 误差检查
// ===============================
template <typename T>
double max_abs_error(const std::vector<T> &A, const std::vector<T> &B) {
  double err = 0.0;
  for (size_t i = 0; i < A.size(); i++) {
    double diff = std::abs(A[i] - B[i]);
    if (diff > err)
      err = diff;
  }
  return err;
}

// ===============================
// GPU GEMM 调用模板
// ===============================
template <typename T>
void run_gemm(int N, cublasOperation_t transA, cublasOperation_t transB) {
  std::cout << "\n=== Running GEMM for type: " << typeid(T).name()
            << ", N = " << N << " ===\n";

  int lda = N, ldb = N, ldc = N;
  int m = N, n = N, k = N;

  size_t bytes = N * N * sizeof(T);

  // Host matrices
  std::vector<T> hA(N * N), hB(N * N), hC(N * N);
  std::vector<T> hC_ref(N * N);

  fill_random(hA);
  fill_random(hB);

  // Device matrices
  T *dA, *dB, *dC;
  cudaMalloc(&dA, bytes);
  cudaMalloc(&dB, bytes);
  cudaMalloc(&dC, bytes);

  cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice);

  // cublas
  cublasHandle_t handle;
  cublasCreate(&handle);

  T alpha = T(1.0);
  T beta = T(0.0);

  // CUDA timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  gpu_gemm<T>(handle, transA, transB, m, n, k, &alpha, dA, lda, dB, ldb, &beta,
              dC, ldc);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost);

  std::cout << "GPU time: " << ms << " ms\n";

  // cleanup
  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

int main(int argc, char **argv) {
  int N = 1024 * 4;
  if (argc > 1)
    N = std::atoi(argv[1]);

  // 默认不转置
  cublasOperation_t TA = CUBLAS_OP_N;
  cublasOperation_t TB = CUBLAS_OP_N;

  // 你可以在这里切换（例如 T 或 H）
  // TA = CUBLAS_OP_T;
  // TB = CUBLAS_OP_C;  // complex conjugate transpose

  run_gemm<double>(N, TA, TB);

  return 0;
}
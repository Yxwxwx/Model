#pragma once
#include <complex>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <type_traits>

// Base template: not defined
template <typename T> struct CublasGemm;

// ----------------------
// float -> sgemm
// ----------------------
template <> struct CublasGemm<float> {
  static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transA,
                             cublasOperation_t transB, int m, int n, int k,
                             const float *alpha, const float *A, int lda,
                             const float *B, int ldb, const float *beta,
                             float *C, int ldc) {
    return cublasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
  }
};

// ----------------------
// double -> dgemm
// ----------------------
template <> struct CublasGemm<double> {
  static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transA,
                             cublasOperation_t transB, int m, int n, int k,
                             const double *alpha, const double *A, int lda,
                             const double *B, int ldb, const double *beta,
                             double *C, int ldc) {
    return cublasDgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
  }
};

// ----------------------
// std::complex<float> -> cgemm
// ----------------------
template <> struct CublasGemm<std::complex<float>> {
  static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transA,
                             cublasOperation_t transB, int m, int n, int k,
                             const std::complex<float> *alpha,
                             const std::complex<float> *A, int lda,
                             const std::complex<float> *B, int ldb,
                             const std::complex<float> *beta,
                             std::complex<float> *C, int ldc) {
    return cublasCgemm(handle, transA, transB, m, n, k,
                       reinterpret_cast<const cuComplex *>(alpha),
                       reinterpret_cast<const cuComplex *>(A), lda,
                       reinterpret_cast<const cuComplex *>(B), ldb,
                       reinterpret_cast<const cuComplex *>(beta),
                       reinterpret_cast<cuComplex *>(C), ldc);
  }
};

// ----------------------
// std::complex<double> -> zgemm
// ----------------------
template <> struct CublasGemm<std::complex<double>> {
  static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transA,
                             cublasOperation_t transB, int m, int n, int k,
                             const std::complex<double> *alpha,
                             const std::complex<double> *A, int lda,
                             const std::complex<double> *B, int ldb,
                             const std::complex<double> *beta,
                             std::complex<double> *C, int ldc) {
    return cublasZgemm(handle, transA, transB, m, n, k,
                       reinterpret_cast<const cuDoubleComplex *>(alpha),
                       reinterpret_cast<const cuDoubleComplex *>(A), lda,
                       reinterpret_cast<const cuDoubleComplex *>(B), ldb,
                       reinterpret_cast<const cuDoubleComplex *>(beta),
                       reinterpret_cast<cuDoubleComplex *>(C), ldc);
  }
};

// -----------------------------------------------------------
// Generic wrapper: call as gemm<T>(...)
// -----------------------------------------------------------
template <typename T>
inline cublasStatus_t gpu_gemm(cublasHandle_t handle, cublasOperation_t transA,
                               cublasOperation_t transB, int m, int n, int k,
                               const T *alpha, const T *A, int lda, const T *B,
                               int ldb, const T *beta, T *C, int ldc) {
  return CublasGemm<T>::gemm(handle, transA, transB, m, n, k, alpha, A, lda, B,
                             ldb, beta, C, ldc);
}
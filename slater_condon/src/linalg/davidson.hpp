#pragma once
#ifndef DAVIDSON_HPP
#define DAVIDSON_HPP

#include "matrix.hpp"
#include "sparse_matrix.hpp"
#include <algorithm>
#include <chrono>

namespace linalg {

// help functions
template <typename T = double> // returns \sqrt(\sum (v * v))
double norm_(const std::vector<T>& v)
{
    double result = 0.0;
    if constexpr (std::is_same_v<T, double>) {
        result = cblas_dnrm2(v.size(), v.data(), 1);
    }
    else if constexpr (std::is_same_v<T, float>) {
        result = cblas_snrm2(v.size(), v.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        result = cblas_dznrm2(v.size(), v.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
        result = cblas_scnrm2(v.size(), v.data(), 1);
    }
    return result;
}
template <typename T = double> // returns \sum (v1 * v2)
T dot_(const std::vector<T>& v1, const std::vector<T>& v2)
{
    T result = T(0);
    if constexpr (std::is_same_v<T, double>) {
        result = cblas_ddot(v1.size(), v1.data(), 1, v2.data(), 1);
    }
    else if constexpr (std::is_same_v<T, float>) {
        result = cblas_sdot(v1.size(), v1.data(), 1, v2.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        std::complex<double> blas_result;
        cblas_zdotc_sub(v1.size(), v1.data(), 1, v2.data(), 1, &blas_result);
        result = blas_result;
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
        std::complex<float> blas_result;
        cblas_cdotc_sub(v1.size(), v1.data(), 1, v2.data(), 1, &blas_result);
        result = blas_result;
    }
    return result;
}
template <typename T = double> // v1 = v1 * scalar
void dot_(std::vector<T>& v1, T scalar)
{
    auto len = v1.size();
    if constexpr (std::is_same_v<T, double>) {
        cblas_dscal(len, scalar, v1.data(), 1);
    }
    else if constexpr (std::is_same_v<T, float>) {
        cblas_sscal(len, scalar, v1.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        std::complex<double> alpha { scalar.real(), scalar.imag() };
        cblas_zscal(len, &alpha, v1.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
        std::complex<float> alpha { scalar.real(), scalar.imag() };
        cblas_cscal(len, &alpha, v1.data(), 1);
    }
}
template <typename T = double> // v1 = v1 + v2 * scalar
void add_(std::vector<T>& v1, const std::vector<T>& v2, const T scalar)
{
    auto len = v1.size();
    if constexpr (std::is_same_v<T, double>) {
        cblas_daxpy(len, scalar, v2.data(), 1, v1.data(), 1);
    }
    else if constexpr (std::is_same_v<T, float>) {
        cblas_saxpy(len, scalar, v2.data(), 1, v1.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        const std::complex<double> alpha { scalar.real(), scalar.imag() };
        cblas_zaxpy(len, &alpha, v2.data(), 1, v1.data(), 1);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
        const std::complex<float> alpha { scalar.real(), scalar.imag() };
        cblas_caxpy(len, &alpha, v2.data(), 1, v1.data(), 1);
    }
}

template <typename T = double>
Matrix<T> gramschmidt(const Matrix<T>& X)
{
    Matrix<T> orthonormal = X;
    const int cols = X.cols();
    const int rows = X.rows();

    for (int i = 0; i < cols; ++i) {
        std::vector<T> col_i = orthonormal.col(i);

        for (int j = 0; j < i; ++j) {
            const std::vector<T>& col_j = orthonormal.col(j);

            auto proj = dot_(col_i, col_j);
            auto denom = dot_(col_j, col_j);

            T coeff = T(0);
            if (std::abs(denom) > 1e-14) {
                coeff = proj / denom;
            }

            // col_i = col_i - coeff * col_j
            add_(col_i, col_j, -coeff);
        }

        double norm_d = norm_(col_i);

        if (norm_d > 1e-7) {
            const T scale = T(1.0 / norm_d);
            dot_(col_i, scale);
        }
        orthonormal.setCol(i, col_i);
    }
    return orthonormal;
}

// The first freeze_cols column keeps the input values unchanged
// while the subsequent ones have been orthogonized
template <typename T>
Matrix<T> gramschmidt_incremental(const Matrix<T>& X, int freeze_cols)
{
    Matrix<T> Y = X;
    const int cols = Y.cols();
    const int rows = Y.rows();

    // only handle the vectors after the freeze_cols column
    for (int i = freeze_cols; i < cols; ++i) {
        std::vector<T> col_i = Y.col(i); // take out the i-th column

        // project to all columns earlier than it, including the frozen 0... column (i-1)
        for (int j = 0; j < i; ++j) {
            const std::vector<T>& col_j = Y.col(j);

            T denom = dot_(col_j, col_j);
            if (std::abs(denom) < 1e-15)
                continue;

            T coeff = dot_(col_i, col_j) / denom;
            add_(col_i, col_j, -coeff);
        }

        // normalization
        double nrm = norm_(col_i);
        if (nrm > 1e-13) {
            dot_(col_i, T(1.0 / nrm));
        }

        Y.setCol(i, col_i);
    }

    return Y;
}

template <typename Transformer, typename T = double>
const std::vector<double> davidson_solver(Transformer transformer, const T* diagonal,
    std::size_t n_dim, std::size_t n_roots = 1,
    std::size_t start_dim = 5,
    std::size_t max_iter = 100,
    double residue_tol = 1e-6)
{

    if (start_dim < n_roots) {
        throw std::runtime_error(
            "start_dim should be greater than or equal to n_roots");
    }
    // initial guess
    Matrix<T> search_space = Matrix<T>::identity(n_dim, start_dim) + 0.01 * Matrix<T>::ones(n_dim, start_dim);
    Matrix<T> Ab_i = Matrix<T>::zero(n_dim, start_dim);
    // start iteration
    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        // project dim
        auto M = start_dim + iter * n_roots;
        Matrix<T> orthonormal_subspace;
        if (iter == 0) {
            orthonormal_subspace = gramschmidt(search_space);
            for (std::size_t j = 0; j < start_dim; j++) {
                auto vec = orthonormal_subspace.col(j);
                Ab_i.setCol(j, transformer(vec));
            }
        }
        else {
            Ab_i.conservativeResize(n_dim, M);
            orthonormal_subspace = gramschmidt_incremental(search_space, M - n_roots);
            for (std::size_t j = M - n_roots; j < M; j++) {
                auto vec = orthonormal_subspace.col(j);
                Ab_i.setCol(j, transformer(vec));
            }
        }

        Matrix<double> B = orthonormal_subspace.transpose().conjugate() * Ab_i;
        auto [eigenvalues, eigenvectors] = eigh(B);

        std::vector<std::vector<T>> xi_n(n_roots, std::vector<T>(n_dim));
        std::vector<double> theta_n(n_roots);
        std::vector<std::uint8_t> has_converged(n_roots, 0);

        fmt::println("davidson diagonalization iter: {:>2} ", iter + 1);

        // eig_pairs[0..n_roots-1]
        for (int n = 0; n < n_roots; ++n) {
            theta_n[n] = eigenvalues[n]; // the nth smallest eigenvalue

            std::vector<T> s = eigenvectors.col(n);
            std::vector<T> residue_n = Ab_i * s;
            add_(residue_n, orthonormal_subspace * s, -theta_n[n]);
            double residue_norm = norm_(residue_n);

            fmt::println("  root {:>2}: theta = {:10.10f}  |residue| = {:10.10e}", n + 1, theta_n[n], residue_norm);

            if (residue_norm < residue_tol) {
                has_converged[n] = 1;
            }

            // update the search space
            for (int i = 0; i < n_dim; ++i) {
                xi_n[n][i] = residue_n[i] / (theta_n[n] - diagonal[i]);
            }
            auto xi_norm = norm_(xi_n[n]);
            std::transform(xi_n[n].begin(), xi_n[n].end(), xi_n[n].begin(),
                [xi_norm](T& val) { return val / xi_norm; });
        }
        search_space = orthonormal_subspace;
        search_space.conservativeResize(n_dim, M + n_roots);
        for (int n = 0; n < n_roots; ++n) {
            search_space.setCol(M + n, xi_n[n]);
        }
        if (std::all_of(has_converged.begin(), has_converged.end(),
                [](auto c) { return c; })) {
            fmt::println("davidson diagonalization converged in {:>2} iterations",
                iter + 1);
            return theta_n;
        }
    }
    throw std::runtime_error("Davidson diagonalization failed");
}

} // namespace linalg
#endif
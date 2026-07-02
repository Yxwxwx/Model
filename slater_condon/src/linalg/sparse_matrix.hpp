#pragma once
#ifndef SPARSE_MATRIX_HPP
#define SPARSE_MATRIX_HPP

#include <complex>
#include <vector>
#include <mkl.h>

namespace linalg {

enum class MatrixFillMode { UPPER, LOWER };

template <typename T>
class SparseMatrix {
private:
    sparse_matrix_t A_ = nullptr;
    matrix_descr    descr_;
    std::size_t     nrow_;
    std::size_t     ncol_;

public:
    SparseMatrix() = delete;

    SparseMatrix(MatrixFillMode fill_mode, const std::vector<T>& values,
                 const std::vector<MKL_INT>& columns,
                 const std::vector<MKL_INT>& rowIndex, MKL_INT nrows,
                 MKL_INT ncols)
        : nrow_(static_cast<std::size_t>(nrows))
        , ncol_(static_cast<std::size_t>(ncols)) {

        if constexpr (std::is_same_v<T, double>) {
            descr_.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            descr_.type = SPARSE_MATRIX_TYPE_HERMITIAN;
        }

        descr_.mode = (fill_mode == MatrixFillMode::UPPER)
                          ? SPARSE_FILL_MODE_UPPER
                          : SPARSE_FILL_MODE_LOWER;
        descr_.diag = SPARSE_DIAG_NON_UNIT;

        sparse_status_t status;
        if constexpr (std::is_same_v<T, double>) {
            status = mkl_sparse_d_create_csr(
                &A_, SPARSE_INDEX_BASE_ZERO, nrows, ncols,
                const_cast<MKL_INT*>(rowIndex.data()),
                const_cast<MKL_INT*>(rowIndex.data()) + 1,
                const_cast<MKL_INT*>(columns.data()),
                const_cast<double*>(values.data()));
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            status = mkl_sparse_z_create_csr(
                &A_, SPARSE_INDEX_BASE_ZERO, nrows, ncols,
                const_cast<MKL_INT*>(rowIndex.data()),
                const_cast<MKL_INT*>(rowIndex.data()) + 1,
                const_cast<MKL_INT*>(columns.data()),
                const_cast<MKL_Complex16*>(values.data()));
        }

        if (status != SPARSE_STATUS_SUCCESS)
            throw std::runtime_error("Failed to create sparse matrix");
    }

    ~SparseMatrix() { mkl_sparse_destroy(A_); }

    std::vector<T> operator*(const std::vector<T>& x) const {
        std::vector<T> y(nrow_, T{0});

        if constexpr (std::is_same_v<T, double>) {
            const double alpha = 1.0, beta = 0.0;
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                            alpha, A_, descr_, x.data(), beta, y.data());
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            const MKL_Complex16 alpha{1.0, 0.0}, beta{0.0, 0.0};
            mkl_sparse_z_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                            alpha, A_, descr_,
                            reinterpret_cast<const MKL_Complex16*>(x.data()),
                            beta,
                            reinterpret_cast<MKL_Complex16*>(y.data()));
        }

        return y;
    }
};

} // namespace linalg
#endif

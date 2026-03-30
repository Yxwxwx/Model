#pragma once
#ifndef MATRIX_HPP
#define MATRIX_HPP

#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>

// C++ standard library
#include <complex>
#include <random>
// Third-party libraries
#include <fmt/core.h>
#include <mkl.h>
#include <omp.h>

namespace linalg {

template <typename T>
class Matrix {
private:
    T* data_ = nullptr; // 数据指针
    CBLAS_LAYOUT layout_ = CblasColMajor; // 存储布局
    CBLAS_TRANSPOSE trans_ = CblasNoTrans; // 转置状态
    std::size_t ncol_ = 0; // 实际列数
    std::size_t nrow_ = 0; // 实际行数
    std::size_t ld_ = 0; // leading dimension

public:
    // 构造函数
    Matrix()
    {
        nrow_ = 0;
        ncol_ = 0;
        ld_ = 0;
        data_ = nullptr;
    };
    Matrix(std::size_t rows, std::size_t cols,
        CBLAS_LAYOUT layout = CblasColMajor)
        : layout_(layout), ncol_(cols), nrow_(rows)
    {
        // 计算leading dimension
        ld_ = (layout_ == CblasColMajor) ? rows : cols;

        // MKL内存对齐分配
        data_ = static_cast<T*>(mkl_malloc(ld_ * ncol_ * sizeof(T), 64));
        if (!data_)
            throw std::bad_alloc();
    }
    // 析构函数
    ~Matrix()
    {
        if (data_) {
            mkl_free(data_);
            data_ = nullptr;
        }
    }

    // 拷贝构造函数
    Matrix(const Matrix& other)
        : layout_(other.layout_), trans_(other.trans_), ncol_(other.ncol_), nrow_(other.nrow_), ld_(other.ld_)
    {
        data_ = static_cast<T*>(mkl_malloc(ld_ * ncol_ * sizeof(T), 64));
        if (!data_)
            throw std::bad_alloc();
        // 使用 MKL 进行数据拷贝
        if constexpr (std::is_same_v<T, double>) {
            cblas_dcopy(nrow_ * ncol_, other.data_, 1, data_, 1);
        }
        else if constexpr (std::is_same_v<T, float>) {
            cblas_scopy(nrow_ * ncol_, other.data_, 1, data_, 1);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>) {
            cblas_zcopy(nrow_ * ncol_, other.data_, 1, data_, 1);
        }
        else if constexpr (std::is_same_v<T, std::complex<float>>) {
            cblas_ccopy(nrow_ * ncol_, other.data_, 1, data_, 1);
        }
        else {
            std::memcpy(data_, other.data_, ld_ * ncol_ * sizeof(T));
        }
    }

    // 拷贝赋值函数
    Matrix& operator=(const Matrix& other)
    {
        if (this != &other) {
            // 分配新内存
            T* new_data = static_cast<T*>(mkl_malloc(other.ld_ * other.ncol_ * sizeof(T), 64));
            if (!new_data)
                throw std::bad_alloc();

            // 计算要拷贝的元素数量
            size_t num_elements = other.nrow_ * other.ncol_;

            // 根据数据类型选择拷贝方式
            if constexpr (std::is_same_v<T, double>) {
                cblas_dcopy(num_elements, other.data_, 1, new_data, 1);
            }
            else if constexpr (std::is_same_v<T, float>) {
                cblas_scopy(num_elements, other.data_, 1, new_data, 1);
            }
            else if constexpr (std::is_same_v<T, std::complex<double>>) {
                cblas_zcopy(num_elements, reinterpret_cast<const double*>(other.data_),
                    1, reinterpret_cast<double*>(new_data), 1);
            }
            else if constexpr (std::is_same_v<T, std::complex<float>>) {
                cblas_ccopy(num_elements, reinterpret_cast<const float*>(other.data_),
                    1, reinterpret_cast<float*>(new_data), 1);
            }
            else {
                std::memcpy(new_data, other.data_, num_elements * sizeof(T));
            }

            // 释放旧数据
            if (data_)
                mkl_free(data_);

            // 更新成员变量
            data_ = new_data;
            layout_ = other.layout_;
            trans_ = other.trans_;
            ncol_ = other.ncol_;
            nrow_ = other.nrow_;
            ld_ = other.ld_;
        }
        return *this;
    }
    // 移动构造函数
    Matrix(Matrix&& other) noexcept
        : data_(other.data_), layout_(other.layout_), trans_(other.trans_), ncol_(other.ncol_), nrow_(other.nrow_), ld_(other.ld_)
    {
        other.data_ = nullptr;
    }

    // 元素访问
    inline T& operator()(std::size_t i, std::size_t j)
    {
        if (trans_ == CblasTrans)
            std::swap(i, j);
        return layout_ == CblasColMajor ? data_[i + j * ld_] : data_[i * ld_ + j];
    }

    const T& operator()(std::size_t i, std::size_t j) const
    {
        if (trans_ == CblasTrans)
            std::swap(i, j);
        return layout_ == CblasColMajor ? data_[i + j * ld_] : data_[i * ld_ + j];
    }

    // 获取矩阵维度信息
    inline std::size_t rows() const { return nrow_; }
    inline std::vector<T> row(std::size_t i) const
    {
        if (trans_ == CblasNoTrans) {
            auto cols = ncol_;
            std::vector<T> row(cols);
            for (std::size_t j = 0; j < cols; ++j) {
                row[j] = (*this)(i, j);
            }
            return row;
        }
        else {
            throw std::runtime_error("Matrix is transposed");
        }
    }
    inline std::size_t cols() const { return ncol_; }
    inline std::vector<T> col(std::size_t j)
    {
        if (trans_ == CblasNoTrans) {
            auto rows = this->nrow_;
            std::vector<T> col(rows);
            for (std::size_t i = 0; i < rows; ++i) {
                col[i] = (*this)(i, j);
            }
            return col;
        }
        else {
            throw std::runtime_error("Matrix is transposed");
        }
    }
    inline T* data() const { return data_; }
    inline std::size_t ld() const { return ld_; }
    inline void resize(std::size_t rows, std::size_t cols,
        CBLAS_LAYOUT layout = CblasColMajor)
    {
        if (data_) {
            mkl_free(data_);
            data_ = nullptr;
        }
        layout_ = layout;
        ncol_ = cols;
        nrow_ = rows;
        ld_ = (layout_ == CblasColMajor) ? rows : cols;
        data_ = static_cast<T*>(mkl_malloc(ld_ * ncol_ * sizeof(T), 64));
        if (!data_)
            throw std::bad_alloc();
    }
    void conservativeResize(std::size_t new_rows, std::size_t new_cols)
    {
        if (new_rows == nrow_ && new_cols == ncol_)
            return;

        T* old_data = data_;
        std::size_t old_ld = ld_;
        std::size_t old_rows = nrow_;
        std::size_t old_cols = ncol_;
        CBLAS_LAYOUT old_layout = layout_;

        layout_ = old_layout;
        ld_ = (layout_ == CblasColMajor) ? new_rows : new_cols;
        std::size_t new_size = (layout_ == CblasColMajor) ? ld_ * new_cols : ld_ * new_rows;
        data_ = static_cast<T*>(mkl_malloc(new_size * sizeof(T), 64));
        if (!data_)
            throw std::bad_alloc();

        std::size_t copy_rows = std::min(old_rows, new_rows);
        std::size_t copy_cols = std::min(old_cols, new_cols);

        if (layout_ == CblasColMajor) {
            for (std::size_t c = 0; c < copy_cols; ++c) {
                if (copy_rows > 0) {
                    cblas_dcopy(copy_rows, &old_data[c * old_ld], 1, &data_[c * ld_], 1);
                }
            }
        }
        else {
            for (std::size_t r = 0; r < copy_rows; ++r) {
                if (copy_cols > 0) {
                    cblas_dcopy(copy_cols, &old_data[r * old_ld], 1, &data_[r * ld_], 1);
                }
            }
        }

        mkl_free(old_data);
        nrow_ = new_rows;
        ncol_ = new_cols;
    }
    inline void setZero()
    {
        if (!data_) {
            throw std::runtime_error("Matrix data is not allocated");
        }
        std::size_t total = nrow_ * ncol_;
        if constexpr (std::is_same_v<T, float>) {
            cblas_sscal(total, 0.0f, data_, 1);
        }
        else if constexpr (std::is_same_v<T, double>) {
            cblas_dscal(total, 0.0, data_, 1);
        }
        else if constexpr (std::is_same_v<T, std::complex<float>>) {
            MKL_Complex8 alpha { 0.0f, 0.0f };
            cblas_cscal(total, &alpha, data_, 1);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>) {
            MKL_Complex16 alpha { 0.0, 0.0 };
            cblas_zscal(total, &alpha, data_, 1);
        }
        else {
#pragma omp parallel for
            for (std::size_t i = 0; i < total; ++i) {
                data_[i] = T { 0 };
            }
        }
    }

    inline void setCol(std::size_t j, const std::vector<T>& col)
    {
        if (trans_ == CblasNoTrans) {
            if (col.size() != nrow_) {
                throw std::runtime_error("Column size does not match matrix rows");
            }
            for (std::size_t i = 0; i < nrow_; ++i) {
                (*this)(i, j) = col[i];
            }
        }
        else {
            throw std::runtime_error("Matrix is transposed");
        }
    }

    // 获取转置矩阵
    inline Matrix<T> transpose()
    {
        Matrix view(*this);
        view.trans_ = (trans_ == CblasNoTrans) ? CblasTrans : CblasNoTrans;
        std::swap(view.nrow_, view.ncol_);
        return view;
    }

    // 原地转置
    inline void transpose_inplace()
    {
        this->trans_ = (trans_ == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    }

    // 复共轭
    inline Matrix<T> conjugate() const
    {
        Matrix view(*this);
        if constexpr (std::is_same_v<T, std::complex<float>>) {
            std::size_t total = view.nrow_ * view.ncol_;
            vcConj(total, this->data_, view.data_);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>) {
            std::size_t total = view.nrow_ * view.ncol_;
            vzConj(total, this->data_, view.data_);
        }
        return view;
    }

    // 转置共轭
    inline Matrix<T> adjoint() const
    {
        Matrix view = this->conjugate();
        view.transpose_inplace();
        return view;
    }

    // 生成全零矩阵
    inline static Matrix zero(std::size_t rows, std::size_t cols,
        CBLAS_LAYOUT layout = CblasColMajor)
    {
        Matrix mat(rows, cols, layout);
        std::size_t total = mat.nrow_ * mat.ncol_;
        if constexpr (std::is_same_v<T, float>) {
            cblas_sscal(total, 0.0f, mat.data_, 1);
        }
        else if constexpr (std::is_same_v<T, double>) {
            cblas_dscal(total, 0.0, mat.data_, 1);
        }
        else if constexpr (std::is_same_v<T, std::complex<float>>) {
            MKL_Complex8 alpha { 0.0f, 0.0f };
            cblas_cscal(total, &alpha, mat.data_, 1);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>) {
            MKL_Complex16 alpha { 0.0, 0.0 };
            cblas_zscal(total, &alpha, mat.data_, 1);
        }
        else {
#pragma omp parallel for
            for (std::size_t i = 0; i < total; ++i) {
                mat.data_[i] = T { 0 };
            }
        }
        return mat;
    }

    // 生成单位矩阵
    inline static Matrix identity(std::size_t n,
        CBLAS_LAYOUT layout = CblasColMajor)
    {
        Matrix mat = zero(n, n, layout);
#pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i) {
            mat(i, i) = T { 1 };
        }
        return mat;
    }
    // 生成非方阵的单位矩阵
    inline static Matrix identity(std::size_t rows, std::size_t cols,
        CBLAS_LAYOUT layout = CblasColMajor)
    {
        Matrix mat = zero(rows, cols, layout);
        std::size_t n = std::min(rows, cols);
#pragma omp parallel for
        for (std::size_t i = 0; i < n; ++i) {
            mat(i, i) = T { 1 };
        }
        return mat;
    }
    // 生成全1矩阵
    inline static Matrix ones(std::size_t rows, std::size_t cols,
        CBLAS_LAYOUT layout = CblasColMajor)
    {
        Matrix mat(rows, cols, layout);
        std::size_t total = mat.nrow_ * mat.ncol_;
#pragma omp parallel for
        for (std::size_t i = 0; i < total; ++i) {
            mat.data_[i] = T { 1 };
        }
        return mat;
    }

    // 生成随机矩阵（均匀分布）
    inline static Matrix random(std::size_t rows, std::size_t cols,
        CBLAS_LAYOUT layout = CblasColMajor,
        T min = T { -1 }, T max = T { 1 })
    {
        Matrix mat(rows, cols, layout);
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MT19937, std::random_device {}());

        const std::size_t total = mat.nrow_ * mat.ncol_;

        // 实数类型处理
        if constexpr (std::is_same_v<T, float>) {
            vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, total, mat.data_, min,
                max);
        }
        else if constexpr (std::is_same_v<T, double>) {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, total, mat.data_, min,
                max);
        }
        else if constexpr (std::is_same_v<T, std::complex<float>>) {
            vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 2 * total,
                reinterpret_cast<float*>(mat.data_), std::real(min),
                std::real(max));
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>) {
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 2 * total,
                reinterpret_cast<double*>(mat.data_), std::real(min),
                std::real(max));
        }
        else {
            for (std::size_t i = 0; i < total; ++i) {
                mat.data_[i] = T { std::uniform_real_distribution<T>(min, max)(
                    std::random_device {}()) };
            }
        }

        vslDeleteStream(&stream);
        return mat;
    }

    inline void print() const
    {
        // 格式化输出实现
        for (std::size_t i = 0; i < rows(); ++i) {
            fmt::print("["); // 行起始符
            for (std::size_t j = 0; j < cols(); ++j) {
                // 类型分发格式化
                if constexpr (std::is_same_v<T, float>) {
                    fmt::print("{:>8.4f}", (*this)(i, j));
                }
                else if constexpr (std::is_same_v<T, double>) {
                    fmt::print("{:>10.4f}", (*this)(i, j));
                }
                else if constexpr (std::is_same_v<T, std::complex<float>>) {
                    fmt::print("({:>6.2f}, {:>6.2f})", (*this)(i, j).real(),
                        (*this)(i, j).imag());
                }
                else if constexpr (std::is_same_v<T, std::complex<double>>) {
                    fmt::print("({:>8.4f}, {:>8.4f})", (*this)(i, j).real(),
                        (*this)(i, j).imag());
                }
                // 列分隔符
                if (j != cols() - 1)
                    fmt::print(", ");
            }
            fmt::print("]\n"); // 行终止符
        }
    }

    // 矩阵加法
    inline Matrix<T> operator+(const Matrix<T>& other) const
    {
        if (this->rows() != other.rows() || this->cols() != other.cols()) {
            throw std::invalid_argument(
                "Matrix dimensions must match for subtraction");
        }
        if (this->layout_ != other.layout_) {
            throw std::invalid_argument("Matrix layouts must match for subtraction");
        }
        Matrix<T> C(this->rows(), this->cols(), this->layout_);
        std::size_t total = this->nrow_ * this->ncol_;
        if (this->trans_ == other.trans_) {
            if constexpr (std::is_same_v<T, double>) {
                vdAdd(total, this->data_, other.data_, C.data_);
            }
            else if constexpr (std::is_same_v<T, float>) {
                vsAdd(total, this->data_, other.data_, C.data_);
            }
            else if constexpr (std::is_same_v<T, std::complex<float>>) {
                vcAdd(total, this->data_, other.data_, C.data_);
            }
            else if constexpr (std::is_same_v<T, std::complex<double>>) {
                vzAdd(total, this->data_, other.data_, C.data_);
            }
            else {
#pragma omp parallel for
                for (std::size_t i = 0; i < total; ++i) {
                    C.data_[i] = this->data_[i] + other.data_[i];
                }
            }
        }
        else {
#pragma omp parallel for
            for (std::size_t j = 0; j < this->ncol_; ++j) {
                for (std::size_t i = 0; i < this->nrow_; ++i) {
                    C(i, j) = (*this)(i, j) + other(i, j);
                }
            }
        }
        return C;
    }

    // 矩阵减法
    inline Matrix<T> operator-(const Matrix<T>& other) const
    {
        if (this->rows() != other.rows() || this->cols() != other.cols()) {
            throw std::invalid_argument(
                "Matrix dimensions must match for subtraction");
        }
        if (this->layout_ != other.layout_) {
            throw std::invalid_argument("Matrix layouts must match for subtraction");
        }
        Matrix<T> C(this->rows(), this->cols(), this->layout_);
        std::size_t total = this->nrow_ * this->ncol_;
        if (this->trans_ == other.trans_) {
            if constexpr (std::is_same_v<T, double>) {
                vdSub(total, this->data_, other.data_, C.data_);
            }
            else if constexpr (std::is_same_v<T, float>) {
                vsSub(total, this->data_, other.data_, C.data_);
            }
            else if constexpr (std::is_same_v<T, std::complex<float>>) {
                vcSub(total, this->data_, other.data_, C.data_);
            }
            else if constexpr (std::is_same_v<T, std::complex<double>>) {
                vzSub(total, this->data_, other.data_, C.data_);
            }
            else {
#pragma omp parallel for
                for (std::size_t i = 0; i < total; ++i) {
                    C.data_[i] = this->data_[i] - other.data_[i];
                }
            }
        }
        else {
#pragma omp parallel for
            for (std::size_t j = 0; j < this->ncol_; ++j) {
                for (std::size_t i = 0; i < this->nrow_; ++i) {
                    C(i, j) = (*this)(i, j) - other(i, j);
                }
            }
        }
        return C;
    }

    // 矩阵数乘
    friend Matrix<T> operator*(T coeff, const Matrix& M)
    {
        Matrix tmp(M);
        if constexpr (std::is_same_v<T, double>) {
            cblas_dscal(tmp.ncol_ * tmp.nrow_, coeff, tmp.data_, 1);
        }
        else if constexpr (std::is_same_v<T, float>) {
            cblas_sscal(tmp.ncol_ * tmp.nrow_, coeff, tmp.data_, 1);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>) {
            cblas_zscal(tmp.ncol_ * tmp.nrow_, coeff, tmp.data_, 1);
        }
        else if constexpr (std::is_same_v<T, std::complex<float>>) {
            cblas_cscal(tmp.ncol_ * tmp.nrow_, coeff, tmp.data_, 1);
        }

        return tmp;
    }

    // 矩阵*向量
    friend std::vector<T> operator*(const Matrix<T>& A, const std::vector<T>& v)
    {
        if (A.cols() != v.size()) {
            throw std::invalid_argument(
                "Matrix dimensions must match for multiplication");
        }
        if (A.layout_ != CblasColMajor) {
            throw std::invalid_argument(
                "Matrix layout must be CblasColMajor for multiplication");
        }
        std::vector<T> result(A.rows());
        if constexpr (std::is_same_v<T, double>) {
            cblas_dgemv(A.layout_, A.trans_, A.rows(), A.cols(), 1.0, A.data_, A.ld_,
                v.data(), 1, 0.0, result.data(), 1);
        }
        else if constexpr (std::is_same_v<T, float>) {
            cblas_sgemv(A.layout_, A.trans_, A.rows(), A.cols(), 1.0, A.data_, A.ld_,
                v.data(), 1, 0.0, result.data(), 1);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>) {
            MKL_Complex16 alpha { 1.0f, 0.0f };
            MKL_Complex16 beta { 0.0f, 0.0f };
            cblas_zgemv(A.layout_, A.trans_, A.rows(), A.cols(), &alpha, A.data_,
                A.ld_, v.data(), 1, &beta, result.data(), 1);
        }
        else if constexpr (std::is_same_v<T, std::complex<float>>) {
            MKL_Complex8 alpha { 1.0f, 0.0f };
            MKL_Complex8 beta { 0.0f, 0.0f };
            cblas_cgemv(A.layout_, A.trans_, A.rows(), A.cols(), &alpha, A.data_,
                A.ld_, v.data(), 1, &beta, result.data(), 1);
        }
        return result;
    }
    // 矩阵乘法
    friend Matrix<T> operator*(const Matrix<T>& A, const Matrix<T>& B)
    {
        if (A.cols() != B.rows()) {
            throw std::invalid_argument(
                "Matrix dimensions must match for multiplication");
        }
        if (A.layout_ != B.layout_) {
            throw std::invalid_argument(
                "Matrix layouts must match for multiplication");
        }
        Matrix<T> C(A.rows(), B.cols(), A.layout_);

        if constexpr (std::is_same_v<T, double>) {
            cblas_dgemm(A.layout_, A.trans_, B.trans_, A.rows(), B.cols(), A.cols(),
                1.0, A.data_, A.ld_, B.data_, B.ld_, 0.0, C.data_, C.ld_);
        }
        else if constexpr (std::is_same_v<T, float>) {
            cblas_sgemm(A.layout_, A.trans_, B.trans_, A.rows(), B.cols(), A.cols(),
                1.0, A.data_, A.ld_, B.data_, B.ld_, 0.0, C.data_, C.ld_);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>) {
            MKL_Complex16 alpha { 1.0f, 0.0f };
            MKL_Complex16 beta { 0.0f, 0.0f };
            cblas_zgemm(A.layout_, A.trans_, B.trans_, A.rows(), B.cols(), A.cols(),
                &alpha, A.data_, A.ld_, B.data_, B.ld_, &beta, C.data_,
                C.ld_);
        }
        else if constexpr (std::is_same_v<T, std::complex<float>>) {
            MKL_Complex8 alpha { 1.0f, 0.0f };
            MKL_Complex8 beta { 0.0f, 0.0f };
            cblas_cgemm(A.layout_, A.trans_, B.trans_, A.rows(), B.cols(), A.cols(),
                &alpha, A.data_, A.ld_, B.data_, B.ld_, &beta, C.data_,
                C.ld_);
        }
        else {
            for (std::size_t j = 0; j < C.cols(); ++j) {
                for (std::size_t i = 0; i < C.rows(); ++i) {
                    C(i, j) = 0;
                    for (std::size_t k = 0; k < A.cols(); ++k) {
                        C(i, j) += A(i, k) * B(k, j);
                    }
                }
            }
        }
        return C;
    }
};

// 类型萃取模板
template <typename T>
struct MatrixTypeInfo {
    using RealType = T;
};
template <>
struct MatrixTypeInfo<std::complex<float>> {
    using RealType = float;
};
template <>
struct MatrixTypeInfo<std::complex<double>> {
    using RealType = double;
};

// 接口声明
template <typename T>
std::pair<std::vector<typename MatrixTypeInfo<T>::RealType>, Matrix<T>>
eigh(const Matrix<T>& H, char uplo = 'L');

template <typename T>
std::pair<std::vector<typename MatrixTypeInfo<T>::RealType>, Matrix<T>>
eigh(const Matrix<T>& A, const Matrix<T>& B, char uplo = 'L');

// 实现部分
namespace internal {

    // 特征值计算分发器 (普通版)
    template <typename T, typename Real = T>
    void lapack_eigh(int n, T* a, int lda, Real* w, int& info);
    // 特征值计算分发器 (广义版)
    template <typename T, typename Real = T>
    void lapack_eigh(int itype, char jobz, char uplo, int n, T* a, int lda, T* b,
        int ldb, Real* w, int& info);

    // 特化版本实现
    //-----------------------------------------------------------------------------
    // double 类型
    template <>
    inline void lapack_eigh<double>(int n, double* a, int lda, double* w,
        int& info)
    {
        char jobz = 'V', uplo = 'L'; // 默认下三角
        int lwork = -1;
        double work_query;
        int liwork = -1, iwork_query;

        // 第一次调用查询工作空间
        dsyevd_(&jobz, &uplo, &n, a, &lda, w, &work_query, &lwork, &iwork_query,
            &liwork, &info);

        // 分配工作空间
        lwork = static_cast<int>(work_query);
        std::vector<double> work(lwork);
        liwork = iwork_query;
        std::vector<int> iwork(liwork);

        // 实际计算
        dsyevd_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, iwork.data(),
            &liwork, &info);
    }

    template <>
    inline void lapack_eigh<double>(int itype, char jobz, char uplo, int n,
        double* a, int lda, double* b, int ldb,
        double* w, int& info)
    {
        int lwork = -1;
        double work_query;
        int liwork = -1, iwork_query;

        // 查询工作空间
        dsygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &work_query, &lwork,
            &iwork_query, &liwork, &info);

        lwork = static_cast<int>(work_query);
        std::vector<double> work(lwork);
        liwork = iwork_query;
        std::vector<int> iwork(liwork);

        dsygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork,
            iwork.data(), &liwork, &info);
    }

    //-----------------------------------------------------------------------------
    // float 类型
    template <>
    inline void lapack_eigh<float>(int n, float* a, int lda, float* w, int& info)
    {
        char jobz = 'V', uplo = 'L';
        int lwork = -1;
        float work_query;
        int liwork = -1, iwork_query;

        ssyevd_(&jobz, &uplo, &n, a, &lda, w, &work_query, &lwork, &iwork_query,
            &liwork, &info);

        lwork = static_cast<int>(work_query);
        std::vector<float> work(lwork);
        liwork = iwork_query;
        std::vector<int> iwork(liwork);

        ssyevd_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, iwork.data(),
            &liwork, &info);
    }

    template <>
    inline void lapack_eigh<float>(int itype, char jobz, char uplo, int n, float* a,
        int lda, float* b, int ldb, float* w,
        int& info)
    {
        int lwork = -1;
        float work_query;
        int liwork = -1, iwork_query;

        ssygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &work_query, &lwork,
            &iwork_query, &liwork, &info);

        lwork = static_cast<int>(work_query);
        std::vector<float> work(lwork);
        liwork = iwork_query;
        std::vector<int> iwork(liwork);

        ssygvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork,
            iwork.data(), &liwork, &info);
    }

    //-----------------------------------------------------------------------------
    // std::complex<double> 类型 (普通特征值)
    template <>
    inline void
    lapack_eigh<std::complex<double>, double>(int n, std::complex<double>* a,
        int lda, double* w, int& info)
    {
        char jobz = 'V', uplo = 'L';
        int lwork = -1;
        std::complex<double> work_query;
        int lrwork = -1;
        double rwork_query;
        int liwork = -1, iwork_query;

        // 第一次调用查询工作空间
        zheevd_(&jobz, &uplo, &n, a, &lda, w, &work_query, &lwork, &rwork_query,
            &lrwork, &iwork_query, &liwork, &info);

        // 分配工作空间
        lwork = static_cast<int>(work_query.real());
        std::vector<std::complex<double>> work(lwork);
        lrwork = static_cast<int>(rwork_query);
        std::vector<double> rwork(lrwork);
        liwork = iwork_query;
        std::vector<int> iwork(liwork);

        // 实际计算
        zheevd_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, rwork.data(),
            &lrwork, iwork.data(), &liwork, &info);
    }
    // std::complex<double> 类型 (广义特征值)
    template <>
    inline void lapack_eigh<std::complex<double>, double>(
        int itype, char jobz, char uplo, int n, std::complex<double>* a, int lda,
        std::complex<double>* b, int ldb, double* w, int& info)
    {
        int lwork = -1;
        std::complex<double> work_query;
        int lrwork = -1;
        double rwork_query;
        int liwork = -1, iwork_query;

        // 查询工作空间
        zhegvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &work_query, &lwork,
            &rwork_query, &lrwork, &iwork_query, &liwork, &info);

        // 分配工作空间
        lwork = static_cast<int>(work_query.real());
        std::vector<std::complex<double>> work(lwork);
        lrwork = static_cast<int>(rwork_query);
        std::vector<double> rwork(lrwork);
        liwork = iwork_query;
        std::vector<int> iwork(liwork);

        // 实际计算
        zhegvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork,
            rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    }
    //-----------------------------------------------------------------------------
    // std::complex<float> 类型
    template <>
    inline void
    lapack_eigh<std::complex<float>, float>(int n, std::complex<float>* a, int lda,
        float* w, int& info)
    {
        char jobz = 'V', uplo = 'L';
        int lwork = -1;
        std::complex<float> work_query;
        int lrwork = -1;
        float rwork_query;
        int liwork = -1, iwork_query;

        cheevd_(&jobz, &uplo, &n, a, &lda, w, &work_query, &lwork, &rwork_query,
            &lrwork, &iwork_query, &liwork, &info);

        lwork = static_cast<int>(work_query.real());
        std::vector<std::complex<float>> work(lwork);
        lrwork = static_cast<int>(rwork_query);
        std::vector<float> rwork(lrwork);
        liwork = iwork_query;
        std::vector<int> iwork(liwork);

        cheevd_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, rwork.data(),
            &lrwork, iwork.data(), &liwork, &info);
    }
    // std::complex<float> 类型 (广义特征值)
    template <>
    inline void lapack_eigh<std::complex<float>, float>(
        int itype, char jobz, char uplo, int n, std::complex<float>* a, int lda,
        std::complex<float>* b, int ldb, float* w, int& info)
    {
        int lwork = -1;
        std::complex<float> work_query;
        int lrwork = -1;
        float rwork_query;
        int liwork = -1, iwork_query;

        // 查询工作空间
        chegvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &work_query, &lwork,
            &rwork_query, &lrwork, &iwork_query, &liwork, &info);

        // 分配工作空间
        lwork = static_cast<int>(work_query.real());
        std::vector<std::complex<float>> work(lwork);
        lrwork = static_cast<int>(rwork_query);
        std::vector<float> rwork(lrwork);
        liwork = iwork_query;
        std::vector<int> iwork(liwork);

        // 实际计算
        chegvd_(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work.data(), &lwork,
            rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    }

} // namespace internal

template <typename T>
std::pair<std::vector<typename MatrixTypeInfo<T>::RealType>, Matrix<T>>
eigh(const Matrix<T>& H, char uplo)
{
    using Real = typename MatrixTypeInfo<T>::RealType;

    Matrix<T> A(H);
    const int n = static_cast<int>(H.rows());
    std::vector<Real> eigenvalues(n); // 使用萃取后的类型

    int info = 0;
    internal::lapack_eigh<T, Real>( // 显式指定模板参数
        n, A.data(), A.ld(), eigenvalues.data(), info);

    return { eigenvalues, A };
}

// 广义版本同理修改
template <typename T>
std::pair<std::vector<typename MatrixTypeInfo<T>::RealType>, Matrix<T>>
eigh(const Matrix<T>& A, const Matrix<T>& B, char uplo)
{
    using Real = typename MatrixTypeInfo<T>::RealType;
    Matrix<T> A_copy(A), B_copy(B);
    std::vector<Real> eigenvalues(A.rows());

    int info = 0;
    internal::lapack_eigh<T, Real>(1, // itype=1对应A*x = λ*B*x
        'V', // 计算特征向量
        uplo, static_cast<int>(A.rows()),
        A_copy.data(), A_copy.ld(), B_copy.data(),
        B_copy.ld(), eigenvalues.data(), info);

    if (info != 0) {
        throw std::runtime_error("LAPACK eigh failed with code " + std::to_string(info));
    }
    return { eigenvalues, A_copy };
}

} // namespace linalg

#endif // MATRIX_HPP
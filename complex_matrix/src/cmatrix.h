// cmatrix.h
#ifndef CMATRIX_H
#define CMATRIX_H

#ifdef __USE_MKL__
#include <mkl.h>
#else
#include <cblas.h>
#include <lapack.h>
#endif

#include <algorithm>
#include <cassert>
#include <complex>
#include <format>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace cmatrix {

class CMatrix {
private:
    int rows, cols;
    std::vector<std::complex<double>> data;
    bool isRowMajor;
    CBLAS_TRANSPOSE trans_; // MKL

public:
    CMatrix(int rows, int cols, bool rowMajor = false);
    void print();
    // 元素访问
    std::complex<double>& operator()(int i, int j);
    const std::complex<double>& operator()(int i, int j) const;

    static CMatrix Random(int rows, int cols, bool rowMajor = false);
    void Random();

    // 获取行数和列数
    int getRows() const;
    int getCols() const;
    bool isRowMajorOrder() const;

    // 矩阵乘法
    static CMatrix multiply(const CMatrix& A, const CMatrix& B);

    std::complex<double>& operator()(size_t i, size_t j);
    const std::complex<double>& operator()(size_t i, size_t j) const;

    // 运算符重载
    CMatrix operator*(const CMatrix& other) const;
    CMatrix operator-(const CMatrix& other) const;

    // 转置操作
    CMatrix transpose() const;
    CMatrix& transpose_inplace(); // 原地转置
    CMatrix T() const { return transpose(); } // T()作为transpose()的简写

    // 共轭操作
    CMatrix conj() const;
    CMatrix& conj_inplace(); // 原地共轭转换

    // 设置矩阵块
    void setBlock(int rowOffset, int colOffset, const CMatrix& block);
    // 获取矩阵块
    CMatrix getBlock(int rowOffset, int colOffset, int blockRows, int blockCols) const;

    // 获取原始数据指针（用于LAPACK）
    std::complex<double>* data_ptr() { return data.data(); }
    const std::complex<double>* data_ptr() const { return data.data(); }

    // 标准特征值问题求解
    static std::pair<std::vector<double>, CMatrix> eigh(const CMatrix& matrix);

    // 广义特征值问题求解
    static std::pair<std::vector<double>, CMatrix> eigh(const CMatrix& matrix, const CMatrix& overlap);

    // 辅助函数：检查矩阵是否为厄米矩阵
    bool isHermitian() const;
    CMatrix convertToLayout(bool targetRowMajor) const;
};
}

#endif // CMATRIX_H
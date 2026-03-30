#pragma once
#ifndef INTEGRAL_HPP
#define INTEGRAL_HPP

#include "linalg/matrix.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace integral {

// help functions

struct ScalerTag { };
struct SpinorTag { };

using Scaler = ScalerTag;
using Spinor = SpinorTag;

// Class to handle quantum chemistry integrals (one and two electron integrals)
// Only ColMajor layout is supported
template <typename T = double>
class Int2e {
private:
    T* data_;
    std::size_t norb_;

public:
    Int2e()
        : data_(nullptr), norb_(0) { }
    ~Int2e()
    {
        if (data_) {
            mkl_free(data_);
            data_ = nullptr;
        }
    }
    inline void resize(std::size_t norb)
    {
        norb_ = norb;
        data_ = static_cast<T*>(
            mkl_malloc(norb_ * norb_ * norb_ * norb_ * sizeof(T), 64));
        if (!data_) {
            throw std::bad_alloc(); // 抛出异常
        }
    }
    inline void setZero()
    {
        if (!data_) {
            throw std::runtime_error("Int2e is not initialized");
        }
        std::size_t total = norb_ * norb_ * norb_ * norb_;
        if constexpr (std::is_same_v<T, double>) {
            cblas_dscal(total, 0.0, data_, 1);
        }
        else if constexpr (std::is_same_v<T, std::complex<double>>) {
            MKL_Complex16 alpha { 0.0, 0.0 };
            cblas_zscal(total, &alpha, data_, 1);
        }
    }

    inline T& operator()(std::size_t p, std::size_t q, std::size_t r,
        std::size_t s)
    {
        return data_[s * norb_ * norb_ * norb_ + r * norb_ * norb_ + q * norb_ + p];
    }

    inline const T& operator()(std::size_t p, std::size_t q, std::size_t r,
        std::size_t s) const
    {
        return data_[s * norb_ * norb_ * norb_ + r * norb_ * norb_ + q * norb_ + p];
    }
};

template <typename Representation = ScalerTag>
class Integral {

    static_assert(std::is_same_v<Representation, ScalerTag> || std::is_same_v<Representation, SpinorTag>,
        "Template argument must be ScalerTag or SpinorTag");

    static constexpr bool is_scaler = std::is_same_v<Representation, ScalerTag>;
    static constexpr bool is_spinor = !is_scaler;

public:
    using Type = std::conditional_t<is_scaler, double, std::complex<double>>;
    // Constructor that reads integral data from a file
    Integral(const std::string& filename)
    {
        norb_ = 0;
        coreE_ = 0;

        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("could not open: " + filename); // 抛出异常
        }

        try {
            std::vector<std::string> lines;
            std::string line;

            // Read all lines from file
            while (std::getline(file, line)) {
                lines.emplace_back(line);
            }

            file.close();

            // Parse the number of orbitals from first line
            std::istringstream iss(lines[0]);
            std::string part;
            std::vector<std::string> parts;
            while (iss >> part) {
                parts.emplace_back(part);
            }
            norb_ = std::stoi(parts[2].substr(0, parts[2].length() - 1));
            // nelec_ = std::stoi(parts[4].substr(0, parts[4].length() - 1));

            // Initialize one-electron integral matrix
            int1e_.resize(norb_, norb_);
            int1e_.setZero();

            // Initialize two-electron integral tensor
            int2e_.resize(norb_);
            int2e_.setZero();

            // Parse integral values from file
            for (size_t i = 4; i < lines.size(); ++i) {
                std::istringstream iss2(lines[i]);
                std::vector<std::string> parts2;
                while (iss2 >> part) {
                    parts2.emplace_back(part);
                }

                double val = std::stod(parts2[0]);
                std::size_t p = std::stoi(parts2[1]) - 1;
                std::size_t q = std::stoi(parts2[2]) - 1;
                std::size_t r = std::stoi(parts2[3]) - 1;
                std::size_t s = std::stoi(parts2[4]) - 1;

                // Handle different types of integrals
                if (r == -1 && s == -1) {
                    if (p == -1 && q == -1) {
                        coreE_ = val; // Nuclear repulsion energy
                    }
                    else {
                        // One-electron integrals
                        int1e_(p, q) = val;
                        int1e_(q, p) = val;
                    }
                }
                else {
                    // Two-electron integrals with symmetry
                    int2e_(p, q, r, s) = val;
                    int2e_(p, q, s, r) = val;
                    int2e_(q, p, r, s) = val;
                    int2e_(q, p, s, r) = val;
                    int2e_(r, s, p, q) = val;
                    int2e_(r, s, q, p) = val;
                    int2e_(s, r, p, q) = val;
                    int2e_(s, r, q, p) = val;
                }
            }
        }
        catch (...) {
            // 确保资源正确释放
            if (file.is_open()) {
                file.close();
            }
            throw; // 重新抛出异常
        }
    }

    // Get one-electron integral value
    const Type h1e(std::size_t p, std::size_t q) const { return int1e_(p, q); }

    // Get two-electron integral value
    const Type h2e(std::size_t p, std::size_t q, std::size_t r,
        std::size_t s) const
    {
        return int2e_(p, q, r, s);
    }

    // Get nuclear repulsion energy
    const double CoreE() { return coreE_; }
    // Get number of orbitals
    const std::size_t norb() { return norb_; }
    const std::size_t nelec() { return nelec_; }
    const linalg::Matrix<Type>& int1e() { return int1e_; }
    const Int2e<Type>& int2e() { return int2e_; }

private:
    std::size_t norb_ { 0 }; // Number of orbitals
    std::size_t nelec_ { 0 }; // Number of electrons
    double coreE_ { 0 }; // Nuclear repulsion energy
    linalg::Matrix<Type> int1e_; // One-electron integrals
    Int2e<Type> int2e_; // Two-electron integrals
};

} // namespace integral

#endif // INTEGRAL_HPP
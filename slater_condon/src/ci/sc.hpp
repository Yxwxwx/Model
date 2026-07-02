#pragma once
#ifndef SC_HPP
#define SC_HPP

#include "integral/integral.hpp"
#include "linalg/davidson.hpp"
#include "linalg/sparse_matrix.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <vector>

namespace ci {

/// Tag type to request explicit (nalpha, nbeta) construction.
struct AlphaBetaTag {};

class Det {
private:
    uint64_t alpha_; // bit i = 1 → orbital i has alpha electron
    uint64_t beta_;  // bit i = 1 → orbital i has beta electron
    size_t norb_;

public:
    Det(uint64_t alpha, uint64_t beta, size_t norb)
        : alpha_(alpha), beta_(beta), norb_(norb) {}

    size_t nalpha() const { return std::popcount(alpha_); }
    size_t nbeta() const { return std::popcount(beta_); }
    size_t nelec() const { return nalpha() + nbeta(); }
    size_t norb() const { return norb_; }
    uint64_t alpha_bits() const { return alpha_; }
    uint64_t beta_bits() const { return beta_; }

    // --- helpers ---

    static double sign(size_t n) { return (n & 1) ? -1.0 : 1.0; }

    /// Count set bits in [p+1, q-1] (or [q+1, p-1] if q < p).
    static size_t count_between(uint64_t bits, size_t p, size_t q) {
        if (p > q) std::swap(p, q);
        if (q <= p + 1) return 0;
        uint64_t mask = ((1ULL << (q - p - 1)) - 1) << (p + 1);
        return std::popcount(bits & mask);
    }

    /// Iterate over every set bit in `bits`, calling fn(index).
    template <typename F>
    static void for_each_bit(uint64_t bits, F&& fn) {
        while (bits) {
            size_t p = std::countr_zero(bits);
            bits &= bits - 1;
            fn(p);
        }
    }

    // --- Slater-Condon rules ---

    /// Diagonal matrix element ⟨I|H|I⟩.
    double Hii(const integral::Integral<integral::ScalerTag>& integral) const {
        double val = 0.0;

        // alpha electrons
        for_each_bit(alpha_, [&](size_t p) {
            val += integral.h1e(p, p);
            // alpha–alpha
            for_each_bit(alpha_, [&](size_t q) {
                val += 0.5 * (integral.h2e(p, p, q, q) -
                              integral.h2e(p, q, p, q));
            });
            // alpha–beta
            for_each_bit(beta_, [&](size_t q) {
                val += 0.5 * integral.h2e(p, p, q, q);
            });
        });

        // beta electrons
        for_each_bit(beta_, [&](size_t p) {
            val += integral.h1e(p, p);
            // beta–beta
            for_each_bit(beta_, [&](size_t q) {
                val += 0.5 * (integral.h2e(p, p, q, q) -
                              integral.h2e(p, q, p, q));
            });
            // beta–alpha  (alpha–beta already counted above → double-count to get 1.0×)
            for_each_bit(alpha_, [&](size_t q) {
                val += 0.5 * integral.h2e(p, p, q, q);
            });
        });

        return val;
    }

    /// Off-diagonal matrix element ⟨I|H|J⟩ using Slater-Condon rules.
    double Hij(const Det& other,
               const integral::Integral<integral::ScalerTag>& integral) const {
        uint64_t diff_a = alpha_ ^ other.alpha_;
        uint64_t diff_b = beta_ ^ other.beta_;

        size_t ndiff_a = std::popcount(diff_a);
        size_t ndiff_b = std::popcount(diff_b);
        size_t ndiff   = ndiff_a + ndiff_b;

        if (ndiff == 0 || ndiff > 4) return 0.0;

        // occupied in this, virtual in other
        uint64_t occa = alpha_ & diff_a;
        uint64_t vira = other.alpha_ & diff_a;
        uint64_t occb = beta_ & diff_b;
        uint64_t virb = other.beta_ & diff_b;

        // ---- single excitation (ndiff == 2) ----
        if (ndiff == 2) {
            if (ndiff_a == 2) {          // alpha single: p → q
                size_t p = std::countr_zero(occa);
                size_t q = std::countr_zero(vira);

                double h1 = integral.h1e(p, q);
                double h2 = 0.0;
                for_each_bit(alpha_, [&](size_t r) {
                    h2 += integral.h2e(p, q, r, r) -
                          integral.h2e(p, r, r, q);
                });
                for_each_bit(beta_, [&](size_t r) {
                    h2 += integral.h2e(p, q, r, r);
                });
                return sign(count_between(alpha_, p, q)) * (h1 + h2);
            }
            if (ndiff_b == 2) {          // beta single: p → q
                size_t p = std::countr_zero(occb);
                size_t q = std::countr_zero(virb);

                double h1 = integral.h1e(p, q);
                double h2 = 0.0;
                for_each_bit(beta_, [&](size_t r) {
                    h2 += integral.h2e(p, q, r, r) -
                          integral.h2e(p, r, r, q);
                });
                for_each_bit(alpha_, [&](size_t r) {
                    h2 += integral.h2e(p, q, r, r);
                });
                return sign(count_between(beta_, p, q)) * (h1 + h2);
            }
            return 0.0;
        }

        // ---- double excitation (ndiff == 4) ----
        if (ndiff_a == 4) {              // αα double:  p,r → q,s
            size_t p = std::countr_zero(occa);  occa &= occa - 1;
            size_t r = std::countr_zero(occa);
            size_t q = std::countr_zero(vira);  vira &= vira - 1;
            size_t s = std::countr_zero(vira);
            if (p > r) std::swap(p, r);
            if (q > s) std::swap(q, s);

            double val  = sign(count_between(alpha_, p, q));
            val        *= sign(count_between(other.alpha_, r, s));
            val        *= (integral.h2e(p, q, r, s) -
                           integral.h2e(p, s, r, q));
            return val;
        }
        if (ndiff_b == 4) {              // ββ double:  p,r → q,s
            size_t p = std::countr_zero(occb);  occb &= occb - 1;
            size_t r = std::countr_zero(occb);
            size_t q = std::countr_zero(virb);  virb &= virb - 1;
            size_t s = std::countr_zero(virb);
            if (p > r) std::swap(p, r);
            if (q > s) std::swap(q, s);

            double val  = sign(count_between(beta_, p, q));
            val        *= sign(count_between(other.beta_, r, s));
            val        *= (integral.h2e(p, q, r, s) -
                           integral.h2e(p, s, r, q));
            return val;
        }
        if (ndiff_a == 2 && ndiff_b == 2) { // αβ double:  p→q (α), r→s (β)
            size_t p = std::countr_zero(occa);
            size_t q = std::countr_zero(vira);
            size_t r = std::countr_zero(occb);
            size_t s = std::countr_zero(virb);

            double val  = sign(count_between(alpha_, p, q));
            val        *= sign(count_between(beta_, r, s));
            val        *= integral.h2e(p, q, r, s);
            return val;
        }

        return 0.0;
    }
};

// ===================================================================
// SlaterCondon  –  driver that builds H and diagonalises it
// ===================================================================

template <typename T>
class SlaterCondon {
private:
    std::vector<Det> dets_;
    size_t det_size_;

    /// Enumerate all (norb choose nelec) bit-strings via Gosper's hack.
    static std::vector<uint64_t> enumerate_strings(size_t norb,
                                                    size_t nelec) {
        if (nelec == 0) return {0};
        if (nelec > norb || norb > 64)
            throw std::runtime_error(
                "enumerate_strings: require 0 <= nelec <= norb <= 64");
        std::vector<uint64_t> result;
        uint64_t comb  = (1ULL << nelec) - 1;
        uint64_t limit = 1ULL << norb;
        while (comb < limit) {
            result.push_back(comb);
            uint64_t t = comb | (comb - 1);
            comb = (t + 1) |
                   (((~t & -~t) - 1) >> (std::countr_zero(comb) + 1));
        }
        return result;
    }

    void init(size_t ncas, size_t nalpha, size_t nbeta) {
        if (ncas > 64)
            throw std::runtime_error(
                "ncas must be ≤ 64 (uint64_t bit representation)");

        auto alpha_strs = enumerate_strings(ncas, nalpha);
        auto beta_strs  = enumerate_strings(ncas, nbeta);

        dets_.reserve(alpha_strs.size() * beta_strs.size());
        for (auto a : alpha_strs)
            for (auto b : beta_strs)
                dets_.emplace_back(a, b, ncas);
        det_size_ = dets_.size();
    }

public:
    /// Construct from total electrons and 2·S_z (0 = singlet).
    /// nalpha = (nelec + twos)/2,  nbeta = (nelec - twos)/2.
    SlaterCondon(size_t ncas, size_t nelec, size_t twos = 0) {
        size_t na = (nelec + twos) / 2;
        size_t nb = (nelec - twos) / 2;
        if ((nelec + twos) & 1 || (nelec - twos) & 1)
            throw std::runtime_error(std::format(
                "Invalid (nelec={}, twos={}): nelec±twos must be even",
                nelec, twos));
        init(ncas, na, nb);
    }

    /// Construct from explicit alpha / beta electron counts (use tag).
    SlaterCondon(AlphaBetaTag, size_t ncas, size_t nalpha, size_t nbeta) {
        init(ncas, nalpha, nbeta);
    }

    // ------- build Hamiltonian & run Davidson -------

    void kernel(integral::Integral<>& integral, size_t n_roots = 1,
                size_t start_dim = 5) {

        struct Element {
            MKL_INT row, col;
            double  value;
        };

        std::vector<Element>  elements;
        std::vector<double>   diagonal(det_size_, 0.0);

        auto t0 = std::chrono::high_resolution_clock::now();

#pragma omp parallel
        {
            std::vector<Element> local;
#pragma omp for schedule(static)
            for (MKL_INT i = 0; i < static_cast<MKL_INT>(det_size_); ++i) {
                double diag = dets_[i].Hii(integral);
                if (std::abs(diag) > 1e-12) {
                    diagonal[i] = diag;
                    local.push_back({i, i, diag});
                }
                for (MKL_INT j = i + 1; j < static_cast<MKL_INT>(det_size_); ++j) {
                    double hij = dets_[i].Hij(dets_[j], integral);
                    if (std::abs(hij) > 1e-12)
                        local.push_back({i, j, hij});
                }
            }
#pragma omp critical
            elements.insert(elements.end(), local.begin(), local.end());
        }

        // sort into column-major order
        std::sort(elements.begin(), elements.end(),
                  [](const Element& a, const Element& b) {
                      return (a.row == b.row) ? (a.col < b.col)
                                              : (a.row < b.row);
                  });

        // build CSR
        std::vector<double>  vals;
        std::vector<MKL_INT> cols;
        std::vector<MKL_INT> row_ptr(det_size_ + 1, 0);

        for (auto& e : elements) row_ptr[e.row + 1]++;
        for (size_t i = 0; i < det_size_; ++i)
            row_ptr[i + 1] += row_ptr[i];

        vals.resize(elements.size());
        cols.resize(elements.size());
        std::vector<MKL_INT> counters(det_size_, 0);
        for (auto& e : elements) {
            MKL_INT idx = row_ptr[e.row] + counters[e.row];
            vals[idx]   = e.value;
            cols[idx]   = e.col;
            counters[e.row]++;
        }

        linalg::SparseMatrix<double> sparse_H(
            linalg::MatrixFillMode::UPPER, vals, cols, row_ptr,
            static_cast<MKL_INT>(det_size_),
            static_cast<MKL_INT>(det_size_));

        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << std::format("Build Hamiltonian: {} s\n",
            std::chrono::duration<double>(t1 - t0).count());
        std::cout << std::format("Number of determinants: {}\n", det_size_);

        auto matvec = [&sparse_H](const std::vector<double>& x) {
            return sparse_H * x;
        };

        std::vector<double> evals =
            linalg::davidson_solver(matvec, diagonal.data(), det_size_,
                                    n_roots, start_dim);

        for (size_t n = 0; n < n_roots; ++n)
            std::cout << std::format("  Eigenvalue {:>2}: {:.10f}\n",
                                     n + 1, evals[n] + integral.CoreE());

        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << std::format("Davidson solver time: {} ms\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1)
                .count());
    }
};

} // namespace ci
#endif

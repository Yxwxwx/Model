#pragma once
#ifndef SC_HPP
#define SC_HPP

#include "integral/integral.hpp"
#include "linalg/davidson.hpp"
#include "linalg/sparse_matrix.hpp"

namespace ci {
class Det {
private:
    std::vector<std::size_t> alpha_;
    std::vector<std::size_t> beta_;
    std::size_t norb_;
    std::size_t nelec_;
    std::size_t nalpha_;
    std::size_t nbeta_;

public:
    // Constructor that creates determinant from occupation string
    // '0': empty orbital
    // 'u': alpha electron
    // 'd': beta electron
    // '2': doubly occupied orbital
    Det(const std::string& str)
    {
        alpha_.clear();
        beta_.clear();
        nalpha_ = 0;
        nbeta_ = 0;
        norb_ = 0;
        nelec_ = 0;

        for (auto c : str) {
            if (c == '0') {
                alpha_.emplace_back(0);
                beta_.emplace_back(0);
            }
            else if (c == 'u') {
                alpha_.emplace_back(1);
                beta_.emplace_back(0);
                nalpha_++;
            }
            else if (c == 'd') {
                alpha_.emplace_back(0);
                beta_.emplace_back(1);
                nbeta_++;
            }
            else if (c == '2') {
                alpha_.emplace_back(1);
                beta_.emplace_back(1);
                nalpha_++;
            }
        }

        norb_ = alpha_.size();
        nelec_ = nalpha_ + nbeta_;
    }
    // Convert determinant to string representation
    std::string to_string() const
    {
        std::stringstream ss;
        for (std::size_t i = 0; i < norb_; ++i) {
            if (alpha_[i] == 1 && beta_[i] == 0) {
                ss << "u";
            }
            else if (alpha_[i] == 0 && beta_[i] == 1) {
                ss << "d";
            }
            else if (alpha_[i] == 1 && beta_[i] == 1) {
                ss << "2";
            }
            else if (alpha_[i] == 0 && beta_[i] == 0) {
                ss << "0";
            }
        }
        return ss.str();
    }

    // Count electrons between orbital p and q
    inline std::size_t count_elec(std::size_t p, std::size_t q,
        bool alpha = true) const
    {
        if (p > q) {
            std::swap(p, q);
        }

        std::size_t count = 0;
        const std::vector<std::size_t>& list = alpha ? alpha_ : beta_;

        for (std::size_t i = p + 1; i < q; ++i) {
            if (list[i] == 1) {
                count++;
            }
        }
        return count;
    }

    // Compare two occupation lists and return occupied and virtual orbitals
    inline std::pair<std::vector<std::size_t>, std::vector<std::size_t>>
    compare(const std::vector<std::size_t>& list1,
        const std::vector<std::size_t>& list2) const
    {
        std::vector<std::size_t> occ;
        std::vector<std::size_t> vir;

        for (std::size_t i = 0; i < norb_; ++i) {
            if (list1[i] != list2[i]) {
                if (list1[i] == 1) {
                    occ.emplace_back(i);
                }
                else {
                    vir.emplace_back(i);
                }
            }
        }
        return std::make_pair(occ, vir);
    }

    // Calculate diagonal matrix element (Hii)
    inline const double
    Hii(const integral::Integral<integral::ScalerTag>& integral) const
    {
        double val { 0.0 };
        // sum over all orbitals
        for (std::size_t p = 0; p < norb_; ++p) {
            // Contribution from alpha electrons
            if (alpha_[p] == 1) {
                val += integral.h1e(p, p);

                for (std::size_t q = 0; q < norb_; ++q) {
                    if (alpha_[q] == 1) {
                        val += 0.5 * (integral.h2e(p, p, q, q) - integral.h2e(p, q, p, q));
                    }
                    if (beta_[q] == 1) {
                        val += 0.5 * integral.h2e(p, p, q, q);
                    }
                }
            }
            // Contribution from beta electrons
            if (beta_[p] == 1) {
                val += integral.h1e(p, p);
                for (std::size_t q = 0; q < norb_; ++q) {
                    if (beta_[q] == 1) {
                        val += 0.5 * (integral.h2e(p, p, q, q) - integral.h2e(p, q, p, q));
                    }
                    if (alpha_[q] == 1) {
                        val += 0.5 * integral.h2e(p, p, q, q);
                    }
                }
            }
        }
        return val;
    }
    // Calculate sign based on number of electrons
    inline const double sign(int nelec) const { return std::pow(-1, nelec); }
    // Calculate off-diagonal matrix element (Hij)
    inline const double
    Hij(const Det& det,
        const integral::Integral<integral::ScalerTag>& integral) const
    {
        auto [occa, vira] = compare(alpha_, det.alpha_);
        auto [occb, virb] = compare(beta_, det.beta_);

        auto ndiffa = occa.size();
        auto ndiffb = occb.size();
        auto ndiff = ndiffa + ndiffb;

        // Slater-Condon rules implementation
        if (ndiff > 2) {
            return 0.0;
        }
        else if (ndiff == 1) { // Single excitation
            if (ndiffa == 1) { // alpha
                auto p = occa[0];
                auto q = vira[0];

                double h1 = integral.h1e(p, q);
                double h2 { 0.0 };
                for (std::size_t r = 0; r < norb_; ++r) {
                    if (alpha_[r] == 1) {
                        h2 += integral.h2e(p, q, r, r) - integral.h2e(p, r, r, q);
                    }
                    if (beta_[r] == 1) {
                        h2 += integral.h2e(p, q, r, r);
                    }
                }
                return sign(count_elec(p, q, true)) * (h1 + h2);
            }
            else if (ndiffb == 1) { // beta
                auto p = occb[0];
                auto q = virb[0];
                double h1 = integral.h1e(p, q);
                double h2 { 0.0 };
                for (std::size_t r = 0; r < norb_; ++r) {
                    if (beta_[r] == 1) {
                        h2 += integral.h2e(p, q, r, r) - integral.h2e(p, r, r, q);
                    }
                    if (alpha_[r] == 1) {
                        h2 += integral.h2e(p, q, r, r);
                    }
                }
                return sign(count_elec(p, q, false)) * (h1 + h2);
            }
            else {
                return 0.0;
            }
        }
        else if (ndiff == 2) { // Double excitation
            if (ndiffa == 2) {
                auto p = occa[0];
                auto r = occa[1];
                auto q = vira[0];
                auto s = vira[1];
                if (p > r) {
                    std::swap(p, r);
                }
                if (q > s) {
                    std::swap(q, s);
                }

                double val = sign(count_elec(p, q, true));
                val *= sign(det.count_elec(r, s, true));
                val *= (integral.h2e(p, q, r, s) - integral.h2e(p, s, r, q));

                return val;
            }
            else if (ndiffb == 2) {
                auto p = occb[0];
                auto r = occb[1];
                auto q = virb[0];
                auto s = virb[1];
                if (p > r) {
                    std::swap(p, r);
                }
                if (q > s) {
                    std::swap(q, s);
                }
                double val = sign(count_elec(p, q, false));
                val *= sign(det.count_elec(r, s, false));
                val *= (integral.h2e(p, q, r, s) - integral.h2e(p, s, r, q));

                return val;
            }
            else if (ndiffa == 1 && ndiffb == 1) { // Alpha-beta excitation
                auto p = occa[0];
                auto q = vira[0];
                auto r = occb[0];
                auto s = virb[0];
                double val = sign(count_elec(p, q, true));
                val *= sign(count_elec(r, s, false));
                val *= integral.h2e(p, q, r, s);
                return val;
            }
            else {
                return 0.0;
            }
        }
        else {
            return 0.0;
        }
    }
};
template <typename T>
class SlaterCondon {
private:
    // integral::Integral<> &integral_;
    std::vector<Det> dets_;
    std::size_t det_size_;

public:
    SlaterCondon(const std::string& fileName)
    {
        std::ifstream file(fileName);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                dets_.emplace_back(Det(line));
            }
            file.close();
        }
        det_size_ = dets_.size();
    }

    void kernel(integral::Integral<>& integral, std::size_t n_roots = 1, std::size_t start_dim = 5)
    {
        struct Element {
            MKL_INT row;
            MKL_INT col;
            double value;
        };

        std::vector<Element> elements;
        std::vector<double> diagonal_elements(det_size_, 0.0);

        auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel
        {
            std::vector<Element> local_elements;

#pragma omp for schedule(static)
            for (MKL_INT i = 0; i < det_size_; ++i) {
                // diagonal element
                double diag = dets_[i].Hii(integral);
                if (std::abs(diag) > 1e-12) {
                    diagonal_elements[i] = diag;
                    local_elements.push_back({ i, i, diag });
                }

                // upper triangle
                for (MKL_INT j = i + 1; j < det_size_; ++j) {
                    double hij = dets_[i].Hij(dets_[j], integral);
                    if (std::abs(hij) > 1e-12) {
                        local_elements.push_back({ i, j, hij });
                    }
                }
            }

#pragma omp critical
            {
                elements.insert(elements.end(), local_elements.begin(),
                    local_elements.end());
            }
        }

        // column major
        std::sort(elements.begin(), elements.end(),
            [](const Element& a, const Element& b) {
                return (a.row == b.row) ? (a.col < b.col) : (a.row < b.row);
            });

        // build CSR matrix
        std::vector<double> values;
        std::vector<MKL_INT> columns;
        std::vector<MKL_INT> row_ptr(det_size_ + 1, 0);

        for (const auto& elem : elements) {
            row_ptr[elem.row + 1]++;
        }
        for (MKL_INT i = 0; i < det_size_; ++i) {
            row_ptr[i + 1] += row_ptr[i];
        }

        values.resize(elements.size());
        columns.resize(elements.size());
        std::vector<MKL_INT> row_counters(det_size_, 0);

        for (const auto& elem : elements) {
            MKL_INT row = elem.row;
            MKL_INT idx = row_ptr[row] + row_counters[row];
            values[idx] = elem.value;
            columns[idx] = elem.col;
            row_counters[row]++;
        }

        // build sparse matrix
        linalg::SparseMatrix<double> sparse_H(linalg::MatrixFillMode::UPPER, values,
            columns, row_ptr, det_size_,
            det_size_);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        fmt::println("Build Hailtonian: {} s", duration.count());
        fmt::println("Number of determinants: {}", det_size_);

        // build transformer
        auto matvec = [&sparse_H](const std::vector<double>& x) {
            return sparse_H * x;
        };

        std::vector<double> result = linalg::davidson_solver(matvec, diagonal_elements.data(), det_size_, n_roots, start_dim);
        for (int n = 0; n < n_roots; ++n) {
            fmt::println("  Eigenvalue {:>2}: {}", n + 1, result[n] + integral.CoreE());
        }

        fmt::println("Davidson solver time: {} ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - end)
                .count());
    }
};
} // namespace ci
#endif
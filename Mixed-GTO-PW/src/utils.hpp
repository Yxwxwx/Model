#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "cint.h"

#ifdef _MSC_VER
constexpr double M_PI{3.1415926535897932384626433832795028}
#endif

constexpr int L_MAX = 6;
// l < 15
consteval auto precompute_factorials() {
  std::array<uint64_t, 21> table{};
  table[0] = 1;
  for (std::size_t i = 1; i < 21; ++i) {
    table[i] = table[i - 1] * i;
  }
  return table;
}

// n!!
consteval auto precompute_double_factorials() {
  std::array<uint64_t, 31> table{};
  table[0] = 1;
  if (table.size() > 1) {
    table[1] = 1;
  }
  if (table.size() > 2) {
    table[2] = 2;
  }
  for (std::size_t i = 3; i < 31; ++i) {
    table[i] = table[i - 2] * i;
  }
  return table;
}

static constexpr auto FACTORIAL_TABLE = precompute_factorials();
static constexpr auto DOUBLE_FACTORIAL_TABLE = precompute_double_factorials();

constexpr uint64_t fact(int n) {
  return (n <= 0) ? 1 : FACTORIAL_TABLE[static_cast<std::size_t>(n)];
}

constexpr uint64_t fact2(int n) {
  return (n <= 0) ? 1 : DOUBLE_FACTORIAL_TABLE[static_cast<std::size_t>(n)];
}

struct ShellInfo {
  std::vector<double> exps;
  std::vector<double> coeffs;
  std::array<int, 3> lmn_base;
  std::array<double, 3> center;
  int n_prim;

  void print() const {
    std::cout << "==> ShellInfo: <== " << '\n';
    std::cout << "n_prim: " << n_prim << '\n';
    std::cout << "exp: ";
    for (auto exp : exps) {
      std::cout << std::format("{:.8e}", exp) << " ";
    }
    std::cout << '\n';
    std::cout << "coeff: ";
    for (auto coeff : coeffs) {
      std::cout << std::format("{:.8e}", coeff) << " ";
    }
    std::cout << '\n';
    std::cout << "lmn: { ";
    for (auto lmn : lmn_base) {
      std::cout << lmn << ", ";
    }
    std::cout << " }" << '\n';
    std::cout << "center: { ";
    for (auto c : center) {
      std::cout << std::format("{:.8e}", c) << ", ";
    }
    std::cout << " }" << '\n';
  }
};
enum class ShellType : std::uint8_t {
  Cartesian,
  Sphere,
};
inline std::vector<std::array<int, 3>> gen_cartesian_lmn(const int l) noexcept {
  std::vector<std::array<int, 3>> res;
  for (int x = l; x >= 0; --x) {
    for (int y = l - x; y >= 0; --y) {
      int z = l - x - y;
      res.push_back({x, y, z});
    }
  }
  return res;
}

// Raw Cartesian monomials -> normalized real solid harmonics, row-major
// (ncart, 2*l+1). p order: x,y,z; l>=2 order: m=-l,...,+l.
// Includes the s/p angular factors (PySCF cart2sph(normalized=None)).
inline const std::vector<long double>& cart2sph_coeff(int angular) {
  if (angular < 0 || angular > L_MAX) {
    throw std::invalid_argument(
        "Spherical angular momentum must be between 0 and 6");
  }
  // ponytail: cached through L_MAX; extend the factorial tables for higher
  // shells.
  static const auto matrices = [] {
    std::array<std::vector<long double>, L_MAX + 1> result;
    for (int l = 0; l <= L_MAX; ++l) {
      const int nsph = 2 * l + 1;
      auto& matrix = result[l];
      matrix.resize((static_cast<std::size_t>(l) + 1) * (l + 2) / 2 * nsph);
      if (l == 1) {
        for (int i = 0; i < 3; ++i) {
          matrix[i * 3 + i] =
              std::sqrt(3.0L / (4 * std::numbers::pi_v<long double>));
        }
        continue;
      }
      // Expand (x+iy)^m times the m-th derivative of the Legendre polynomial:
      // sum_k (-1)^k (2l-2k)! z^(l-m-2k) (x*x+y*y+z*z)^k
      //       / (2^l k! (l-k)! (l-m-2k)!). Real/imaginary parts give +/-m.
      for (int m = 0; m <= l; ++m) {
        const long double norm =
            std::sqrt(
                (2.0L * l + 1) * fact(l - m) /
                (4 * std::numbers::pi_v<long double> * fact(l + m))) *
            (m == 0 ? 1.0L : std::sqrt(2.0L));
        for (int k = 0; k <= (l - m) / 2; ++k) {
          const long double legendre = (k % 2 ? -1.0L : 1.0L) *
              fact(2 * l - 2 * k) /
              (std::ldexp(1.0L, l) * fact(l - k) * fact(l - m - 2 * k));
          for (int y_power = 0; y_power <= m; ++y_power) {
            const int column = y_power % 2 ? l - m : l + m;
            const long double phase = y_power % 4 < 2 ? 1.0L : -1.0L;
            const long double binomial = static_cast<long double>(fact(m)) /
                (static_cast<long double>(fact(y_power)) * fact(m - y_power));
            for (int x_pairs = 0; x_pairs <= k; ++x_pairs) {
              for (int y_pairs = 0; y_pairs <= k - x_pairs; ++y_pairs) {
                const int z_pairs = k - x_pairs - y_pairs;
                const int x = m - y_power + 2 * x_pairs;
                const int z = l - m - 2 * k + 2 * z_pairs;
                const int row = (l - x) * (l - x + 1) / 2 + z;
                matrix[row * nsph + column] += norm * legendre * phase *
                    binomial /
                    (static_cast<long double>(fact(x_pairs)) * fact(y_pairs) *
                     fact(z_pairs));
              }
            }
          }
        }
      }
    }
    return result;
  }();
  return matrices[angular];
}

template <ShellType T = ShellType::Cartesian>
inline std::vector<ShellInfo> get_shell_infos(
    int sh_id,
    const int* atm,
    const int* bas,
    const double* env) {
  const int atom_idx = bas(ATOM_OF, sh_id);
  const int l_val = bas(ANG_OF, sh_id);
  const int n_prim = bas(NPRIM_OF, sh_id);
  const int n_ctr = bas(NCTR_OF, sh_id);
  const int ptr_exp = bas(PTR_EXP, sh_id);
  const int ptr_coeff = bas(PTR_COEFF, sh_id);
  const int ptr_coord = atm(PTR_COORD, atom_idx);

  std::array<double, 3> center = {
      env[ptr_coord + 0], env[ptr_coord + 1], env[ptr_coord + 2]};

  const std::vector<double> exps(env + ptr_exp, env + ptr_exp + n_prim);

  std::vector<std::array<int, 3>> lmn_list;
  std::vector<ShellInfo> shells;
  //  Cartesian lmn
  if constexpr (T == ShellType::Cartesian) {
    lmn_list = gen_cartesian_lmn(l_val);
    shells.reserve(n_ctr * lmn_list.size());
  }

  // PySCF orders contractions first, then Cartesian components within each.
  for (int ctr = 0; ctr < n_ctr; ++ctr) {
    const double* coeff_ptr =
        env + ptr_coeff + static_cast<std::ptrdiff_t>(ctr) * n_prim;
    const std::vector<double> coeffs(coeff_ptr, coeff_ptr + n_prim);
    for (const auto& lmn : lmn_list) {
      ShellInfo sh;
      sh.exps = exps;
      sh.coeffs = coeffs;
      sh.lmn_base = lmn;
      sh.center = center;
      sh.n_prim = n_prim;
      shells.push_back(std::move(sh));
    }
  }

  return shells;
}

inline std::vector<std::pair<int, std::array<double, 3>>> get_nuclear_info(
    const int* atm,
    const int natm,
    const double* env) {
  std::vector<std::pair<int, std::array<double, 3>>> nuclei;
  nuclei.reserve(natm);

  for (int i = 0; i < natm; ++i) {
    // nuclear charge
    const int charge = atm(CHARGE_OF, i);

    // nuclear position
    const int ptr_coord = atm(PTR_COORD, i);
    const std::array<double, 3> pos = {
        env[ptr_coord + 0], env[ptr_coord + 1], env[ptr_coord + 2]};

    nuclei.emplace_back(charge, pos);
  }
  return nuclei;
}

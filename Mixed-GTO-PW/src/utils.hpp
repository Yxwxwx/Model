#pragma once
#include <array>
#include <cstdint>
#include <format>
#include <iostream>
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
  if (table.size() > 1) table[1] = 1;
  if (table.size() > 2) table[2] = 2;
  for (std::size_t i = 3; i < 31; ++i) {
    table[i] = table[i - 2] * i;
  }
  return table;
}

static constexpr auto FACTORIAL_TABLE = precompute_factorials();
static constexpr auto DOUBLE_FACTORIAL_TABLE = precompute_double_factorials();

inline constexpr uint64_t fact(int n) {
  return (n <= 0) ? 1 : FACTORIAL_TABLE[static_cast<std::size_t>(n)];
}

inline constexpr uint64_t fact2(int n) {
  return (n <= 0) ? 1 : DOUBLE_FACTORIAL_TABLE[static_cast<std::size_t>(n)];
}

struct ShellInfo {
  std::vector<double> exps;
  std::vector<double> coeffs;
  std::array<int, 3> lmn_base;
  std::array<double, 3> center;
  int n_prim;

  void print() const {
    std::cout << "==> ShellInfo: <== " << std::endl;
    std::cout << "n_prim: " << n_prim << std::endl;
    std::cout << "exp: ";
    for (auto exp : exps) std::cout << std::format("{:.8e}", exp) << " ";
    std::cout << std::endl;
    std::cout << "coeff: ";
    for (auto coeff : coeffs) std::cout << std::format("{:.8e}", coeff) << " ";
    std::cout << std::endl;
    std::cout << "lmn: { ";
    for (auto lmn : lmn_base) std::cout << lmn << ", ";
    std::cout << " }" << std::endl;
    std::cout << "center: { ";
    for (auto c : center) std::cout << std::format("{:.8e}", c) << ", ";
    std::cout << " }" << std::endl;
  }
};
enum class ShellType {
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

template <ShellType T = ShellType::Cartesian>
inline std::vector<ShellInfo> get_shell_infos(int sh_id, const int* atm,
                                              const int* bas,
                                              const double* env) {
  const int atom_idx = bas(ATOM_OF, sh_id);
  const int l_val = bas(ANG_OF, sh_id);
  const int n_prim = bas(NPRIM_OF, sh_id);
  const int ptr_exp = bas(PTR_EXP, sh_id);
  const int ptr_coeff = bas(PTR_COEFF, sh_id);
  const int ptr_coord = atm(PTR_COORD, atom_idx);

  std::array<double, 3> center = {env[ptr_coord + 0], env[ptr_coord + 1],
                                  env[ptr_coord + 2]};

  // exp and coeff
  std::vector<double> exps;
  std::vector<double> coeffs;
  exps.reserve(n_prim);
  coeffs.reserve(n_prim);
  for (int i = 0; i < n_prim; ++i) {
    exps.push_back(env[ptr_exp + i]);
    coeffs.push_back(env[ptr_coeff + i]);
  }

  std::vector<std::array<int, 3>> lmn_list;
  std::vector<ShellInfo> shells;
  //  Cartesian lmn
  if constexpr (T == ShellType::Cartesian) {
    lmn_list = gen_cartesian_lmn(l_val);
    shells.reserve(lmn_list.size());
  }

  for (const auto& lmn : lmn_list) {
    ShellInfo sh;
    sh.exps = exps;
    sh.coeffs = coeffs;
    sh.lmn_base = lmn;
    sh.center = center;
    sh.n_prim = n_prim;
    shells.push_back(std::move(sh));
  }

  return shells;
}

inline std::vector<std::pair<int, std::array<double, 3>>> get_nuclear_info(
    const int* atm, const int natm, const double* env) {
  std::vector<std::pair<int, std::array<double, 3>>> nuclei;
  nuclei.reserve(natm);

  for (int i = 0; i < natm; ++i) {
    // nuclear charge
    const int charge = atm(CHARGE_OF, i);

    // nuclear position
    const int ptr_coord = atm(PTR_COORD, i);
    const std::array<double, 3> pos = {env[ptr_coord + 0], env[ptr_coord + 1],
                                       env[ptr_coord + 2]};

    nuclei.push_back({charge, pos});
  }
  return nuclei;
}
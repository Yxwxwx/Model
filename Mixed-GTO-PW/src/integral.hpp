#pragma once

#include "cint.h"
#include "plane.hpp"
// Cartesian integrals
// overlap
void int1e_ovlp_cart(std::complex<double>* buf, const int* shls, const int* atm,
                     const int natm, const int* bas, const int nbas,
                     const double* env, const double* k_vector) noexcept;
// kinetic
void int1e_kin_cart(std::complex<double>* buf, const int* shls, const int* atm,
                    const int natm, const int* bas, const int nbas,
                    const double* env, const double* k_vector) noexcept;
// nuclear
void int1e_nuc_cart(std::complex<double>* buf, const int* shls, const int* atm,
                    const int natm, const int* bas, const int nbas,
                    const double* env, const double* k_vector) noexcept;
// 2e integrals
void int2e_cart(std::complex<double>* buf, const int* shls, const int* atm,
                const int natm, const int* bas, const int nbas,
                const double* env, const double* k_vector) noexcept;
// Spherical integrals
// overlap
void int1e_ovlp_sph(std::complex<double>* buf, const int* shls, const int* atm,
                    const int natm, const int* bas, const int nbas,
                    const double* env, const double* k_vector) noexcept;
// kinetic
void int1e_kin_sph(std::complex<double>* buf, const int* shls, const int* atm,
                   const int natm, const int* bas, const int nbas,
                   const double* env, const double* k_vector) noexcept;
// nuclear
void int1e_nuc_sph(std::complex<double>* buf, const int* shls, const int* atm,
                   const int natm, const int* bas, const int nbas,
                   const double* env, const double* k_vector) noexcept;
// 2e integrals
void int2e_sph(std::complex<double>* buf, const int* shls, const int* atm,
               const int natm, const int* bas, const int nbas,
               const double* env, const double* k_vector) noexcept;

template <typename Func, typename... ExtraArgs>
inline void int1e_generic_cart(std::complex<double>* buf, const int* shls,
                               const int* atm, const int natm, const int* bas,
                               const int nbas, const double* env,
                               const double* k_ptr, Func&& func,
                               ExtraArgs&&... args) {
  // 1.Shell info
  const auto shells_i =
      get_shell_infos<ShellType::Cartesian>(shls[0], atm, bas, env);
  //   for (const auto& sh : shells_i) sh.print();

  // 2. Plane wave vector
  const std::array<double, 3> k = {k_ptr[0], k_ptr[1], k_ptr[2]};

  // 3. Fill buffer
  for (std::size_t i = 0; i < shells_i.size(); ++i)
    buf[i] = func(shells_i[i], k, std::forward<ExtraArgs>(args)...);
}

template <typename Func>
inline void int2e_generic_cart(std::complex<double>* buf, const int* shls,
                               const int* atm, const int natm, const int* bas,
                               const int nbas, const double* env,
                               const double* k_ptr, Func&& func) {
  // 1.Shell info
  const auto shells_p =
      get_shell_infos<ShellType::Cartesian>(shls[0], atm, bas, env);
  //   for (const auto& sh : shells_p) sh.print();
  const auto shells_q =
      get_shell_infos<ShellType::Cartesian>(shls[1], atm, bas, env);
  //   for (const auto& sh : shells_q) sh.print();
  const auto shells_r =
      get_shell_infos<ShellType::Cartesian>(shls[2], atm, bas, env);
  //   for (const auto& sh : shells_r) sh.print();

  // 2. Plane wave vector
  const std::array<double, 3> k = {k_ptr[0], k_ptr[1], k_ptr[2]};

  // 3. Fill buffer
  std::size_t idx = 0;
  for (std::size_t p = 0; p < shells_p.size(); ++p)
    for (std::size_t q = 0; q < shells_q.size(); ++q)
      for (std::size_t r = 0; r < shells_r.size(); ++r)
        buf[idx++] = func(shells_p[p], k, shells_q[q], shells_r[r]);
}

// For Overlap and Kinetic only accept sh and k
const auto ovlp_wrapper = [](const ShellInfo& sh,
                             const std::array<double, 3>& k) {
  return contracted_overlap_gp(sh.exps, sh.coeffs, sh.lmn_base, sh.center, k);
};

const auto kin_wrapper = [](const ShellInfo& sh,
                            const std::array<double, 3>& k) {
  return contracted_kinetic_gp(sh.exps, sh.coeffs, sh.lmn_base, sh.center, k);
};

// For Nuclear, it extra receives nuclei (i.e., q)
const auto nuc_wrapper =
    [](const ShellInfo& sh, const std::array<double, 3>& k,
       const std::vector<std::pair<int, std::array<double, 3>>>& q) {
      return contracted_nuclear_gp(sh.exps, sh.coeffs, sh.lmn_base, sh.center,
                                   k, q);
    };
// For 2e three sh and one k
const auto twoe_wrapper = [](const ShellInfo& sh1,
                             const std::array<double, 3>& k,
                             const ShellInfo& sh2, const ShellInfo& sh3) {
  return contracted_eri_gpgg(sh1.exps, sh1.coeffs, sh1.lmn_base, sh1.center, k,
                             sh2.exps, sh2.coeffs, sh2.lmn_base, sh2.center,
                             sh3.exps, sh3.coeffs, sh3.lmn_base, sh3.center);
};
// overlap
inline void int1e_ovlp_cart(std::complex<double>* buf, const int* shls,
                            const int* atm, const int natm, const int* bas,
                            const int nbas, const double* env,
                            const double* k_vector) noexcept {
  int1e_generic_cart(buf, shls, atm, natm, bas, nbas, env, k_vector,
                     ovlp_wrapper);
}
// kinetic
inline void int1e_kin_cart(std::complex<double>* buf, const int* shls,
                           const int* atm, const int natm, const int* bas,
                           const int nbas, const double* env,
                           const double* k_vector) noexcept {
  int1e_generic_cart(buf, shls, atm, natm, bas, nbas, env, k_vector,
                     kin_wrapper);
}
// nuclear
inline void int1e_nuc_cart(std::complex<double>* buf, const int* shls,
                           const int* atm, const int natm, const int* bas,
                           const int nbas, const double* env,
                           const double* k_vector) noexcept {
  const auto nuclei = get_nuclear_info(atm, natm, env);
  int1e_generic_cart(buf, shls, atm, natm, bas, nbas, env, k_vector,
                     nuc_wrapper, nuclei);
}

// 2e integrals
inline void int2e_cart(std::complex<double>* buf, const int* shls,
                       const int* atm, const int natm, const int* bas,
                       const int nbas, const double* env,
                       const double* k_vector) noexcept {
  int2e_generic_cart(buf, shls, atm, natm, bas, nbas, env, k_vector,
                     twoe_wrapper);
}
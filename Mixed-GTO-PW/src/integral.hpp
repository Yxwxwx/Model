#pragma once

#include "cint.h"
#include "plane.hpp"
#include "utils.hpp"

template <typename Output, typename Func, typename... ExtraArgs>
inline void int1e_generic_cart(
    std::complex<Output>* buf,
    const int* shls,
    const int* atm,
    const int natm,
    const int* bas,
    const int nbas,
    const double* env,
    const double* k_ptr,
    Func&& func,
    const ExtraArgs&... args) {
  // 1.Shell info
  const auto shells_i =
      get_shell_infos<ShellType::Cartesian>(shls[0], atm, bas, env);
  //   for (const auto& sh : shells_i) sh.print();

  // 2. Plane wave vector
  const std::array<double, 3> k = {k_ptr[0], k_ptr[1], k_ptr[2]};

  // 3. Fill buffer
  for (std::size_t i = 0; i < shells_i.size(); ++i) {
    buf[i] = static_cast<std::complex<Output>>(func(shells_i[i], k, args...));
  }
}

template <typename Output, typename Func>
inline void int2e_generic_cart(
    std::complex<Output>* buf,
    const int* shls,
    const int* atm,
    const int natm,
    const int* bas,
    const int nbas,
    const double* env,
    const double* k_ptr,
    Func&& func) {
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
  for (std::size_t p = 0; p < shells_p.size(); ++p) {
    for (std::size_t q = 0; q < shells_q.size(); ++q) {
      for (std::size_t r = 0; r < shells_r.size(); ++r) {
        buf[idx++] = static_cast<std::complex<Output>>(
            func(shells_p[p], k, shells_q[q], shells_r[r]));
      }
    }
  }
}

// For Overlap and Kinetic only accept sh and k
template <typename Real = double>
inline constexpr auto ovlp_wrapper =
    [](const ShellInfo& sh, const std::array<double, 3>& k) {
      return contracted_overlap_gp<Real>(
          sh.exps, sh.coeffs, sh.lmn_base, sh.center, k);
    };

template <typename Real = double>
inline constexpr auto kin_wrapper =
    [](const ShellInfo& sh, const std::array<double, 3>& k) {
      return contracted_kinetic_gp<Real>(
          sh.exps, sh.coeffs, sh.lmn_base, sh.center, k);
    };

// For Nuclear, it extra receives nuclei (i.e., q)
template <typename Real = double>
inline constexpr auto nuc_wrapper =
    [](const ShellInfo& sh,
       const std::array<double, 3>& k,
       const std::vector<std::pair<int, std::array<double, 3>>>& q) {
      return contracted_nuclear_gp<Real>(
          sh.exps, sh.coeffs, sh.lmn_base, sh.center, k, q);
    };
// For 2e three sh and one k
template <typename Real = double>
inline constexpr auto twoe_wrapper = [](const ShellInfo& sh1,
                                        const std::array<double, 3>& k,
                                        const ShellInfo& sh2,
                                        const ShellInfo& sh3) {
  return contracted_eri_gpgg<Real>(
      sh1.exps,
      sh1.coeffs,
      sh1.lmn_base,
      sh1.center,
      k,
      sh2.exps,
      sh2.coeffs,
      sh2.lmn_base,
      sh2.center,
      sh3.exps,
      sh3.coeffs,
      sh3.lmn_base,
      sh3.center);
};
// overlap
template <typename Real = double, typename Output = double>
inline void int1e_ovlp_cart(
    std::complex<Output>* buf,
    const int* shls,
    const int* atm,
    const int natm,
    const int* bas,
    const int nbas,
    const double* env,
    const double* k_vector) noexcept {
  int1e_generic_cart(
      buf, shls, atm, natm, bas, nbas, env, k_vector, ovlp_wrapper<Real>);
}
// kinetic
template <typename Real = double, typename Output = double>
inline void int1e_kin_cart(
    std::complex<Output>* buf,
    const int* shls,
    const int* atm,
    const int natm,
    const int* bas,
    const int nbas,
    const double* env,
    const double* k_vector) noexcept {
  int1e_generic_cart(
      buf, shls, atm, natm, bas, nbas, env, k_vector, kin_wrapper<Real>);
}
// nuclear
template <typename Real = double, typename Output = double>
inline void int1e_nuc_cart(
    std::complex<Output>* buf,
    const int* shls,
    const int* atm,
    const int natm,
    const int* bas,
    const int nbas,
    const double* env,
    const double* k_vector) noexcept {
  const auto nuclei = get_nuclear_info(atm, natm, env);
  int1e_generic_cart(
      buf,
      shls,
      atm,
      natm,
      bas,
      nbas,
      env,
      k_vector,
      nuc_wrapper<Real>,
      nuclei);
}

// 2e integrals
template <typename Real = double, typename Output = double>
inline void int2e_cart(
    std::complex<Output>* buf,
    const int* shls,
    const int* atm,
    const int natm,
    const int* bas,
    const int nbas,
    const double* env,
    const double* k_vector) noexcept {
  int2e_generic_cart(
      buf, shls, atm, natm, bas, nbas, env, k_vector, twoe_wrapper<Real>);
}

template <typename Real = double, typename Output = double>
inline void int2e_gppg_cart(
    std::complex<Output>* buf,
    const int* shls,
    const int* atm,
    const int natm,
    const int* bas,
    const int nbas,
    const double* env,
    const double* k_vector,
    const double* l_vector) noexcept {
  const auto shells_p =
      get_shell_infos<ShellType::Cartesian>(shls[0], atm, bas, env);
  const auto shells_q =
      get_shell_infos<ShellType::Cartesian>(shls[1], atm, bas, env);
  const std::array<double, 3> k = {k_vector[0], k_vector[1], k_vector[2]};
  const std::array<double, 3> l = {l_vector[0], l_vector[1], l_vector[2]};
  std::size_t idx = 0;
  for (const auto& p : shells_p) {
    for (const auto& q : shells_q) {
      buf[idx++] = static_cast<std::complex<Output>>(contracted_eri_gppg<Real>(
          p.exps,
          p.coeffs,
          p.lmn_base,
          p.center,
          k,
          l,
          q.exps,
          q.coeffs,
          q.lmn_base,
          q.center));
    }
  }
}

// Transform each GTO axis separately, keeping contraction blocks independent.
// All intermediate integrals and transformations retain the selected precision.
template <typename Real, std::size_t N, typename Kernel, typename... Vectors>
inline void integral_sph(
    std::complex<double>* buf,
    const int* shls,
    const int* atm,
    int natm,
    const int* bas,
    int nbas,
    const double* env,
    Kernel kernel,
    Vectors... vectors) {
  std::array<std::size_t, N> dimensions;
  std::size_t total = 1;
  for (std::size_t axis = 0; axis < N; ++axis) {
    const int l = bas[shls[axis] * BAS_SLOTS + ANG_OF];
    cart2sph_coeff(l); // Validate before allocating the Cartesian tensor.
    dimensions[axis] = (l + 1) * (l + 2) / 2 *
        static_cast<std::size_t>(bas[shls[axis] * BAS_SLOTS + NCTR_OF]);
    total *= dimensions[axis];
  }
  std::vector<std::complex<Real>> values(total);
  kernel(values.data(), shls, atm, natm, bas, nbas, env, vectors...);
  for (std::size_t axis = 0; axis < N; ++axis) {
    const int l = bas[shls[axis] * BAS_SLOTS + ANG_OF];
    const int nctr = bas[shls[axis] * BAS_SLOTS + NCTR_OF];
    const int ncart = (l + 1) * (l + 2) / 2, nsph = 2 * l + 1;
    const auto& coefficients = cart2sph_coeff(l);
    std::size_t left = 1, right = 1;
    for (std::size_t i = 0; i < axis; ++i) {
      left *= dimensions[i];
    }
    for (std::size_t i = axis + 1; i < N; ++i) {
      right *= dimensions[i];
    }
    std::vector<std::complex<Real>> next(left * nctr * nsph * right);
    for (std::size_t outer = 0; outer < left; ++outer) {
      for (int ctr = 0; ctr < nctr; ++ctr) {
        for (int s = 0; s < nsph; ++s) {
          for (int c = 0; c < ncart; ++c) {
            const Real coefficient = Real(coefficients[c * nsph + s]);
            if (coefficient == 0) {
              continue;
            }
            const auto src =
                (outer * dimensions[axis] + ctr * ncart + c) * right;
            const auto dst = ((outer * nctr + ctr) * nsph + s) * right;
            for (std::size_t inner = 0; inner < right; ++inner) {
              next[dst + inner] += coefficient * values[src + inner];
            }
          }
        }
      }
    }
    values = std::move(next);
    dimensions[axis] = nctr * nsph;
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    buf[i] = static_cast<std::complex<double>>(values[i]);
  }
}

// Spherical shell dimensions: nctr * (2*l+1); last GTO axis is fastest.
template <typename Real = double>
inline void int1e_ovlp_sph(
    std::complex<double>* buf,
    const int* shls,
    const int* atm,
    int natm,
    const int* bas,
    int nbas,
    const double* env,
    const double* k_vector) {
  integral_sph<Real, 1>(
      buf,
      shls,
      atm,
      natm,
      bas,
      nbas,
      env,
      int1e_ovlp_cart<Real, Real>,
      k_vector);
}

// Spherical shell dimensions: nctr * (2*l+1); last GTO axis is fastest.
template <typename Real = double>
inline void int1e_kin_sph(
    std::complex<double>* buf,
    const int* shls,
    const int* atm,
    int natm,
    const int* bas,
    int nbas,
    const double* env,
    const double* k_vector) {
  integral_sph<Real, 1>(
      buf,
      shls,
      atm,
      natm,
      bas,
      nbas,
      env,
      int1e_kin_cart<Real, Real>,
      k_vector);
}

// Spherical shell dimensions: nctr * (2*l+1); last GTO axis is fastest.
template <typename Real = double>
inline void int1e_nuc_sph(
    std::complex<double>* buf,
    const int* shls,
    const int* atm,
    int natm,
    const int* bas,
    int nbas,
    const double* env,
    const double* k_vector) {
  integral_sph<Real, 1>(
      buf,
      shls,
      atm,
      natm,
      bas,
      nbas,
      env,
      int1e_nuc_cart<Real, Real>,
      k_vector);
}

// Spherical shell dimensions: nctr * (2*l+1); last GTO axis is fastest.
template <typename Real = double>
inline void int2e_sph(
    std::complex<double>* buf,
    const int* shls,
    const int* atm,
    int natm,
    const int* bas,
    int nbas,
    const double* env,
    const double* k_vector) {
  integral_sph<Real, 3>(
      buf, shls, atm, natm, bas, nbas, env, int2e_cart<Real, Real>, k_vector);
}

// Spherical shell dimensions: nctr * (2*l+1); last GTO axis is fastest.
template <typename Real = double>
inline void int2e_gppg_sph(
    std::complex<double>* buf,
    const int* shls,
    const int* atm,
    int natm,
    const int* bas,
    int nbas,
    const double* env,
    const double* k_vector,
    const double* l_vector) {
  integral_sph<Real, 2>(
      buf,
      shls,
      atm,
      natm,
      bas,
      nbas,
      env,
      int2e_gppg_cart<Real, Real>,
      k_vector,
      l_vector);
}

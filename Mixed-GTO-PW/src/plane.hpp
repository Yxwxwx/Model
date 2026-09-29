#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>

#include <Faddeeva.hh>
#include "utils.hpp"

// Normalization of GTOs
inline void normalization(
    std::vector<double>& coeff,
    const std::array<int, 3>& lmn,
    const std::vector<double>& exps) {
  const auto [l, m, n] = lmn;
  const auto L = l + m + n;

  const auto prefactor = std::pow(M_PI, 1.5) *
      static_cast<double>(fact2(2 * l - 1)) *
      static_cast<double>(fact2(2 * m - 1)) *
      static_cast<double>(fact2(2 * n - 1));

  std::vector<double> norm(exps.size());
  for (std::size_t i = 0; i < exps.size(); i++) {
    norm[i] = std::sqrt(
        prefactor * std::pow(exps[i], l + m + n + 1.5) /
        std::pow(2.0, 2 * (l + m + n) + 1.5) /
        static_cast<double>(fact2(2 * l - 1)) /
        static_cast<double>(fact2(2 * m - 1)) /
        static_cast<double>(fact2(2 * n - 1)));
  }

  double N{0.0};
  for (std::size_t i = 0; i < coeff.size(); i++) {
    for (std::size_t j = 0; j < coeff.size(); j++) {
      N += norm[i] * norm[j] * coeff[i] * coeff[j] /
          std::pow(exps[i] + exps[j], L + 1.5);
    }
  }

  N *= prefactor;
  N = 1.0 / std::sqrt(N);

  std::transform(
      coeff.begin(),
      coeff.end(),
      norm.begin(),
      coeff.begin(),
      [N](double c, double n) { return c * N * n; });
}

// Iterative definition of Hermite Gaussian coefficients
template <typename Real = double>
inline const std::complex<Real> expansion_coefficients(
    const int i,
    const int j,
    const int t,
    const std::complex<Real>& Qx,
    const Real a,
    const Real b) noexcept {
  /*
    Recursive definition of Hermite Gaussian coefficients.
    Returns a complex number.
    a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
    b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
    i,j: orbital angular momentum number on Gaussian 'a' and 'b'
    t: number nodes in Hermite (depends on type of integral,
       e.g. always zero for overlap integrals)
    Qx: distance between origins of Gaussian 'a' and 'b' in a complex plane
  */
  const auto p = a + b;
  const auto q = a * b / p;
  if (t < 0 || t > (i + j)) {
    return {0.0, 0.0};
  }

  std::complex<Real> result{0.0, 0.0};
  if (i == 0 && j == 0 && t == 0) {
    result = std::exp(-q * Qx * Qx); // K_AB
  } else if (j == 0) {
    result = (Real(1) / (Real(2) * p)) *
            expansion_coefficients(i - 1, j, t - 1, Qx, a, b) -
        (q * Qx / a) * expansion_coefficients(i - 1, j, t, Qx, a, b) +
        Real(t + 1) * expansion_coefficients(i - 1, j, t + 1, Qx, a, b);
  } else {
    result = (Real(1) / (Real(2) * p)) *
            expansion_coefficients(i, j - 1, t - 1, Qx, a, b) +
        (q * Qx / b) * expansion_coefficients(i, j - 1, t, Qx, a, b) +
        Real(t + 1) * expansion_coefficients(i, j - 1, t + 1, Qx, a, b);
  }

  return result;
}

template <typename Real = double>
inline const std::array<std::complex<Real>, 3> gaussian_product_center(
    const Real a,
    const std::array<std::complex<Real>, 3>& A,
    const Real b,
    const std::array<std::complex<Real>, 3>& B) noexcept {
  const Real denom = a + b;
  return {
      (a * A[0] + b * B[0]) / denom,
      (a * A[1] + b * B[1]) / denom,
      (a * A[2] + b * B[2]) / denom};
}
template <typename Real = double>
inline const std::complex<Real> boys(
    const int m,
    const std::complex<Real>& T) noexcept {
  // 1. Use a stable series near the real axis, retaining long-double guard
  // digits. F_m(T) = 1F1(m+1/2; m+3/2; -T)/(2m+1); for Re(T)>=0, use Kummer's
  // transformation: https://dlmf.nist.gov/13.2.E39. For negative Re(T), keep
  // guard digits throughout the representable exponential range instead of
  // seeding the recurrence with double Faddeeva. The imaginary bound limits
  // cancellation in the series.
  const auto magnitude = std::abs(T);
  const auto exponential_limit =
      std::log(std::numeric_limits<Real>::max()) - Real(16);
  std::complex<Real> f0;
  if (std::abs(T.imag()) < 10.0 &&
      (magnitude < 50.0 || (T.real() < 0.0 && magnitude < exponential_limit))) {
    // Extra precision also limits cancellation for nearly imaginary arguments.
    const std::complex<long double> z{T.real(), T.imag()};
    const bool direct = T.real() < 0.0;
    // A common seed keeps neighboring orders consistent in Hermite differences.
    // In this region every upward-recurrence multiplier has magnitude < 1.
    const bool recur = m > 0 && T.real() < -m - 1.0;
    const int order = recur ? 0 : m;
    std::complex<long double> term = 1.0L / (2.0L * order + 1.0L);
    auto sum = term;
    const int max_terms = 256 + static_cast<int>(2 * magnitude);
    for (int n = 1; n < max_terms; ++n) {
      if (direct) {
        term *= -z * (2.0L * order + 2.0L * n - 1.0L) /
            (n * (2.0L * order + 2.0L * n + 1.0L));
      } else {
        term *= z / (order + n + 0.5L);
      }
      sum += term;
      // Component norms avoid squaring huge values under fast-math.
      if (std::max(std::abs(term.real()), std::abs(term.imag())) <=
          std::numeric_limits<long double>::epsilon() *
              std::max(std::abs(sum.real()), std::abs(sum.imag()))) {
        break;
      }
    }
    f0 = static_cast<std::complex<Real>>(direct ? sum : std::exp(-z) * sum);
    if (!recur) {
      return f0;
    }
  } else {
    // Outside the series region: F0(T) = sqrt(pi)/(2*sqrt(T)) * erf(sqrt(T)).
    const std::complex<Real> sqrt_T = std::sqrt(T);
    f0 = Real(0.88622692545275801365L) *
        std::complex<Real>(Faddeeva::erf(std::complex<double>(sqrt_T))) /
        sqrt_T;
  }
  if (m == 0) {
    return f0;
  }

  // 3. Fi(T) = [ (2i-1) * Fi-1(T) - exp(-T) ] / 2T
  const std::complex<Real> exp_neg_T = std::exp(-T);
  std::complex<Real> fi = f0;
  for (int i = 1; i <= m; ++i) {
    fi = ((Real(2) * i - Real(1)) * fi - exp_neg_T) / (Real(2) * T);
  }
  return fi;
}
template <typename Real = double>
inline const std::complex<Real> coulomb_auxiliary_hermite_integrals(
    int t,
    int u,
    int v,
    int n,
    Real p,
    const std::array<std::complex<Real>, 3>& PC,
    const std::complex<Real>& T) noexcept {
  /*
      Returns the Coulomb auxiliary Hermite integrals
      Returns a complex number.
      Arguments:
      t,u,v:   order of Coulomb Hermite derivative in x,y,z
               (see defs in Helgaker and Taylor)
      n:       order of Boys function
      PC[x,y,z]: Cartesian vector distance between Gaussian
               composite center P and nuclear center C
      T:     p * RPC * RPC (Distance between P and C)
  */

  if (t == 0 && u == 0 && v == 0) {
    return std::pow(Real(-2) * p, n) * boys(n, T);
  }

  // recursively reduce t, u, v
  // prioritize t, then u, then v
  if (t > 0) {
    auto term = PC[0] *
        coulomb_auxiliary_hermite_integrals(t - 1, u, v, n + 1, p, PC, T);
    if (t > 1) {
      term += static_cast<Real>(t - 1) *
          coulomb_auxiliary_hermite_integrals(t - 2, u, v, n + 1, p, PC, T);
    }
    return term;
  } else if (u > 0) {
    auto term = PC[1] *
        coulomb_auxiliary_hermite_integrals(t, u - 1, v, n + 1, p, PC, T);
    if (u > 1) {
      term += static_cast<Real>(u - 1) *
          coulomb_auxiliary_hermite_integrals(t, u - 2, v, n + 1, p, PC, T);
    }
    return term;
  } else { // v > 0
    auto term = PC[2] *
        coulomb_auxiliary_hermite_integrals(t, u, v - 1, n + 1, p, PC, T);
    if (v > 1) {
      term += static_cast<Real>(v - 1) *
          coulomb_auxiliary_hermite_integrals(t, u, v - 2, n + 1, p, PC, T);
    }
    return term;
  }
}

// Overlap integral between two Gaussians
template <typename Real = double>
inline std::complex<Real> overlap_elem(
    const Real a,
    const std::array<int, 3>& lmn1,
    const std::array<std::complex<Real>, 3>& A,
    const Real b,
    const std::array<int, 3>& lmn2,
    const std::array<std::complex<Real>, 3>& B) noexcept {
  /*
  Evaluates overlap integral between two Gaussians
  Returns a complex number.
  a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
  b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
  lmn1: int array containing orbital angular momentum (e.g. {1, 0, 0})
        for Gaussian 'a'
  lmn2: int array containing orbital angular momentum for Gaussian 'b'
  A:    array containing origin of Gaussian 'a', e.g. {1.0, 2.0, 0.0}
  B:    array containing origin of Gaussian 'b'
*/

  const auto [l1, m1, n1] = lmn1;
  const auto [l2, m2, n2] = lmn2;

  const auto Qx = A[0] - B[0];
  const auto Qy = A[1] - B[1];
  const auto Qz = A[2] - B[2];

  const auto p = a + b;
  const auto factor = std::pow(std::numbers::pi_v<Real> / p, Real(1.5));

  const auto S1 = expansion_coefficients(l1, l2, 0, Qx, a, b);
  const auto S2 = expansion_coefficients(m1, m2, 0, Qy, a, b);
  const auto S3 = expansion_coefficients(n1, n2, 0, Qz, a, b);

  return S1 * S2 * S3 * factor;
}

// Uncontracted nuclear attraction integral
template <typename Real = double>
inline std::complex<Real> nuclear_elem(
    const Real a,
    const std::array<int, 3>& lmn1,
    const std::array<std::complex<Real>, 3>& A,
    const Real b,
    const std::array<int, 3>& lmn2,
    const std::array<std::complex<Real>, 3>& B,
    const std::array<Real, 3>& C) {
  /*
      Evaluates nuclear attraction integral between two Gaussians
       Returns a std::complex<Real>.
       a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
       b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
       lmn1: int array containing orbital angular momentum (e.g. {1,0,0})
             for Gaussian 'a'
       lmn2: int array containing orbital angular momentum for Gaussian 'b'
       A:    array containing origin of Gaussian 'a', e.g. {1.0, 2.0, 0.0}
       B:    array containing origin of Gaussian 'b'
       C:    array containing origin of nuclear center 'C'
  */
  const auto [l1, m1, n1] = lmn1;
  const auto [l2, m2, n2] = lmn2;

  const auto p = a + b;
  const auto P = gaussian_product_center(a, A, b, B);
  const auto T = p *
      (std::pow(P[0] - C[0], 2) + std::pow(P[1] - C[1], 2) +
       std::pow(P[2] - C[2], 2));

  const std::array<std::complex<Real>, 3> PC_vec = {
      P[0] - C[0], P[1] - C[1], P[2] - C[2]};

  // sum initialized to 0.0
  std::complex<Real> val{0.0, 0.0};

  for (int t = 0; t <= l1 + l2; t++) {
    for (int u = 0; u <= m1 + m2; u++) {
      for (int v = 0; v <= n1 + n2; v++) {
        val += expansion_coefficients(l1, l2, t, A[0] - B[0], a, b) *
            expansion_coefficients(m1, m2, u, A[1] - B[1], a, b) *
            expansion_coefficients(n1, n2, v, A[2] - B[2], a, b) *
            coulomb_auxiliary_hermite_integrals(t, u, v, 0.0, p, PC_vec, T);
      }
    }
  }

  val *= Real(2) * std::numbers::pi_v<Real> / p;
  return val;
}

// Uncontracted electron repulsion integral
template <typename Real = double>
inline const std::complex<Real> electron_repulsion(
    const Real a,
    const std::array<int, 3>& lmn1,
    const std::array<std::complex<Real>, 3>& A,
    const Real b,
    const std::array<int, 3>& lmn2,
    const std::array<std::complex<Real>, 3>& B,
    const Real c,
    const std::array<int, 3>& lmn3,
    const std::array<std::complex<Real>, 3>& C,
    const Real d,
    const std::array<int, 3>& lmn4,
    const std::array<std::complex<Real>, 3>& D) {
  const auto [l1, m1, n1] = lmn1;
  const auto [l2, m2, n2] = lmn2;
  const auto [l3, m3, n3] = lmn3;
  const auto [l4, m4, n4] = lmn4;

  const auto p = a + b;
  const auto q = c + d;
  const auto alpha = p * q / (p + q);
  const auto P = gaussian_product_center(a, A, b, B);
  const auto Q = gaussian_product_center(c, C, d, D);
  const auto T = alpha *
      (std::pow(P[0] - Q[0], 2) + std::pow(P[1] - Q[1], 2) +
       std::pow(P[2] - Q[2], 2));
  const std::array<std::complex<Real>, 3> PQ_vec = {
      P[0] - Q[0], P[1] - Q[1], P[2] - Q[2]};

  std::complex<Real> val{0.0, 0.0};
  for (int t = 0; t <= l1 + l2; t++) {
    for (int u = 0; u <= m1 + m2; u++) {
      for (int v = 0; v <= n1 + n2; v++) {
        for (int tau = 0; tau <= l3 + l4; tau++) {
          for (int nu = 0; nu <= m3 + m4; nu++) {
            for (int phi = 0; phi <= n3 + n4; phi++) {
              val += expansion_coefficients(l1, l2, t, A[0] - B[0], a, b) *
                  expansion_coefficients(m1, m2, u, A[1] - B[1], a, b) *
                  expansion_coefficients(n1, n2, v, A[2] - B[2], a, b) *
                  expansion_coefficients(l3, l4, tau, C[0] - D[0], c, d) *
                  expansion_coefficients(m3, m4, nu, C[1] - D[1], c, d) *
                  expansion_coefficients(n3, n4, phi, C[2] - D[2], c, d) *
                  Real((tau + nu + phi) % 2 ? -1 : 1) *
                  coulomb_auxiliary_hermite_integrals(
                         t + tau, u + nu, v + phi, 0, alpha, PQ_vec, T);
            }
          }
        }
      }
    }
  }

  val *= Real(2) * std::pow(std::numbers::pi_v<Real>, Real(2.5)) /
      (p * q * std::sqrt(p + q));
  return val;
}

// Contracted overlap integral between GTO and PW
template <typename Real = double>
inline std::complex<Real> contracted_overlap_gp(
    const std::vector<double>& exps,
    const std::vector<double>& coeffs,
    const std::array<int, 3>& lmn,
    const std::array<double, 3>& center,
    const std::array<double, 3>& k_vector) noexcept {
  const std::array<Real, 3> k = {k_vector[0], k_vector[1], k_vector[2]};
  std::complex<Real> S = {0.0, 0.0};
  const auto k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];

  for (std::size_t i = 0; i < exps.size(); ++i) {
    const Real alpha = exps[i];
    const Real coeff = coeffs[i];

    const auto pref =
        std::exp(
            std::complex<Real>(
                0.0, k[0] * center[0] + k[1] * center[1] + k[2] * center[2])) *
        std::exp(-k2 / (2.0 * alpha));

    const auto a = alpha / 2.0;
    const auto b = alpha / 2.0;

    const std::array<std::complex<Real>, 3> A = {
        center[0], center[1], center[2]};
    const std::array<std::complex<Real>, 3> B = {
        Real(center[0]) + std::complex<Real>(0, k[0] / alpha),
        Real(center[1]) + std::complex<Real>(0, k[1] / alpha),
        Real(center[2]) + std::complex<Real>(0, k[2] / alpha)};

    const std::array<int, 3> lmn_pw = {0, 0, 0};
    const auto S_prim = overlap_elem(a, lmn, A, b, lmn_pw, B);

    S += coeff * pref * S_prim;
  }
  const auto pw_norm =
      std::pow(Real(2) * std::numbers::pi_v<Real>, Real(-1.5)) * std::sqrt(k2);
  return S * pw_norm;
}

template <typename Real = double>
inline std::complex<Real> contracted_kinetic_gp(
    const std::vector<double>& exps,
    const std::vector<double>& coeffs,
    const std::array<int, 3>& lmn,
    const std::array<double, 3>& center,
    const std::array<double, 3>& k_vector) noexcept {
  const std::array<Real, 3> k = {k_vector[0], k_vector[1], k_vector[2]};
  const auto k2 = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]) / 2.0;
  return k2 * contracted_overlap_gp<Real>(exps, coeffs, lmn, center, k_vector);
}

// Contracted nuclear repulsion
template <typename Real = double>
inline std::complex<Real> contracted_nuclear_gp(
    const std::vector<double>& exps,
    const std::vector<double>& coeffs,
    const std::array<int, 3>& lmn,
    const std::array<double, 3>& center,
    const std::array<double, 3>& k_vector,
    const std::vector<std::pair<int, std::array<double, 3>>>& q) {
  const std::array<Real, 3> k = {k_vector[0], k_vector[1], k_vector[2]};
  std::complex<Real> N{0.0, 0.0};
  const auto k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];

  for (std::size_t i = 0; i < exps.size(); ++i) {
    for (const auto& nuc_cent : q) {
      const Real alpha = exps[i];
      const Real coeff = coeffs[i];

      const auto pref =
          std::exp(
              std::complex<Real>(
                  0.0,
                  k[0] * center[0] + k[1] * center[1] + k[2] * center[2])) *
          std::exp(-k2 / (2.0 * alpha));

      const auto a = alpha / 2.0;
      const auto b = alpha / 2.0;

      const std::array<std::complex<Real>, 3> A = {
          center[0], center[1], center[2]};
      const std::array<std::complex<Real>, 3> B = {
          Real(center[0]) + std::complex<Real>(0, k[0] / alpha),
          Real(center[1]) + std::complex<Real>(0, k[1] / alpha),
          Real(center[2]) + std::complex<Real>(0, k[2] / alpha)};

      const std::array<int, 3> lmn_pw = {0, 0, 0};
      const auto N_prim = nuclear_elem(
          a,
          lmn,
          A,
          b,
          lmn_pw,
          B,
          std::array<Real, 3>{
              nuc_cent.second[0], nuc_cent.second[1], nuc_cent.second[2]});
      N += -static_cast<Real>(nuc_cent.first) * coeff * pref * N_prim;
    }
  }
  const auto pw_norm =
      std::pow(Real(2) * std::numbers::pi_v<Real>, Real(-1.5)) * std::sqrt(k2);
  return N * pw_norm;
}

// Corrected Contracted ERI in Chemist's Notation: (GP | GG)
// Matches the physics of <pq || kr> where electron 1 is (p, k) and electron 2
// is (q, r)
template <typename Real = double>
inline std::complex<Real> contracted_eri_gpgg(
    const std::vector<double>& exps_p,
    const std::vector<double>& coeffs_p,
    const std::array<int, 3>& lmn_p,
    const std::array<double, 3>& center_p,
    const std::array<double, 3>& k_vector,
    const std::vector<double>& exps_q,
    const std::vector<double>& coeffs_q,
    const std::array<int, 3>& lmn_q,
    const std::array<double, 3>& center_q,
    const std::vector<double>& exps_r,
    const std::vector<double>& coeffs_r,
    const std::array<int, 3>& lmn_r,
    const std::array<double, 3>& center_r) {
  const std::array<Real, 3> k = {k_vector[0], k_vector[1], k_vector[2]};
  std::complex<Real> eri_total{0.0, 0.0};
  const Real k_mag2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];

  for (std::size_t i = 0; i < exps_p.size(); ++i) {
    const Real alpha = exps_p[i];
    const Real weight_p = coeffs_p[i];

    const Real exp_a = alpha / 2.0;
    const Real exp_b = alpha / 2.0;

    const std::array<std::complex<Real>, 3> center_a = {
        center_p[0], center_p[1], center_p[2]};
    const std::array<std::complex<Real>, 3> center_b = {
        Real(center_p[0]) + std::complex<Real>(0, k[0] / alpha),
        Real(center_p[1]) + std::complex<Real>(0, k[1] / alpha),
        Real(center_p[2]) + std::complex<Real>(0, k[2] / alpha)};

    const std::complex<Real> phase_factor(
        0.0, k[0] * center_p[0] + k[1] * center_p[1] + k[2] * center_p[2]);
    const std::complex<Real> prefactor_e1 =
        std::exp(phase_factor) * std::exp(-k_mag2 / (2.0 * alpha));

    for (std::size_t j = 0; j < exps_q.size(); ++j) {
      for (std::size_t m = 0; m < exps_r.size(); ++m) {
        const Real beta = exps_q[j];
        const Real gamma = exps_r[m];
        const Real weight_qr = Real(coeffs_q[j]) * coeffs_r[m];

        const std::array<std::complex<Real>, 3> center_c = {
            center_q[0], center_q[1], center_q[2]};
        const std::array<std::complex<Real>, 3> center_d = {
            center_r[0], center_r[1], center_r[2]};

        const std::array<int, 3> lmn_pw = {0, 0, 0};

        const auto eri_prim = electron_repulsion(
            exp_a,
            lmn_p,
            center_a,
            exp_b,
            lmn_pw,
            center_b, // e1
            beta,
            lmn_q,
            center_c,
            gamma,
            lmn_r,
            center_d // e2
        );

        eri_total += weight_p * weight_qr * prefactor_e1 * eri_prim;
      }
    }
  }

  const Real pw_norm =
      std::pow(Real(2) * std::numbers::pi_v<Real>, Real(-1.5)) *
      std::sqrt(k_mag2);
  return eri_total * pw_norm;
}

// Chemist's notation (G_p P_k | P_l G_q): exp(+ik.r1) exp(-il.r2).
// Both real GTOs use the same raw Cartesian normalization as GP|GG.
template <typename Real = double>
inline std::complex<Real> contracted_eri_gppg(
    const std::vector<double>& exps_p,
    const std::vector<double>& coeffs_p,
    const std::array<int, 3>& lmn_p,
    const std::array<double, 3>& center_p,
    const std::array<double, 3>& k_vector,
    const std::array<double, 3>& l_vector,
    const std::vector<double>& exps_q,
    const std::vector<double>& coeffs_q,
    const std::array<int, 3>& lmn_q,
    const std::array<double, 3>& center_q) {
  // Retain guard digits through centers, Hermite recurrence, and contraction.
  const std::array<Real, 3> k = {k_vector[0], k_vector[1], k_vector[2]};
  const std::array<Real, 3> l = {l_vector[0], l_vector[1], l_vector[2]};
  const Real k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
  const Real l2 = l[0] * l[0] + l[1] * l[1] + l[2] * l[2];
  // With the repository's |k| normalization, a zero wavevector has zero weight.
  if (k2 == 0.0 || l2 == 0.0) {
    return {0.0, 0.0};
  }

  const std::array<int, 3> lmn_pw = {0, 0, 0};
  const std::array<std::complex<Real>, 3> A = {
      center_p[0], center_p[1], center_p[2]};
  const std::array<std::complex<Real>, 3> D = {
      center_q[0], center_q[1], center_q[2]};
  const auto phase = std::exp(
      std::complex<Real>(
          0.0,
          k[0] * center_p[0] + k[1] * center_p[1] + k[2] * center_p[2] -
              l[0] * center_q[0] - l[1] * center_q[1] - l[2] * center_q[2]));

  std::complex<Real> eri_total{0.0, 0.0};
  for (std::size_t i = 0; i < exps_p.size(); ++i) {
    const Real alpha = exps_p[i];
    auto B = A;
    for (int axis = 0; axis < 3; ++axis) {
      B[axis] += std::complex<Real>(0.0L, k[axis] / alpha);
    }
    for (std::size_t j = 0; j < exps_q.size(); ++j) {
      const Real beta = exps_q[j];
      auto C = D;
      for (int axis = 0; axis < 3; ++axis) {
        C[axis] -= std::complex<Real>(0.0L, l[axis] / beta);
      }
      const auto primitive = electron_repulsion(
          alpha / 2.0,
          lmn_p,
          A,
          alpha / 2.0,
          lmn_pw,
          B,
          beta / 2.0,
          lmn_pw,
          C,
          beta / 2.0,
          lmn_q,
          D);
      eri_total += static_cast<Real>(coeffs_p[i]) * coeffs_q[j] *
          std::exp(-k2 / (2.0 * alpha) - l2 / (2.0 * beta)) * primitive;
    }
  }
  const Real pw_norm = std::pow(Real(2) * std::numbers::pi_v<Real>, Real(-3)) *
      std::sqrt(k2) * std::sqrt(l2);
  return eri_total * phase * pw_norm;
}

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>

#include <Faddeeva.hh>
#include "utils.hpp"

using namespace std::complex_literals;
// Normalization of GTOs
inline void normalization(std::vector<double>& coeff,
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
    norm[i] = std::sqrt(prefactor * std::pow(exps[i], l + m + n + 1.5) /
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

  std::transform(coeff.begin(), coeff.end(), norm.begin(), coeff.begin(),
                 [N](double c, double n) { return c * N * n; });
}

// Iterative definition of Hermite Gaussian coefficients
inline const std::complex<double> expansion_coefficients(
    const int i, const int j, const int t, const std::complex<double>& Qx,
    const double a, const double b) noexcept {
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

  std::complex<double> result{0.0, 0.0};
  if (i == 0 && j == 0 && t == 0) {
    result = std::exp(-q * Qx * Qx);  // K_AB
  } else if (j == 0) {
    result =
        (1.0 / (2.0 * p)) * expansion_coefficients(i - 1, j, t - 1, Qx, a, b) -
        (q * Qx / a) * expansion_coefficients(i - 1, j, t, Qx, a, b) +
        (t + 1.0) * expansion_coefficients(i - 1, j, t + 1, Qx, a, b);
  } else {
    result =
        (1.0 / (2.0 * p)) * expansion_coefficients(i, j - 1, t - 1, Qx, a, b) +
        (q * Qx / b) * expansion_coefficients(i, j - 1, t, Qx, a, b) +
        (t + 1.0) * expansion_coefficients(i, j - 1, t + 1, Qx, a, b);
  }

  return result;
}

inline const std::array<std::complex<double>, 3> gaussian_product_center(
    const double a, const std::array<std::complex<double>, 3>& A,
    const double b, const std::array<std::complex<double>, 3>& B) noexcept {
  const double denom = a + b;
  return {(a * A[0] + b * B[0]) / denom, (a * A[1] + b * B[1]) / denom,
          (a * A[2] + b * B[2]) / denom};
}
inline const std::complex<double> boys(const int m,
                                       const std::complex<double>& T) noexcept {
  constexpr double eps = 1e-14;
  // 1. For small values of T (|T|<10) : Taylor series
  if (std::abs(T) < 10.0) {
    std::complex<double> sum = 0.0;
    std::complex<double> term = 1.0;

    for (int n = 0; n < 100; ++n) {
      const double denom = std::tgamma(m + n + 1.5);
      sum += term / denom;

      term *= T;

      if (std::abs(term) < eps) break;
    }

    return 0.5 * std::exp(-T) * std::tgamma(m + 0.5) * sum;
  }

  // 2. For large values of T (|T|>=10) : Faddeeva function and recurrence
  // F0(T) = sqrt(pi)/(2*sqrt(T)) * erf(sqrt(T))
  const std::complex<double> sqrt_T = std::sqrt(T);
  std::complex<double> f0 =
      0.88622692545275801365 * Faddeeva::erf(sqrt_T) / sqrt_T;
  if (m == 0) return f0;

  // 3. Fi(T) = [ (2i-1) * Fi-1(T) - exp(-T) ] / 2T
  const std::complex<double> exp_neg_T = std::exp(-T);
  std::complex<double> fi = f0;
  for (int i = 1; i <= m; ++i) {
    fi = ((2.0 * i - 1.0) * fi - exp_neg_T) / (2.0 * T);
  }
  return fi;
}
inline const std::complex<double> coulomb_auxiliary_hermite_integrals(
    int t, int u, int v, int n, double p,
    const std::array<std::complex<double>, 3>& PC,
    const std::complex<double>& T) noexcept {
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
    return std::pow(-2.0 * p, n) * boys(n, T);
  }

  // recursively reduce t, u, v
  // prioritize t, then u, then v
  if (t > 0) {
    auto term = PC[0] * coulomb_auxiliary_hermite_integrals(t - 1, u, v, n + 1,
                                                            p, PC, T);
    if (t > 1) {
      term += static_cast<double>(t - 1) *
              coulomb_auxiliary_hermite_integrals(t - 2, u, v, n + 1, p, PC, T);
    }
    return term;
  } else if (u > 0) {
    auto term = PC[1] * coulomb_auxiliary_hermite_integrals(t, u - 1, v, n + 1,
                                                            p, PC, T);
    if (u > 1) {
      term += static_cast<double>(u - 1) *
              coulomb_auxiliary_hermite_integrals(t, u - 2, v, n + 1, p, PC, T);
    }
    return term;
  } else {  // v > 0
    auto term = PC[2] * coulomb_auxiliary_hermite_integrals(t, u, v - 1, n + 1,
                                                            p, PC, T);
    if (v > 1) {
      term += static_cast<double>(v - 1) *
              coulomb_auxiliary_hermite_integrals(t, u, v - 2, n + 1, p, PC, T);
    }
    return term;
  }
}

// Overlap integral between two Gaussians
inline std::complex<double> overlap_elem(
    const double a, const std::array<int, 3>& lmn1,
    const std::array<std::complex<double>, 3>& A, const double b,
    const std::array<int, 3>& lmn2,
    const std::array<std::complex<double>, 3>& B) noexcept {
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
  const auto factor = std::pow(M_PI / p, 1.5);

  const auto S1 = expansion_coefficients(l1, l2, 0, Qx, a, b);
  const auto S2 = expansion_coefficients(m1, m2, 0, Qy, a, b);
  const auto S3 = expansion_coefficients(n1, n2, 0, Qz, a, b);

  return S1 * S2 * S3 * factor;
}

// Uncontracted nuclear attraction integral
inline std::complex<double> nuclear_elem(
    const double a, const std::array<int, 3>& lmn1,
    const std::array<std::complex<double>, 3>& A, const double b,
    const std::array<int, 3>& lmn2,
    const std::array<std::complex<double>, 3>& B,
    const std::array<double, 3>& C) {
  /*
      Evaluates nuclear attraction integral between two Gaussians
       Returns a std::complex<double>.
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
  const auto T = p * (std::pow(P[0] - C[0], 2) + std::pow(P[1] - C[1], 2) +
                      std::pow(P[2] - C[2], 2));

  const std::array<std::complex<double>, 3> PC_vec = {P[0] - C[0], P[1] - C[1],
                                                      P[2] - C[2]};

  // sum initialized to 0.0
  std::complex<double> val{0.0, 0.0};

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

  val *= 2.0 * M_PI / p;
  return val;
}

// Uncontracted electron repulsion integral
inline const std::complex<double> electron_repulsion(
    const double a, const std::array<int, 3>& lmn1,
    const std::array<std::complex<double>, 3>& A, const double b,
    const std::array<int, 3>& lmn2,
    const std::array<std::complex<double>, 3>& B, const double c,
    const std::array<int, 3>& lmn3,
    const std::array<std::complex<double>, 3>& C, const double d,
    const std::array<int, 3>& lmn4,
    const std::array<std::complex<double>, 3>& D) {
  const auto [l1, m1, n1] = lmn1;
  const auto [l2, m2, n2] = lmn2;
  const auto [l3, m3, n3] = lmn3;
  const auto [l4, m4, n4] = lmn4;

  const auto p = a + b;
  const auto q = c + d;
  const auto alpha = p * q / (p + q);
  const auto P = gaussian_product_center(a, A, b, B);
  const auto Q = gaussian_product_center(c, C, d, D);
  const auto T = alpha * (std::pow(P[0] - Q[0], 2) + std::pow(P[1] - Q[1], 2) +
                          std::pow(P[2] - Q[2], 2));
  const std::array<std::complex<double>, 3> PQ_vec = {P[0] - Q[0], P[1] - Q[1],
                                                      P[2] - Q[2]};

  std::complex<double> val{0.0, 0.0};
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
                     std::pow(-1, tau + nu + phi) *
                     coulomb_auxiliary_hermite_integrals(
                         t + tau, u + nu, v + phi, 0, alpha, PQ_vec, T);
            }
          }
        }
      }
    }
  }

  val *= 2 * std::pow(M_PI, 2.5) / (p * q * std::sqrt(p + q));
  return val;
}

// Contracted overlap integral between GTO and PW
inline const std::complex<double> contracted_overlap_gp(
    const std::vector<double>& exps, const std::vector<double>& coeffs,
    const std::array<int, 3>& lmn, const std::array<double, 3>& center,
    const std::array<double, 3>& k) noexcept {
  std::complex<double> S = {0.0, 0.0};
  const auto k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];

  for (std::size_t i = 0; i < exps.size(); ++i) {
    const auto alpha = exps[i];
    const auto coeff = coeffs[i];

    const auto pref =
        std::exp(std::complex<double>(
            0.0, k[0] * center[0] + k[1] * center[1] + k[2] * center[2])) *
        std::exp(-k2 / (2.0 * alpha));

    const auto a = alpha / 2.0;
    const auto b = alpha / 2.0;

    const std::array<std::complex<double>, 3> A = {center[0], center[1],
                                                   center[2]};
    const std::array<std::complex<double>, 3> B = {
        center[0] + std::complex<double>(0, k[0] / alpha),
        center[1] + std::complex<double>(0, k[1] / alpha),
        center[2] + std::complex<double>(0, k[2] / alpha)};

    const std::array<int, 3> lmn_pw = {0, 0, 0};
    const auto S_prim = overlap_elem(a, lmn, A, b, lmn_pw, B);

    S += coeff * pref * S_prim;
  }
  const auto pw_norm = std::pow(2.0 * M_PI, -1.5) * std::sqrt(k2);
  return S * pw_norm;
}

inline const std::complex<double> contracted_kinetic_gp(
    const std::vector<double>& exps, const std::vector<double>& coeffs,
    const std::array<int, 3>& lmn, const std::array<double, 3>& center,
    const std::array<double, 3>& k) noexcept {
  const auto k2 = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]) / 2.0;
  return k2 * contracted_overlap_gp(exps, coeffs, lmn, center, k);
}

// Contracted nuclear repulsion
std::complex<double> contracted_nuclear_gp(
    const std::vector<double>& exps, const std::vector<double>& coeffs,
    const std::array<int, 3>& lmn, const std::array<double, 3>& center,
    const std::array<double, 3>& k,
    const std::vector<std::pair<int, std::array<double, 3>>>& q) {
  std::complex<double> N{0.0, 0.0};
  const auto k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];

  for (std::size_t i = 0; i < exps.size(); ++i) {
    for (const auto& nuc_cent : q) {
      const auto alpha = exps[i];
      const auto coeff = coeffs[i];

      const auto pref =
          std::exp(std::complex<double>(
              0.0, k[0] * center[0] + k[1] * center[1] + k[2] * center[2])) *
          std::exp(-k2 / (2.0 * alpha));

      const auto a = alpha / 2.0;
      const auto b = alpha / 2.0;

      const std::array<std::complex<double>, 3> A = {center[0], center[1],
                                                     center[2]};
      const std::array<std::complex<double>, 3> B = {
          center[0] + std::complex<double>(0, k[0] / alpha),
          center[1] + std::complex<double>(0, k[1] / alpha),
          center[2] + std::complex<double>(0, k[2] / alpha)};

      const std::array<int, 3> lmn_pw = {0, 0, 0};
      const auto N_prim =
          nuclear_elem(a, lmn, A, b, lmn_pw, B, nuc_cent.second);
      N += -static_cast<double>(nuc_cent.first) * coeff * pref * N_prim;
    }
  }
  const auto pw_norm = std::pow(2.0 * M_PI, -1.5) * std::sqrt(k2);
  return N * pw_norm;
}

// Corrected Contracted ERI in Chemist's Notation: (GP | GG)
// Matches the physics of <pq || kr> where electron 1 is (p, k) and electron 2
// is (q, r)
std::complex<double> contracted_eri_gpgg(
    const std::vector<double>& exps_p, const std::vector<double>& coeffs_p,
    const std::array<int, 3>& lmn_p, const std::array<double, 3>& center_p,
    const std::array<double, 3>& k, const std::vector<double>& exps_q,
    const std::vector<double>& coeffs_q, const std::array<int, 3>& lmn_q,
    const std::array<double, 3>& center_q, const std::vector<double>& exps_r,
    const std::vector<double>& coeffs_r, const std::array<int, 3>& lmn_r,
    const std::array<double, 3>& center_r) {
  std::complex<double> eri_total{0.0, 0.0};
  const double k_mag2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];

  for (std::size_t i = 0; i < exps_p.size(); ++i) {
    const double alpha = exps_p[i];
    const double weight_p = coeffs_p[i];

    const double exp_a = alpha / 2.0;
    const double exp_b = alpha / 2.0;

    const std::array<std::complex<double>, 3> center_a = {
        center_p[0], center_p[1], center_p[2]};
    const std::array<std::complex<double>, 3> center_b = {
        center_p[0] + std::complex<double>(0, k[0] / alpha),
        center_p[1] + std::complex<double>(0, k[1] / alpha),
        center_p[2] + std::complex<double>(0, k[2] / alpha)};

    const std::complex<double> phase_factor(
        0.0, k[0] * center_p[0] + k[1] * center_p[1] + k[2] * center_p[2]);
    const std::complex<double> prefactor_e1 =
        std::exp(phase_factor) * std::exp(-k_mag2 / (2.0 * alpha));

    for (std::size_t j = 0; j < exps_q.size(); ++j) {
      for (std::size_t m = 0; m < exps_r.size(); ++m) {
        const double beta = exps_q[j];
        const double gamma = exps_r[m];
        const double weight_qr = coeffs_q[j] * coeffs_r[m];

        const std::array<std::complex<double>, 3> center_c = {
            center_q[0], center_q[1], center_q[2]};
        const std::array<std::complex<double>, 3> center_d = {
            center_r[0], center_r[1], center_r[2]};

        const std::array<int, 3> lmn_pw = {0, 0, 0};

        const auto eri_prim = electron_repulsion(
            exp_a, lmn_p, center_a, exp_b, lmn_pw, center_b,  // e1
            beta, lmn_q, center_c, gamma, lmn_r, center_d     // e2
        );

        eri_total += weight_p * weight_qr * prefactor_e1 * eri_prim;
      }
    }
  }

  const double pw_norm = std::pow(2.0 * M_PI, -1.5) * std::sqrt(k_mag2);
  return eri_total * pw_norm;
}
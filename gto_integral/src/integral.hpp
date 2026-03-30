#include <algorithm>
#include <array>
#include <cmath>
// #include <fmt/core.h>
#include <format>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <vector>

#ifdef _MSC_VER
constexpr double M_PI { 3.1415926535897932384626433832795028 }
#endif

// Cache for previously computed expansion coefficients
std::unordered_map<std::string, double>
    coefficient_cache;

// Cache for previously computed Coulomb auxiliary Hermite integrals
std::unordered_map<std::string, double> coulomb_cache;

inline const int fact2(int n)
{
    return (n <= 0) ? 1 : static_cast<int>(std::tgamma(n + 1));
}

inline void normalization(std::vector<double>& coeff,
    const std::array<int, 3>& lmn,
    const std::vector<double>& exps)
{
    auto [l, m, n] = lmn;
    auto L = l + m + n;

    double prefactor = std::pow(M_PI, 1.5) * fact2(2 * l - 1) * fact2(2 * m - 1) * fact2(2 * n - 1);

    std::vector<double> norm(exps.size());
    for (size_t i = 0; i < exps.size(); i++) {
        norm[i] = std::sqrt(prefactor * std::pow(exps[i], l + m + n + 1.5) / std::pow(2.0, 2 * (l + m + n) + 1.5) / fact2(2 * l - 1) / fact2(2 * m - 1) / fact2(2 * n - 1));
    }

    double N = 0.0;
    for (size_t i = 0; i < coeff.size(); i++) {
        for (size_t j = 0; j < coeff.size(); j++) {
            N += norm[i] * norm[j] * coeff[i] * coeff[j] / std::pow(exps[i] + exps[j], L + 1.5);
        }
    }

    N *= prefactor;
    N = 1.0 / std::sqrt(N);

    std::transform(coeff.begin(), coeff.end(), norm.begin(), coeff.begin(),
        [N](double c, double n) { return c * N * n; });
}

// Generate a unique key for the coefficient based on parameters
inline const std::string generate_key(int i, int j, int t, double Qx, double a,
    double b)
{
    return std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(t) + "_" + std::to_string(Qx) + "_" + std::to_string(a) + "_" + std::to_string(b);
}

// Iterative definition of Hermite Gaussian coefficients
inline const double expansion_coefficients(int i, int j, int t, double Qx, double a,
    double b)
{
    /*
        Recursive definition of Hermite Gaussian coefficients.
        Returns a double.
        a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
        i,j: orbital angular momentum number on Gaussian 'a' and 'b'
        t: number nodes in Hermite (depends on type of integral,
           e.g. always zero for overlap integrals)
        Qx: distance between origins of Gaussian 'a' and 'b'
    */
    double p = a + b;
    double q = a * b / p;

    if (t < 0 || t > (i + j)) {
        // Out of bounds for t
        return 0.0;
    }

    // Check the cache first
    std::string key = generate_key(i, j, t, Qx, a, b);
    if (coefficient_cache.find(key) != coefficient_cache.end()) {
        return coefficient_cache[key];
    }

    double result = 0.0;
    if (i == 0 && j == 0 && t == 0) {
        result = std::exp(-q * Qx * Qx); // K_AB
    }
    else if (j == 0) {
        // Decrement index i
        result = (1 / (2 * p)) * expansion_coefficients(i - 1, j, t - 1, Qx, a, b) - (q * Qx / a) * expansion_coefficients(i - 1, j, t, Qx, a, b) + (t + 1) * expansion_coefficients(i - 1, j, t + 1, Qx, a, b);
    }
    else {
        // Decrement index j
        result = (1 / (2 * p)) * expansion_coefficients(i, j - 1, t - 1, Qx, a, b) + (q * Qx / b) * expansion_coefficients(i, j - 1, t, Qx, a, b) + (t + 1) * expansion_coefficients(i, j - 1, t + 1, Qx, a, b);
    }

    // Cache the result for future use
    coefficient_cache[key] = result;
    return result;
}

// Evaluates overlap integral between two Gaussians
inline const double overlap_elem(double a, const std::array<int, 3>& lmn1,
    const std::array<double, 3>& A, double b,
    const std::array<int, 3>& lmn2,
    const std::array<double, 3>& B)
{
    /*
        Evaluates overlap integral between two Gaussians
        Returns a double.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int array containing orbital angular momentum (e.g. {1, 0, 0})
              for Gaussian 'a'
        lmn2: int array containing orbital angular momentum for Gaussian 'b'
        A:    array containing origin of Gaussian 'a', e.g. {1.0, 2.0, 0.0}
        B:    array containing origin of Gaussian 'b'
    */

    auto [l1, m1, n1] = lmn1;
    auto [l2, m2, n2] = lmn2;

    double Qx = A[0] - B[0];
    double Qy = A[1] - B[1];
    double Qz = A[2] - B[2];

    double p = a + b;
    double factor = std::pow(M_PI / p, 1.5);

    // Precompute values that do not change during the calculation
    double S1 = expansion_coefficients(l1, l2, 0, Qx, a, b); // X
    double S2 = expansion_coefficients(m1, m2, 0, Qy, a, b); // Y
    double S3 = expansion_coefficients(n1, n2, 0, Qz, a, b); // Z

    return S1 * S2 * S3 * factor;
}

inline const double kinetic_elem(double a, const std::array<int, 3>& lmn1,
    const std::array<double, 3>& A, double b,
    const std::array<int, 3>& lmn2,
    const std::array<double, 3>& B)
{
    auto [l1, m1, n1] = lmn1;
    auto [l2, m2, n2] = lmn2;

    std::array<int, 3> lmn1_2 { l1, m1, n1 };
    std::array<int, 3> lmn2_2 { l2 + 2, m2, n2 };
    std::array<int, 3> lmn2_3 { l2, m2 + 2, n2 };
    std::array<int, 3> lmn2_4 { l2, m2, n2 + 2 };
    std::array<int, 3> lmn2_5 { l2 - 2, m2, n2 };
    std::array<int, 3> lmn2_6 { l2, m2 - 2, n2 };
    std::array<int, 3> lmn2_7 { l2, m2, n2 - 2 };

    auto term0 = b * (2 * (l2 + m2 + n2) + 3) * overlap_elem(a, lmn1, A, b, lmn2, B);
    auto term1 = -2 * pow(b, 2) * (overlap_elem(a, lmn1_2, A, b, lmn2_2, B) + overlap_elem(a, lmn1_2, A, b, lmn2_3, B) + overlap_elem(a, lmn1_2, A, b, lmn2_4, B));
    auto term2 = -0.5 * (l2 * (l2 - 1) * overlap_elem(a, lmn1_2, A, b, lmn2_5, B) + m2 * (m2 - 1) * overlap_elem(a, lmn1_2, A, b, lmn2_6, B) + n2 * (n2 - 1) * overlap_elem(a, lmn1_2, A, b, lmn2_7, B));
    return term0 + term1 + term2;
}

inline const double boys(int n, double T)
{
    if (T < 1e-8) {
        return 1.0 / (2.0 * n + 1.0);
    }

    double Fm = 0.5 * std::sqrt(M_PI) * std::erf(std::sqrt(T)) / std::sqrt(T); // F_0(T)

    for (int m = 0; m < n; ++m) {
        Fm = (1.0 / (2 * (n - m) + 1)) + (2.0 * T * Fm / (2 * (n - m) + 1));
    }

    return Fm;
}

inline const std::array<double, 3> gaussian_product_center(double a, const std::array<double, 3>& A,
    double b, const std::array<double, 3>& B)
{
    double denom = a + b;
    return {
        (a * A[0] + b * B[0]) / denom,
        (a * A[1] + b * B[1]) / denom,
        (a * A[2] + b * B[2]) / denom
    };
}

// Generate a unique key for the Coulomb auxiliary Hermite integrals based on parameters
inline const std::string generate_coulomb_key(int t, int u, int v, double n, double p, double PCx, double PCy, double PCz, double RPC)
{
    return std::to_string(t) + "_" + std::to_string(u) + "_" + std::to_string(v) + "_" + std::to_string(n) + "_" + std::to_string(p) + "_" + std::to_string(PCx) + "_" + std::to_string(PCy) + "_" + std::to_string(PCz) + "_" + std::to_string(RPC);
}

inline const double coulomb_auxiliary_hermite_integrals(int t, int u, int v, double n, double p, double PCx, double PCy, double PCz, double RPC)
{
    /*
        Returns the Coulomb auxiliary Hermite integrals
        Returns a float.
        Arguments:
        t,u,v:   order of Coulomb Hermite derivative in x,y,z
                 (see defs in Helgaker and Taylor)
        n:       order of Boys function
        PCx,y,z: Cartesian vector distance between Gaussian
                 composite center P and nuclear center C
        RPC:     Distance between P and C
    */
    std::string key = generate_coulomb_key(t, u, v, n, p, PCx, PCy, PCz, RPC);
    if (coulomb_cache.find(key) != coulomb_cache.end()) {
        return coulomb_cache[key];
    }

    auto T = p * RPC * RPC;
    double val = 0.0;
    if (t == 0 && u == 0 && v == 0) {
        val += pow(-2 * p, n) * boys(n, T);
    }
    else if (t == 0 && u == 0) {
        if (v > 1) {
            val += (v - 1) * coulomb_auxiliary_hermite_integrals(t, u, v - 2, n + 1, p, PCx, PCy, PCz, RPC);
        }
        val += PCz * coulomb_auxiliary_hermite_integrals(t, u, v - 1, n + 1, p, PCx, PCy, PCz, RPC);
    }
    else if (t == 0) {
        if (u > 1) {
            val += (u - 1) * coulomb_auxiliary_hermite_integrals(t, u - 2, v, n + 1, p, PCx, PCy, PCz, RPC);
        }
        val += PCy * coulomb_auxiliary_hermite_integrals(t, u - 1, v, n + 1, p, PCx, PCy, PCz, RPC);
    }
    else {
        if (t > 1) {
            val += (t - 1) * coulomb_auxiliary_hermite_integrals(t - 2, u, v, n + 1, p, PCx, PCy, PCz, RPC);
        }
        val += PCx * coulomb_auxiliary_hermite_integrals(t - 1, u, v, n + 1, p, PCx, PCy, PCz, RPC);
    }

    coulomb_cache[key] = val;
    return val;
}

inline const double nuclear_elem(double a, const std::array<int, 3>& lmn1, const std::array<double, 3>& A, double b, const std::array<int, 3>& lmn2, const std::array<double, 3>& B, const std::array<double, 3>& C)
{
    /*
        Evaluates kinetic energy integral between two Gaussians
         Returns a float.
         a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
         b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
         lmn1: int array containing orbital angular momentum (e.g. {1,0,0})
               for Gaussian 'a'
         lmn2: int array containing orbital angular momentum for Gaussian 'b'
         A:    array containing origin of Gaussian 'a', e.g. {1.0, 2.0, 0.0}
         B:    array containing origin of Gaussian 'b'
         C:    array containing origin of nuclear center 'C'
    */
    auto [l1, m1, n1] = lmn1;
    auto [l2, m2, n2] = lmn2;

    auto p = a + b;
    auto P = gaussian_product_center(a, A, b, B);
    auto RPC = std::sqrt(std::pow(P[0] - C[0], 2) + std::pow(P[1] - C[1], 2) + std::pow(P[2] - C[2], 2));

    double val { 0.0 };

    // #pragma omp parallel for reduction(+:val)
    for (int t = 0; t <= l1 + l2; t++) {
        for (int u = 0; u <= m1 + m2; u++) {
            for (int v = 0; v <= n1 + n2; v++) {
                val += expansion_coefficients(l1, l2, t, A[0] - B[0], a, b) * expansion_coefficients(m1, m2, u, A[1] - B[1], a, b) * expansion_coefficients(n1, n2, v, A[2] - B[2], a, b) * coulomb_auxiliary_hermite_integrals(t, u, v, 0.0, p, P[0] - C[0], P[1] - C[1], P[2] - C[2], RPC);
            }
        }
    }
    val *= 2 * M_PI / p;
    return val;
}

inline const double electron_repulsion(double a, const std::array<int, 3>& lmn1, const std::array<double, 3>& A, double b, const std::array<int, 3>& lmn2, const std::array<double, 3>& B, double c, const std::array<int, 3>& lmn3, const std::array<double, 3>& C, double d, const std::array<int, 3>& lmn4, const std::array<double, 3>& D)
{
    auto [l1, m1, n1] = lmn1;
    auto [l2, m2, n2] = lmn2;
    auto [l3, m3, n3] = lmn3;
    auto [l4, m4, n4] = lmn4;

    auto p = a + b;
    auto q = c + d;
    auto alpha = p * q / (p + q);
    auto P = gaussian_product_center(a, A, b, B);
    auto Q = gaussian_product_center(c, C, d, D);
    auto RPQ = std::sqrt(std::pow(P[0] - Q[0], 2) + std::pow(P[1] - Q[1], 2) + std::pow(P[2] - Q[2], 2));

    double val { 0.0 };
    for (int t = 0; t <= l1 + l2; t++) {
        for (int u = 0; u <= m1 + m2; u++) {
            for (int v = 0; v <= n1 + n2; v++) {
                for (int tau = 0; tau <= l3 + l4; tau++) {
                    for (int nu = 0; nu <= m3 + m4; nu++) {
                        for (int phi = 0; phi <= n3 + n4; phi++) {
                            val += expansion_coefficients(l1, l2, t, A[0] - B[0], a, b) * expansion_coefficients(m1, m2, u, A[1] - B[1], a, b) * expansion_coefficients(n1, n2, v, A[2] - B[2], a, b) * expansion_coefficients(l3, l4, tau, C[0] - D[0], c, d) * expansion_coefficients(m3, m4, nu, C[1] - D[1], c, d) * expansion_coefficients(n3, n4, phi, C[2] - D[2], c, d) * std::pow(-1, tau + nu + phi) * coulomb_auxiliary_hermite_integrals(t + tau, u + nu, v + phi, 0, alpha, P[0] - Q[0], P[1] - Q[1], P[2] - Q[2], RPQ);
                        }
                    }
                }
            }
        }
    }

    val *= 2 * std::pow(M_PI, 2.5) / (p * q * std::sqrt(p + q));
    return val;
}
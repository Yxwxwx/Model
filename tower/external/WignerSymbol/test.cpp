#include "WignerSymbol.hpp"
#include <functional>
#include <gsl/gsl_specfunc.h>
#include <iostream>
#include <random>

constexpr double sqrt_2 = 1.41421356237309504880;

using namespace util;

void test_3j()
{
    const int N = 10;
    wigner_init(N, "Jmax", 3);
    double diff = 0;
    for (int dj1 = N; dj1 <= 2 * N; ++dj1)
    {
        for (int dj2 = N; dj2 <= 2 * N; ++dj2)
        {
            for (int dj3 = N; dj3 <= 2 * N; ++dj3)
            {
                for (int dm1 = -dj1; dm1 <= dj1; ++dm1)
                {
                    for (int dm2 = -dj2; dm2 <= dj2; ++dm2)
                    {
                        for (int dm3 = -dj3; dm3 <= dj3; ++dm3)
                        {
                            double x = wigner_3j(dj1, dj2, dj3, dm1, dm2, dm3);
                            double y = gsl_sf_coupling_3j(dj1, dj2, dj3, dm1, dm2, dm3);
                            diff += std::abs(x - y);
                        }
                    }
                }
            }
        }
    }
    std::cout << "test 3j, diff = " << diff << std::endl;
}

void test_CG0()
{
    const int N = 20;
    wigner_init(2 * N, "Jmax", 3);
    double diff = 0;
    for (int j1 = 0; j1 <= N; ++j1)
    {
        for (int j2 = 0; j2 <= N; ++j2)
        {
            for (int j3 = std::abs(j1 - j2); j3 <= j1 + j2; ++j3)
            {
                double x = CG(2 * j1, 2 * j2, 2 * j3, 0, 0, 0);
                double y = CG0(j1, j2, j3);
                diff += std::abs(x - y);
            }
        }
    }
    std::cout << "test CG0, diff = " << diff << std::endl;
}

void test_6j()
{
    const int N = 20;
    wigner_init(N, "Jmax", 6);
    double diff = 0;
    for (int dj1 = N; dj1 <= 2 * N; ++dj1)
    {
        for (int dj2 = N; dj2 <= 2 * N; ++dj2)
        {
            for (int dj3 = N; dj3 <= 2 * N; ++dj3)
            {
                for (int dj4 = N; dj4 <= 2 * N; ++dj4)
                {
                    for (int dj5 = N; dj5 <= 2 * N; ++dj5)
                    {
                        for (int dj6 = N; dj6 <= 2 * N; ++dj6)
                        {
                            double x = wigner_6j(dj1, dj2, dj3, dj4, dj5, dj6);
                            double y = gsl_sf_coupling_6j(dj1, dj2, dj3, dj4, dj5, dj6);
                            diff += std::abs(x - y);
                        }
                    }
                }
            }
        }
    }
    std::cout << "test 6j, diff = " << diff << std::endl;
}

void test_9j()
{
    const int N = 6;
    wigner_init(N, "Jmax", 9);
    double diff = 0;
    for (int dj1 = 0; dj1 <= N; ++dj1)
    {
        for (int dj2 = 0; dj2 <= N; ++dj2)
        {
            for (int dj3 = 0; dj3 <= N; ++dj3)
            {
                for (int dj4 = 0; dj4 <= N; ++dj4)
                {
                    for (int dj5 = 0; dj5 <= N; ++dj5)
                    {
                        for (int dj6 = 0; dj6 <= N; ++dj6)
                        {
                            for (int dj7 = 0; dj7 <= N; ++dj7)
                            {
                                for (int dj8 = 0; dj8 <= N; ++dj8)
                                {
                                    for (int dj9 = 0; dj9 <= N; ++dj9)
                                    {
                                        double x = wigner_9j(dj1, dj2, dj3, dj4, dj5, dj6, dj7, dj8, dj9);
                                        double y = gsl_sf_coupling_9j(dj1, dj2, dj3, dj4, dj5, dj6, dj7, dj8, dj9);
                                        diff += std::abs(x - y);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "test 9j, diff = " << diff << std::endl;
}

struct Moshinsky_case
{
    int N, L, n, l, n1, l1, n2, l2, Lambda;
};

// clang-format off
// Ref: Buck et al. Nuc. Phys. A 600 (1996) 387-402
const Moshinsky_case Moshinsky_test_set[] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0, 0, 1, 1},
    {0, 0, 0, 1, 0, 0, 0, 1, 1},
    {0, 2, 0, 0, 0, 0, 0, 2, 2},
    {0, 1, 0, 1, 0, 0, 0, 2, 2},
    {0, 0, 0, 2, 0, 0, 0, 2, 2},
    {1, 0, 0, 0, 0, 1, 0, 1, 0},
    {0, 1, 0, 1, 0, 1, 0, 1, 0},
    {0, 0, 1, 0, 0, 1, 0, 1, 0},
    {0, 1, 0, 1, 0, 1, 0, 1, 1},
    {0, 2, 0, 0, 0, 1, 0, 1, 2},
    {0, 1, 0, 1, 0, 1, 0, 1, 2},
    {0, 0, 0, 2, 0, 1, 0, 1, 2},
};

// here use x = tan(beta) = b2/b1
const std::function<double(double)> Moshinsky_test_set_result[] = {
    [](double x){ return 1.; },
    [](double x){ return 1./std::sqrt(1. + x*x); },
    [](double x){ return -x/std::sqrt(1. + x*x); },
    [](double x){ return 1./(1. + x*x); },
    [](double x){ return -sqrt_2*x/(1 + x*x); },
    [](double x){ return x*x/(1 + x*x); },
    [](double x){ return sqrt_2*x/(1 + x*x); },
    [](double x){ return (1 - x*x)/(1 + x*x); },
    [](double x){ return -sqrt_2*x/(1 + x*x); },
    [](double x){ return -1.; },
    [](double x){ return sqrt_2*x/(1 + x*x); },
    [](double x){ return (1 - x*x)/(1 + x*x); },
    [](double x){ return -sqrt_2*x/(1 + x*x); },
};
// clang-format on

void test_Moshinsky()
{
    wigner_init(2, "Moshinsky", 0);
    constexpr double tan_betas[] = {1. / 3., 0.5, 1., 2., 3.};
    double diff = 0.;
    for (double tan_beta : tan_betas)
    {
        for (int idx = 6; idx < 13; ++idx)
        {
            Moshinsky_case m = Moshinsky_test_set[idx];
            auto exact_func = Moshinsky_test_set_result[idx];
            double x = Moshinsky(m.N, m.L, m.n, m.l, m.n1, m.l1, m.n2, m.l2, m.Lambda, tan_beta);
            double y = exact_func(tan_beta);
            diff += std::abs(x - y);
            if (std::abs(x - y) > 1e-12)
            {
                std::cout << "idx = " << idx << ", tan_beta = " << tan_beta << ": ";
                std::cout << "Moshinsky = " << x << ", exact = " << y << std::endl;
            }
        }
    }
    std::cout << "test Moshinsky, diff = " << diff << std::endl;
}

void test_CGspin()
{
    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dist(-1, 1); // has invalid input
    constexpr int N = 100'000;
    bool has_error = false;
    for (int i = 0; i < N; ++i)
    {
        int dm1 = dist(gen);
        int dm2 = dist(gen);
        int S = dist(gen);
        double x = CGspin(dm1, dm2, S);
        double y = CG(1, 1, 2 * S, dm1, dm2, dm1 + dm2);
        if (std::abs(x - y) > 1e-12)
        {
            std::cerr << "dm1 = " << dm1 << ", dm2 = " << dm2 << ", S = " << S << std::endl;
            std::cerr << "CGspin = " << x << ", CG = " << y << std::endl;
            has_error = true;
        }
    }
    if (!has_error)
    {
        std::cout << "test CGspin passed" << std::endl;
    }
    else
    {
        std::cerr << "test CGspin failed" << std::endl;
    }
    for (int i = 0; i < N; ++i)
    {
        int dm1 = dist(gen);
        int dm2 = dist(gen);
        int dm3 = dist(gen);
        int S12 = dist(gen);
        int dS = dist(gen) + 2;
        double x = CG3spin(dm1, dm2, dm3, S12, dS);
        double y = CG(1, 1, 2 * S12, dm1, dm2, dm1 + dm2) * CG(2 * S12, 1, dS, dm1 + dm2, dm3, dm1 + dm2 + dm3);
        if (std::abs(x - y) > 1e-12)
        {
            std::cerr << "dm1 = " << dm1 << ", dm2 = " << dm2 << ", dm3 = " << dm3 << ", S12 = " << S12
                      << ", dS = " << dS << std::endl;
            std::cerr << "CG3spin = " << x << ", CG = " << y << std::endl;
            has_error = true;
        }
    }
    if (!has_error)
    {
        std::cout << "test CG3spin passed" << std::endl;
    }
    else
    {
        std::cerr << "test CG3spin failed" << std::endl;
    }
}

void test_lsjj()
{
    const int Lmax = 20;
    wigner_init(2 * Lmax, "Jmax", 9);
    double lsjj_sum = 0;
    double norm9j_sum = 0;
    for (int l1 = 0; l1 <= Lmax; ++l1)
    {
        for (int l2 = 0; l2 <= Lmax; ++l2)
        {
            std::vector<std::pair<int, int>> dj_pairs = {
                {2 * l1 - 1, 2 * l2 - 1}, {2 * l1 - 1, 2 * l2 + 1}, {2 * l1 + 1, 2 * l2 - 1}, {2 * l1 + 1, 2 * l2 + 1}};
            for (auto [dj1, dj2] : dj_pairs)
            {
                for (int L = std::abs(l1 - l2); L <= l1 + l2; ++L)
                {
                    std::vector<std::pair<int, int>> SJ_pairs = {{0, L}, {1, L - 1}, {1, L}, {1, L + 1}};
                    for (auto [S, J] : SJ_pairs)
                    {
                        double x = lsjj(l1, l2, dj1, dj2, L, S, J);
                        double y = wigner_norm9j(2 * l1, 1, dj1, 2 * l2, 1, dj2, 2 * L, 2 * S, 2 * J);
                        if (std::abs(x - y) > 1e-10)
                        {
                            std::cout << "l1 = " << l1 << ", l2 = " << l2 << ", dj1 = " << dj1 << ", dj2 = " << dj2
                                      << ", L = " << L << ", S = " << S << ", J = " << J << std::endl;
                            std::cout << "lsjj = " << x << ", norm9j = " << y << std::endl;
                        }
                        lsjj_sum += std::abs(x);
                        norm9j_sum += std::abs(y);
                    }
                }
            }
        }
    }
    std::cout << "test lsjj, diff = " << std::abs(lsjj_sum - norm9j_sum) << std::endl;
}

int main(int argc, char const *argv[])
{
    test_3j();
    test_CG0();
    test_6j();
    test_9j();
    test_Moshinsky();
    test_CGspin();
    test_lsjj();
    return 0;
}

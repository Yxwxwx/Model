#include "cmatrix.h"
#include "integral.h"
#include <cint.h>

auto main() -> int
{
    // cint::Integral test;
    // test.calc_spsp();
    cmatrix::CMatrix A(6, 6);
    // 填充矩阵
    A(0, 0) = { -0.52413994, 0.0 };
    A(0, 1) = { -0.51283426, -0.63625368 };
    A(0, 2) = { 0.98599858, 0.58408489 };
    A(0, 3) = { 0.00165023, 0.55444763 };
    A(0, 4) = { 0.04215218, 0.81859376 };
    A(0, 5) = { 0.57161272, 0.09268565 };

    A(1, 0) = { -0.51283426, 0.63625368 };
    A(1, 1) = { 0.29457813, 0.0 };
    A(1, 2) = { 0.79919297, 0.06756653 };
    A(1, 3) = { 0.27996441, 0.87547518 };
    A(1, 4) = { 0.09209668, 0.18279094 };
    A(1, 5) = { -0.40971724, 0.54668111 };

    A(2, 0) = { 0.98599858, -0.58408489 };
    A(2, 1) = { 0.79919297, -0.06756653 };
    A(2, 2) = { 0.04676328, 0.0 };
    A(2, 3) = { 0.65808715, -0.08995854 };
    A(2, 4) = { -0.79550395, 0.80490959 };
    A(2, 5) = { 0.85263385, -0.14781538 };

    A(3, 0) = { 0.00165023, -0.55444763 };
    A(3, 1) = { 0.27996441, -0.87547518 };
    A(3, 2) = { 0.65808715, 0.08995854 };
    A(3, 3) = { -0.37373147, 0.0 };
    A(3, 4) = { -0.86311539, -0.99791398 };
    A(3, 5) = { -0.24112506, 0.36214372 };

    A(4, 0) = { 0.04215218, -0.81859376 };
    A(4, 1) = { 0.09209668, -0.18279094 };
    A(4, 2) = { -0.79550395, -0.80490959 };
    A(4, 3) = { -0.86311539, 0.99791398 };
    A(4, 4) = { 0.44351182, 0.0 };
    A(4, 5) = { 0.38562695, -0.18593323 };

    A(5, 0) = { 0.57161272, -0.09268565 };
    A(5, 1) = { -0.40971724, -0.54668111 };
    A(5, 2) = { 0.85263385, 0.14781538 };
    A(5, 3) = { -0.24112506, -0.36214372 };
    A(5, 4) = { 0.38562695, 0.18593323 };
    A(5, 5) = { -0.98404336, 0.0 };
    // A.print();
    auto eig = cmatrix::CMatrix::eigh(A);
    auto lambda = eig.first;
    // for (auto&& i : lambda) {
    //     std::cout << std::format("{:.5f}", i) << " ";
    // }

    cint::Integral test;
    test.calc_spsp();
    return 0;
}
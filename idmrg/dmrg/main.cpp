#include "src/dmrg.hpp"
#include "src/mpo.hpp"
#include "src/mps.hpp"
#include "src/operators.hpp"
#include "src/toolkit.hpp"
#include "unsupported/Eigen/CXX11/src/Tensor/Tensor.h"
#include <chrono>

constexpr auto MAX_BOND_DIMENSION = 50;
constexpr auto MAX_SWEEPS = 20;
constexpr auto PHY_DIM = 2;
constexpr auto SITE_NUM = 10;
constexpr auto ERROR_THRESHOLD = 1e-7;

const Eigen::Tensor<double, 4> creat_heisenberg_operator()
{
    const auto id = DMRG::Operator::Identity; // Identity operator for 2x2 matrices
    const auto zero = DMRG::Operator::Zero; // Zero operator for 2x2 matrices
    const auto sz = DMRG::Operator::Sz; // Pauli Z operator
    const auto sp = DMRG::Operator::Splus; // Pauli S+ operator
    const auto sm = DMRG::Operator::Sminus; // Pauli S- operator

    Eigen::Tensor<double, 4> single_mpo(5, 5, 2, 2);
    single_mpo.setZero();
    DMRG::Toolkit::assign_block<double>(single_mpo, id, 0, 0);
    DMRG::Toolkit::assign_block<double>(single_mpo, sp, 1, 0);
    DMRG::Toolkit::assign_block<double>(single_mpo, sm, 2, 0);
    DMRG::Toolkit::assign_block<double>(single_mpo, sz, 3, 0);
    DMRG::Toolkit::assign_block<double>(single_mpo, 0.5 * sm, 4, 1);
    DMRG::Toolkit::assign_block<double>(single_mpo, 0.5 * sp, 4, 2);
    DMRG::Toolkit::assign_block<double>(single_mpo, sz, 4, 3);
    DMRG::Toolkit::assign_block<double>(single_mpo, id, 4, 4);
    return single_mpo;
}
int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    auto single_mpo = creat_heisenberg_operator();
    DMRG::MPO<> mpo(single_mpo, SITE_NUM);
    DMRG::MPS<> mps(PHY_DIM, MAX_BOND_DIMENSION, SITE_NUM);
    DMRG::DMRG<> dmrg(mpo.mpo_list_, mps.mps_list_, MAX_BOND_DIMENSION, MAX_SWEEPS, ERROR_THRESHOLD);

    auto E = dmrg.kernel();

    fmt::println("total time: {: .3f}s", (std::chrono::high_resolution_clock::now() - start).count() / 1e9);

    // // allocate tensors
    // // Example dimensions for a 6D tensor
    // constexpr int dim = 6;
    // int perm[dim] = { 5, 2, 0, 4, 1, 3 };
    // int size[dim] = { 48, 28, 48, 28, 28, 1 };
    // double* A = new double[48 * 28 * 48 * 28 * 28 * 1];

    // // create a plan (pointer)
    // auto plan = hptt::create_plan(perm, dim, 1.0, A, size, NULL, 0.0, A, NULL, hptt::ESTIMATE, 16);

    // // execute the transposition
    // plan->execute();
    return 0;
}
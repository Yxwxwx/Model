#pragma once
#ifndef DMRG_HPP
#define DMRG_HPP
#include "mpo.hpp"
#include "mps.hpp"
#include "toolkit.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <hptt.h>
#include <optional>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/CXX11/ThreadPool>
namespace DMRG {

template <typename T = double>
inline std::tuple<Eigen::Tensor<T, 2>, Eigen::Tensor<T, 1>, Eigen::Tensor<T, 2>>
svd_compress(const Eigen::Tensor<T, 3>& tensor, const std::string& direction,
    const int maxM)
{
    const auto [left, phy_dim, right] = std::make_tuple(tensor.dimension(0), tensor.dimension(1), tensor.dimension(2));

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat;
    if (direction == "left" || direction == "l") {
        mat = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(tensor.data(), left * phy_dim, right);
    }
    else if (direction == "right" || direction == "r") {
        mat = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(tensor.data(), left, phy_dim * right);
    }
    else {
        throw std::invalid_argument("Invalid direction: " + direction);
    }

    Eigen::JacobiSVD<Eigen::MatrixX<T>> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);

    const int actualM = std::min(static_cast<int>(svd.singularValues().size()), maxM);

    // matrix to tensor
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrixU = svd.matrixU().leftCols(actualM);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrixV = svd.matrixV().leftCols(actualM).transpose();

    Eigen::Tensor<T, 2> U = Eigen::TensorMap<Eigen::Tensor<T, 2>>(
        matrixU.data(),
        Eigen::array<Eigen::Index, 2> { (direction == "left") ? left * phy_dim : left, actualM });

    Eigen::Tensor<T, 1> S(actualM);
    S = Eigen::TensorMap<Eigen::Tensor<const T, 1>>(svd.singularValues().data(), actualM);
    Eigen::Tensor<T, 2> V = Eigen::TensorMap<Eigen::Tensor<T, 2>>(
        matrixV.data(),
        Eigen::array<Eigen::Index, 2> { actualM, (direction == "left") ? right : phy_dim * right });
    return { U, S, V };
}

template <typename T = double>
struct DMRG {
    int max_bond_dim_ = 0; // Maximum bond dimension
    int max_sweeps_ = 20; // Maximum number of sweeps
    double error_threshold_ = 1e-7; // Error threshold for convergence
    int site_num_ = 0; // Total number of sites
    std::vector<Eigen::Tensor<T, 3>> mps_list_; // List of MPS tensors
    std::vector<Eigen::Tensor<T, 4>> mpo_list_; // List of MPO tensors
    std::vector<std::optional<Eigen::Tensor<T, 6>>> F_list_; // List of F tensors for DMRG
    std::vector<std::optional<Eigen::Tensor<T, 6>>> L_list_; // List of L tensors for DMRG
    std::vector<std::optional<Eigen::Tensor<T, 6>>> R_list_; // List of R tensors for DMRG

    Eigen::ThreadPool pool;
    Eigen::ThreadPoolDevice my_device;

    DMRG() = delete;
    DMRG(const std::vector<Eigen::Tensor<T, 4>>& mpo, const std::vector<Eigen::Tensor<T, 3>>& mps, const int max_bond_dim = 0, const int max_sweeps = 20, const double error_threshold = 1e-7);
    inline void update_local_site(const int idx, const Eigen::Tensor<T, 3>& newState);
    inline void left_canonicalize_at(const int idx);
    inline void left_canonicalize_from(const int idx);
    inline void right_canonicalize_at(const int idx);
    inline void right_canonicalize_from(const int idx);
    inline Eigen::Tensor<T, 6> tensorF_at(const int idx);
    inline Eigen::Tensor<T, 6> tensorL_at(const int idx);
    inline Eigen::Tensor<T, 6> tensorR_at(const int idx);
    inline Eigen::Tensor<T, 6> variational_tensor_at(const int idx);
    inline double sweep_at(const int idx, const std::string& direction);
    double kernel();
}; // struct DMRG

template <typename T>
DMRG<T>::DMRG(const std::vector<Eigen::Tensor<T, 4>>& mpo,
    const std::vector<Eigen::Tensor<T, 3>>& mps,
    int max_bond_dim,
    int max_sweeps,
    double error_threshold)
    : max_bond_dim_(max_bond_dim), max_sweeps_(max_sweeps), error_threshold_(error_threshold), pool(std::thread::hardware_concurrency()), my_device(&pool, pool.NumThreads() /* number of threads to use */)
{
    // 参数校验
    assert(max_bond_dim_ > 0);
    assert(!mpo.empty() && mpo.size() == mps.size());
    site_num_ = mpo.size();

    auto create_boundary_mpo = [] {
        Eigen::Tensor<T, 4> boundary(1, 1, 1, 1);
        boundary.setZero();
        return boundary;
    };

    auto create_boundary_mps = [] {
        Eigen::Tensor<T, 3> boundary(1, 1, 1);
        boundary.setZero();
        return boundary;
    };
    auto create_boundary_F = [] {
        Eigen::Tensor<T, 6> boundary(1, 1, 1, 1, 1, 1);
        boundary.setConstant(1.0);
        return boundary;
    };

    mpo_list_.reserve(site_num_ + 2);
    mpo_list_.emplace_back(create_boundary_mpo());
    mpo_list_.insert(mpo_list_.end(), mpo.begin(), mpo.end());
    mpo_list_.emplace_back(create_boundary_mpo());

    mps_list_.reserve(site_num_ + 2);
    mps_list_.emplace_back(create_boundary_mps());
    mps_list_.insert(mps_list_.end(), mps.begin(), mps.end());
    mps_list_.emplace_back(create_boundary_mps());

    F_list_.reserve(site_num_ + 2);
    F_list_.emplace_back(create_boundary_F());
    F_list_.insert(F_list_.end(), site_num_, std::nullopt);
    F_list_.emplace_back(create_boundary_F());

    L_list_ = F_list_;
    R_list_ = F_list_;

    right_canonicalize_from(site_num_);
}

template <typename T>
inline void DMRG<T>::update_local_site(int idx, const Eigen::Tensor<T, 3>& newState)
{
    mps_list_[idx] = newState;
    F_list_[idx].reset();

    for (int i = idx + 1; i <= site_num_; ++i) {
        if (L_list_[i].has_value())
            L_list_[i].reset();
        else
            break;
    }
    for (int i = idx - 1; i >= 0; --i) {
        if (R_list_[i].has_value())
            R_list_[i].reset();
        else
            break;
    }
}

// #define likely(x) __builtin_expect(!!(x), 1)
// #define unlikely(x) __builtin_expect(!!(x), 0)

// head dummy site: idx == 0
// real sites : idx == 1~size
// last dummy site: idx == size+1
template <typename T>
inline void DMRG<T>::left_canonicalize_at(const int idx)
{
    // Early return for boundary cases (head dummy site: idx == 0)
    if (idx >= site_num_)
        return;

    // Extract current site tensor dimensions
    const auto site = mps_list_[idx];
    const auto [left, phy_dim, right] = std::make_tuple(
        site.dimension(0), site.dimension(1), site.dimension(2));

    // Perform SVD compression (truncate to max bond dimension)
    const auto [u, s, v] = svd_compress<T>(site, "left", max_bond_dim_);

    // Update current site with reshaped U tensor
    update_local_site(idx, u.reshape(Eigen::array<Eigen::Index, 3> { left, phy_dim, s.dimension(0) }));

    // Construct diagonal matrix from singular values
    Eigen::Tensor<T, 2> s_diag(s.dimension(0), s.dimension(0));
    s_diag.setZero();
    for (int i = 0; i < s.dimension(0); ++i) {
        s_diag(i, i) = s(i);
    }

    // Combine S*V and contract with next site tensor (equivalent to tensordot)

    Eigen::Tensor<T, 2> s_diag_v(s_diag.dimension(0), v.dimension(1));
    s_diag_v.device(my_device) = s_diag.contract(v, Eigen::array<Eigen::IndexPair<int>, 1> { Eigen::IndexPair<int>(1, 0) });

    Eigen::Tensor<T, 3> newState(s_diag_v.dimension(0), mps_list_[idx + 1].dimension(1), mps_list_[idx + 1].dimension(2));
    newState.device(my_device) = s_diag_v.contract(mps_list_[idx + 1], Eigen::array<Eigen::IndexPair<int>, 1> { Eigen::IndexPair<int>(1, 0) });
    update_local_site(idx + 1, newState);
}

template <typename T>
inline void DMRG<T>::left_canonicalize_from(const int idx)
{
    for (int i = idx; i < site_num_; ++i) {
        left_canonicalize_at(i);
    }
}

template <typename T>
inline void DMRG<T>::right_canonicalize_at(const int idx)
{
    if (idx <= 1)
        return;

    const auto site = mps_list_[idx];
    const auto [left, phy_dim, right] = std::make_tuple(
        site.dimension(0), site.dimension(1), site.dimension(2));

    const auto [u, s, v] = svd_compress<T>(site, "right", max_bond_dim_);
    update_local_site(idx, v.reshape(Eigen::array<Eigen::Index, 3> { s.dimension(0), phy_dim, right }));
    // Construct diagonal matrix from singular values
    Eigen::Tensor<T, 2> s_diag(s.dimension(0), s.dimension(0));
    s_diag.setZero();
    for (int i = 0; i < s.dimension(0); ++i) {
        s_diag(i, i) = s(i);
    }

    Eigen::Tensor<T, 2> u_s_diag(u.dimension(0), s_diag.dimension(1));
    u_s_diag.device(my_device) = u.contract(s_diag, Eigen::array<Eigen::IndexPair<int>, 1> { Eigen::IndexPair<int>(1, 0) });

    Eigen::Tensor<T, 3> newState(mps_list_[idx - 1].dimension(0), mps_list_[idx - 1].dimension(1), u_s_diag.dimension(1));
    newState.device(my_device) = mps_list_[idx - 1].contract(u_s_diag, Eigen::array<Eigen::IndexPair<int>, 1> { Eigen::IndexPair<int>(2, 0) });

    update_local_site(idx - 1, newState);
}

template <typename T>
inline void DMRG<T>::right_canonicalize_from(const int idx)
{
    for (int i = idx; i > 1; --i) {
        right_canonicalize_at(i);
    }
}

template <typename T>
inline Eigen::Tensor<T, 6> DMRG<T>::tensorF_at(const int idx)
{
    /*
    calculate F for this site.
            graphical representation (* for MPS and # for MPO,
            numbers represents a set of imaginary bond dimensions used for comments below):
                                      1 --*-- 5
                                          | 4
                                      2 --#-- 3
                                          | 4
                                      1 --*-- 5
            :return the calculated F
    */
    if (!F_list_[idx].has_value()) {
        // compute tensor F for idx
        const auto site = mps_list_[idx];
        const auto op = mpo_list_[idx];
        // site is (1,4,5)
        // operator is (2,3,4,4)
        // compute <site|operator
        // contract 4, F is (1,5,2,3,4)
        Eigen::Tensor<T, 5> tmp(site.dimension(0), site.dimension(2), op.dimension(0), op.dimension(1), op.dimension(3));
        tmp.device(my_device) = site.conjugate().contract(op, Eigen::array<Eigen::IndexPair<int>, 1> { Eigen::IndexPair<int>(1, 2) });
        // compute <site|operator|site>
        // contract 4
        Eigen::Tensor<T, 6> F(tmp.dimension(0), tmp.dimension(1), tmp.dimension(2), tmp.dimension(3), site.dimension(0), site.dimension(2));
        F.device(my_device) = tmp.contract(site, Eigen::array<Eigen::IndexPair<int>, 1> { Eigen::IndexPair<int>(4, 1) });
        // F is (1,5,2,3,1,5)
        F_list_[idx] = std::move(F);
    }
    return F_list_[idx].value();
}

template <typename T>
inline Eigen::Tensor<T, 6> DMRG<T>::tensorL_at(const int idx)
{
    /*
    calculate L in a recursive way
    */
    if (!L_list_[idx].has_value()) {
        if (idx <= 1)
            L_list_[idx] = tensorF_at(idx);
        else {
            auto leftL = tensorL_at(idx - 1);
            auto currentF = tensorF_at(idx);
            /*
            do the contraction.
            graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
              0 --*-- 1          0 --*-- 1                   0 --*-- 2                     0 --*-- 1
                  |                  |                           |                            |
              2 --#-- 3     +    2 --#-- 3  --tensordot-->   4 --#-- 1    --transpose-->   2 --#-- 3
                  |                  |                           |                            |
              4 --*-- 5          4 --*-- 5                   3 --*-- 5                     4 --*-- 5

            */

            // tensordot (0,2,4,1,3,5) -transpose-> (0,1,2,3,4,5)
            Eigen::Tensor<T, 6> currentL(leftL.dimension(0), leftL.dimension(2), leftL.dimension(4), currentF.dimension(1), currentF.dimension(3), currentF.dimension(5));
            currentL.device(my_device) = leftL
                                             .contract(currentF,
                                                 Eigen::array<Eigen::IndexPair<int>, 3> {
                                                     Eigen::IndexPair<int>(1, 0),
                                                     Eigen::IndexPair<int>(3, 2),
                                                     Eigen::IndexPair<int>(5, 4) });
            Eigen::Tensor<T, 6> L = currentL.shuffle(Eigen::array<int, 6> { 0, 3, 1, 4, 2, 5 });
            L_list_[idx] = std::move(L);
        }
    }
    return L_list_[idx].value();
}

template <typename T>
inline Eigen::Tensor<T, 6> DMRG<T>::tensorR_at(const int idx)
{
    /*
    calculate R in a recursive way
    */
    if (!R_list_[idx].has_value()) {
        if (idx >= site_num_)
            R_list_[idx] = tensorF_at(idx);
        else {
            auto rightR = tensorR_at(idx + 1);
            auto currentF = tensorF_at(idx);
            Eigen::Tensor<T, 6> currentR(currentF.dimension(0), currentF.dimension(2), currentF.dimension(4), rightR.dimension(1), rightR.dimension(3), rightR.dimension(5));
            currentR.device(my_device) = currentF
                                             .contract(rightR,
                                                 Eigen::array<Eigen::IndexPair<int>, 3> {
                                                     Eigen::IndexPair<int>(1, 0),
                                                     Eigen::IndexPair<int>(3, 2),
                                                     Eigen::IndexPair<int>(5, 4) });
            int perm[6] = { 0, 3, 1, 4, 2, 5 };
            int size[6] = { currentR.dimension(0), currentR.dimension(1), currentR.dimension(2), currentR.dimension(3), currentR.dimension(4), currentR.dimension(5) };
            // Eigen::Tensor<T, 6> R = currentR.shuffle(Eigen::array<int, 6> { 0, 3, 1, 4, 2, 5 });
            Eigen::Tensor<T, 6> R = Eigen::Tensor<T, 6>(currentR.dimension(0), currentR.dimension(3), currentR.dimension(1), currentR.dimension(4), currentR.dimension(2), currentR.dimension(5));
            auto plan = hptt::create_plan(perm, 6, 1.0, currentR.data(), size, NULL, 0.0, R.data(), NULL, hptt::ESTIMATE, 1);
            plan->execute();
            R_list_[idx] = std::move(R);
        }
    }
    return R_list_[idx].value();
}

template <typename T>
inline Eigen::Tensor<T, 6> DMRG<T>::variational_tensor_at(const int idx)
{
    /*
    calculate the variational tensor for the ground state search. L * MPO * R
    graphical representation (* for MPS and # for MPO):
                                   --*--     --*--
                                     |         |
                                   --#----#----#--
                                     |         |
                                   --*--     --*--
                                     L   MPO   R
    */
    const auto [left, phy_dim, right] = std::make_tuple(
        mps_list_[idx].dimension(0), mps_list_[idx].dimension(1), mps_list_[idx].dimension(2));
    const auto op = mpo_list_[idx];
    /*
    do the contraction for L and MPO
    graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
      0 --*-- 1                                    0 --*-- 1
          |                | 2                         |    | 6
      2 --#-- 3    +   0 --#-- 1  --tensordot-->   2 --#----#-- 5
          |                | 3                         |    | 7
      4 --*-- 5                                    3 --*-- 4
          L                MPO                       left_middle
    */
    auto L = tensorL_at(idx - 1);
    Eigen::Tensor<T, 8> left_middle(L.dimension(0), L.dimension(1), L.dimension(2), L.dimension(4), L.dimension(5), op.dimension(1), op.dimension(2), op.dimension(3));
    left_middle.device(my_device) = L.contract(op, Eigen::array<Eigen::IndexPair<int>, 1> { Eigen::IndexPair<int>(3, 0) });
    /*
    do the contraction for L and MPO
    graphical representation (* for MPS and # for MPO, numbers represents the index of the degree in tensor.shape):
      0 --*-- 1             0 --*-- 1                   0 --*-- 1 8 --*-- 9
          |    | 6              |                           |    | 6  |
      2 --#----#-- 5   +    2 --#-- 3  --tensordot-->   2 --#----#----#-- 10
          |    | 7              |                           |    | 7  |
      3 --*-- 4             4 --*-- 5                   3 --*-- 4 11--*-- 12
        left_middle             R                       raw variational tensor
    Note the dimension of 0, 2, 3, 9, 10, 12 are all 1, so the dimension could be reduced
    */
    auto R = tensorR_at(idx + 1);
    Eigen::Tensor<T, 12> raw_variational_tensor(
        left_middle.dimension(0), left_middle.dimension(1), left_middle.dimension(2), left_middle.dimension(3), left_middle.dimension(4), left_middle.dimension(6), left_middle.dimension(7),
        R.dimension(0), R.dimension(1), R.dimension(3), R.dimension(4), R.dimension(5));
    raw_variational_tensor.device(my_device) = left_middle.contract(R, Eigen::array<Eigen::IndexPair<int>, 1> { Eigen::IndexPair<int>(5, 2) });
    Eigen::Tensor<T, 6> result = raw_variational_tensor
                                     .reshape(Eigen::array<Eigen::Index, 6> { left, left, phy_dim, phy_dim, right, right })
                                     .shuffle(Eigen::array<int, 6> { 0, 2, 4, 1, 3, 5 });
    return result;
}

template <typename T>
inline double DMRG<T>::sweep_at(const int idx, const std::string& direction)
{
    /*
    DMRG sweep
    */
    const auto [left, phy_dim, right] = std::make_tuple(
        mps_list_[idx].dimension(0), mps_list_[idx].dimension(1), mps_list_[idx].dimension(2));
    const auto localDimension = left * phy_dim * right;
    // reshape the variational tensor to a square matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> variationalTensor = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(
        variational_tensor_at(idx).data(), localDimension, localDimension);
    // solve for eigen values and vectors
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> solver(variationalTensor);
    // update current site
    Eigen::Tensor<T, 3> newState = Eigen::TensorMap<const Eigen::Tensor<T, 3>>(
        solver.eigenvectors().col(0).data(), Eigen::array<Eigen::Index, 3> { left, phy_dim, right });
    update_local_site(idx, newState);

    // normalization
    if ((direction == "left") || (direction == "l"))
        left_canonicalize_at(idx);
    else
        right_canonicalize_at(idx);

    return solver.eigenvalues()[0]; // ground state energy
}

template <typename T>
double DMRG<T>::kernel()
{
    /*
    The main kernel for DMRG algorithm.
    */
    auto E_old { 0.0 };
    auto E_new { 0.0 };
    fmt::println("Max bond dimension:{}", max_bond_dim_);
    fmt::println("Max sweeps:{}", max_sweeps_);
    fmt::println("Error threshold:{}", error_threshold_);
    fmt::println("Site number:{}", site_num_);
    fmt::println("********* naive DMRG for spin model *********");
    for (int sweep = 0; sweep < max_sweeps_; ++sweep) {
        fmt::println("DMRG sweep:{}/{}", sweep + 1, max_sweeps_);
        fmt::println(">>>>>>>>>> sweep from left to right >>>>>>>>>>");
        // left -> right sweep
        for (int idx = 1; idx < site_num_ + 1; ++idx) {
            E_new = sweep_at(idx, "left");
            fmt::println("Left sweep at site {}, energy: {:.6f}", idx, E_new);
        }

        // right -> left sweep
        fmt::println(">>>>>>>>>> sweep from right to left >>>>>>>>>>");
        for (int idx = site_num_; idx > 0; --idx) {
            E_new = sweep_at(idx, "right");
            fmt::println("Right sweep at site {}, energy: {:.6f}", idx, E_new);
        }

        // check convergence
        if (std::abs(E_new - E_old) < error_threshold_) {
            fmt::println("DMRG converged at sweep {} with energy: {:.6f}", sweep + 1, E_new);
            return E_new;
        }
        else {
            E_old = E_new;
        }
    }
    fmt::println("DMRG did not converge after {} sweeps", max_sweeps_);
    fmt::println("Final energy: {:.6f}", E_new);
    return E_new;
}
} // namespace DMRG
#endif // DMRG_HPP
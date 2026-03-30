#pragma once
#ifndef MPS_HPP
#define MPS_HPP

#include <unsupported/Eigen/CXX11/Tensor>

namespace DMRG {
template <typename T = double>
struct MPS {
    /*
    build MPS list for DMRG.
    local matrix product state tensor
    bond order: [left, physical, right]
                    1
                    |
                0 --*-- 2
    */
    MPS() = delete;
    MPS(const int phy_dim, const int bond_dim, const int site_num);
    std::vector<Eigen::Tensor<T, 3>> mps_list_; // List of MPS tensors
    int site_num_ = 0; // Total number of sites
}; // class MPS

template <typename T>
MPS<T>::MPS(const int phy_dim, const int bond_dim, const int site_num)
    : site_num_(site_num)

{
    mps_list_.reserve(site_num_);
    mps_list_.emplace_back(
        [](int phy, int bond) {
            Eigen::Tensor<T, 3> t(1, phy, bond);
            t.setRandom();
            return t;
        }(phy_dim, bond_dim));

    auto gen_mid = [phy_dim, bond_dim] {
        Eigen::Tensor<T, 3> t(bond_dim, phy_dim, bond_dim);
        t.setRandom();
        return t;
    };
    mps_list_.insert(mps_list_.end(), site_num_ - 2, gen_mid());
    mps_list_.emplace_back(
        [](int phy, int bond) {
            Eigen::Tensor<T, 3> t(bond, phy, 1);
            t.setRandom();
            return t;
        }(phy_dim, bond_dim));
}
} // namespace DMRG

#endif // MPS_HPP
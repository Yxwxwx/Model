#pragma once
#ifndef OPERATOR_HPP
#define OPERATOR_HPP
#include <unsupported/Eigen/CXX11/Tensor>

namespace DMRG::Operator {

// Identity operator for 2x2 matrices
inline const Eigen::Tensor<double, 2> Identity = [] {
    Eigen::Tensor<double, 2> id(2, 2);
    id.setValues({ { 1.0, 0.0 }, { 0.0, 1.0 } });
    return id;
}();

// Zero operator for 2x2 matrices
inline const Eigen::Tensor<double, 2> Zero = [] {
    Eigen::Tensor<double, 2> zero(2, 2);
    zero.setZero();
    return zero;
}();

// Pauli operators for 2x2 matrices
inline const Eigen::Tensor<double, 2> Sz = [] {
    Eigen::Tensor<double, 2> sz(2, 2);
    sz.setValues({ { 0.5, 0.0 }, { 0.0, -0.5 } });
    return sz;
}();

inline const Eigen::Tensor<double, 2> Splus = [] {
    Eigen::Tensor<double, 2> sp(2, 2);
    sp.setValues({ { 0.0, 0.0 }, { 1.0, 0.0 } });
    return sp;
}();

inline const Eigen::Tensor<double, 2> Sminus = [] {
    Eigen::Tensor<double, 2> sm(2, 2);
    sm.setValues({ { 0.0, 1.0 }, { 0.0, 0.0 } });
    return sm;
}();

} // namespace DMRG::Operator
#endif // OPERATOR_HPP

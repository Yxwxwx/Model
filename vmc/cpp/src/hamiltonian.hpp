#pragma once
#include <cmath>
#include <numeric>
#include <span>
#include <vector>
#include "model.hpp"

template <typename T = float>
struct TFIM {
  T J, h;
  std::size_t L;

  TFIM(std::size_t L, T J, T h) : L(L), J(J), h(h) {}

  // 接受一个可变的 buffer 避免重复分配
  T local_energy(const RBM<T>& model, std::span<const T> sigma,
                 std::vector<T>& buffer) const {
    // 1. Interaction (Diagonal)
    T interaction = 0;
    for (std::size_t i = 0; i < L - 1; ++i) {
      interaction -= J * sigma[i] * sigma[i + 1];
    }

    // 2. Transverse Field (Off-diagonal)
    T field_sum = 0;
    T log_psi_current = model(sigma);

    // 使用传入的 buffer 进行原地操作
    std::copy(sigma.begin(), sigma.end(), buffer.begin());

    for (std::size_t i = 0; i < L; ++i) {
      buffer[i] *= -1.0;  // flip
      T log_psi_flipped = model(std::span<const T>(buffer));
      field_sum += std::exp(log_psi_flipped - log_psi_current);
      buffer[i] *= -1.0;  // flip back
    }

    return interaction - h * field_sum;
  }
};
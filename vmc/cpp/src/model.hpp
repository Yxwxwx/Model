#pragma once

#include <algorithm>
#include <cmath>
#include <mdspan/mdspan.hpp>  // 使用 Kokkos 或 std 参考实现
#include <memory>
#include <numbers>
#include <random>
#include <span>

#include "parameter.hpp"

// 缩写别名，增加可读性
namespace md = Kokkos;

template <typename T = float>
struct RBM {
  /**
   * log \Psi(\sigma) = a · \sigma + \sum_j log(2 cosh(b_j + W_j · \sigma))
   */

  std::size_t n_visible, n_hidden;

  // 1. 资源持有者 (Parameter 负责 RAII 内存管理)
  Parameter<T> v_bias;
  Parameter<T> h_bias;
  Parameter<T> weights;

  // 2. 视图 (mdspan 提供多维索引接口)
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent>> v_view;
  md::mdspan<T, md::extents<std::size_t, md::dynamic_extent>> h_view;
  md::mdspan<T,
             md::extents<std::size_t, md::dynamic_extent, md::dynamic_extent>>
      w_view;

  RBM(std::size_t n_v, std::size_t n_h, std::mt19937& rng,
      std::shared_ptr<VectorAllocator<T>> alloc)
      : n_visible(n_v),
        n_hidden(n_h),
        v_bias(alloc, n_v),
        h_bias(alloc, n_h),
        weights(alloc, n_h * n_v),
        v_view(v_bias.ptr, n_v),
        h_view(h_bias.ptr, n_h),
        w_view(weights.ptr, n_h, n_v) {
    // 初始化参数
    std::fill_n(v_bias.ptr, n_visible, 0.0f);
    std::fill_n(h_bias.ptr, n_hidden, 0.0f);

    std::normal_distribution<T> dist(0.0f, 0.01f);
    for (std::size_t i = 0; i < n_hidden * n_visible; ++i) {
      weights.ptr[i] = dist(rng);
    }
  }

  ~RBM() = default;

  // 核心计算逻辑
  T operator()(std::span<const T> sigma) const {
    // 1. Visible term: a · σ
    T visible_term = 0;
    for (std::size_t i = 0; i < n_visible; ++i) {
      visible_term += sigma[i] * v_view[i];
    }

    // 2. Hidden term: \sum_j log(2 cosh(pre_j))
    T hidden_term_sum = 0;
    for (std::size_t j = 0; j < n_hidden; ++j) {
      T pre_j = h_view[j];
      for (std::size_t i = 0; i < n_visible; ++i) {
        pre_j += w_view[j, i] * sigma[i];
      }
      hidden_term_sum += logaddexp(pre_j, -pre_j);
    }

    return visible_term + hidden_term_sum;
  }

  static T logaddexp(T x, T y) {
    if (x == y) return x + std::numbers::ln2_v<T>;
    T max_val = std::max(x, y);
    T min_val = std::min(x, y);
    return max_val + std::log1p(std::exp(min_val - max_val));
  }
};
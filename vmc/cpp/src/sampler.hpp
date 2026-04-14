#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <span>
#include <vector>

#include "hamiltonian.hpp"
#include "model.hpp"

// 用于返回采样结果的结构体
template <typename T = float>
struct SamplerResult {
  std::vector<T> energies;
  T accept_rate;
};

template <typename T = float>
class MetropolisSampler {
 public:
  /**
   * 构造函数
   * @param rbm 神经网络模型
   * @param ham 哈密顿量计算器
   * @param rng 随机数引擎 (必须传入引用，保证每条链状态独立)
   * @param init_state 初始构型
   */
  MetropolisSampler(const RBM<T>& rbm, const TFIM<T>& ham, std::mt19937& rng,
                    std::span<const T> init_state)
      : model(rbm),
        hamiltonian(ham),
        rng(rng),
        L(init_state.size()),
        n_sweeps(init_state.size()),  // 默认 n_sweeps = L
        sigma(init_state.begin(), init_state.end()),
        buffer(init_state.size())  // 预分配 buffer 供 Hamiltonian 使用
  {
    // 初始化计算一次当前态的 log_psi
    current_log_psi = model(sigma);
  }

  void set_sweeps(std::size_t sweeps) { n_sweeps = sweeps; }

  // --- 1. Burn-in 期 ---
  void burn_in(std::size_t steps) {
    for (std::size_t i = 0; i < steps; ++i) {
      sweep_step();
    }
  }

  // --- 2. 正式采样期 ---
  SamplerResult<T> sample(std::size_t n_samples) {
    SamplerResult<T> result;
    result.energies.reserve(n_samples);

    std::size_t total_accepts = 0;
    std::size_t total_flips = n_samples * n_sweeps;

    for (std::size_t i = 0; i < n_samples; ++i) {
      // 走过 n_sweeps 步
      total_accepts += sweep_step();

      // 计算局部能量 E_loc (传入 buffer 避免内部分配内存)
      T e_loc = hamiltonian.local_energy(model, sigma, buffer);
      result.energies.push_back(e_loc);
    }

    result.accept_rate =
        static_cast<T>(total_accepts) / static_cast<T>(total_flips);
    return result;
  }

  // 获取最后的构型
  const std::vector<T>& get_final_state() const { return sigma; }

 private:
  const RBM<T>& model;
  const TFIM<T>& hamiltonian;
  std::mt19937& rng;
  std::size_t L;
  std::size_t n_sweeps;

  // 状态缓存 (核心)
  std::vector<T> sigma;
  std::vector<T> buffer;
  T current_log_psi;

  // --- 核心原子操作：执行连续的 n_sweeps 次翻转 ---
  std::size_t sweep_step() {
    std::size_t accepts = 0;
    std::uniform_int_distribution<std::size_t> dist_site(0, L - 1);
    std::uniform_real_distribution<T> dist_acc(0.0, 1.0);

    for (std::size_t i = 0; i < n_sweeps; ++i) {
      std::size_t site = dist_site(rng);

      // 1. Propose: 原位翻转！极其高效
      sigma[site] *= -1.0f;
      T log_psi_proposal = model(sigma);

      // 2. MH 判据: min(1, |Psi(s')/Psi(s)|^2) -> exp(min(0, 2*(log'-log)))
      T log_ratio = 2.0f * (log_psi_proposal - current_log_psi);
      T prob = std::exp(std::min(static_cast<T>(0.0), log_ratio));

      if (dist_acc(rng) < prob) {
        // 3. 接受 (Accept)
        current_log_psi = log_psi_proposal;
        accepts++;
      } else {
        // 4. 拒绝 (Reject): 再次翻转恢复原状
        sigma[site] *= -1.0f;
      }
    }
    return accepts;
  }
};
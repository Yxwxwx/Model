#include <chrono>
#include <execution>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "hamiltonian.hpp"
#include "model.hpp"
#include "sampler.hpp"

// 辅助函数：将数据写入 CSV，方便后续用 Python 画图
void save_to_csv(const std::vector<float>& energies,
                 std::size_t steps_per_chain, const std::string& filename) {
  std::ofstream file(filename);
  if (!file.is_open()) return;

  // 我们只保存前 5 条链的数据，和 Python 代码中的 plot_trace 保持一致
  std::size_t n_plot_chains =
      std::min(std::size_t(5), energies.size() / steps_per_chain);

  file << "Sample_Index,Energy\n";
  for (std::size_t c = 0; c < n_plot_chains; ++c) {
    for (std::size_t s = 0; s < steps_per_chain; ++s) {
      std::size_t global_idx = c * steps_per_chain + s;
      file << global_idx << "," << energies[global_idx] << "\n";
    }
  }
  file.close();
  std::cout << "📈 Plot data saved to " << filename << std::endl;
}

int main() {
  // 1. 基础配置
  std::cout << "✅ Selected Device: CPU (Parallel TBB)" << std::endl;

  // 物理参数
  const std::size_t L = 10;
  const float J = 1.0f;
  const float h = 0.5f;
  const std::size_t num_hidden = 2 * L;

  // 采样参数：并行化配置
  const std::size_t n_chains = 1000;       // 并行 1000 条马尔可夫链
  const std::size_t steps_per_chain = 50;  // 每条链采集 50 个样本
  const std::size_t burn_in = 200;         // 每条链先走 200 步
  const std::size_t total_samples = n_chains * steps_per_chain;

  // 2. 初始化模型与参数
  std::mt19937 global_rng(42);
  auto allocator = std::make_shared<VectorAllocator<float>>();

  RBM<float> model(L, num_hidden, global_rng, allocator);
  TFIM<float> hamiltonian(L, J, h);

  std::cout << "🚀 Running Parallel Metropolis Sampling..." << std::endl;
  std::cout << "🔗 Chains: " << n_chains
            << " | Steps per chain: " << steps_per_chain
            << " | Total: " << total_samples << std::endl;

  // 3. 准备并行采样的输出容器
  // 平铺存储所有的能量，形状等同于 jnp.ravel() 后的 (50000,)
  std::vector<float> flat_energies(total_samples, 0.0f);
  std::vector<float> accepts(n_chains, 0.0f);

  // 构造链索引，用于并行遍历
  std::vector<std::size_t> chain_indices(n_chains);
  std::iota(chain_indices.begin(), chain_indices.end(), 0);

  // 4. 执行并行采样 (对标 jax.vmap)
  auto start_time = std::chrono::high_resolution_clock::now();

  std::for_each(std::execution::par, chain_indices.begin(), chain_indices.end(),
                [&](std::size_t i) {
                  // 为每条链生成独立的随机种子 (对应 jax.random.split)
                  std::mt19937 local_rng(42 + i);

                  // 随机初始化起始状态 (±1.0)
                  std::vector<float> init_state(L);
                  std::uniform_int_distribution<int> spin_dist(0, 1);
                  for (std::size_t j = 0; j < L; ++j) {
                    init_state[j] = spin_dist(local_rng) == 0 ? -1.0f : 1.0f;
                  }

                  // 实例化当前链的采样器
                  MetropolisSampler<float> sampler(model, hamiltonian,
                                                   local_rng, init_state);

                  // 执行 Burn-in
                  sampler.burn_in(burn_in);

                  // 正式采样
                  auto result = sampler.sample(steps_per_chain);

                  // 将当前链的能量序列拷贝到全局一维数组的对应切片中
                  auto start_iter = flat_energies.begin() + i * steps_per_chain;
                  std::copy(result.energies.begin(), result.energies.end(),
                            start_iter);

                  // 记录接受率
                  accepts[i] = result.accept_rate;
                });

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end_time - start_time;

  std::cout << "⏱️ Sampling Time: " << std::fixed << std::setprecision(4)
            << diff.count() << " seconds" << std::endl;

  // 5. 后处理统计信息
  float avg_e =
      std::accumulate(flat_energies.begin(), flat_energies.end(), 0.0f) /
      total_samples;
  float avg_acc =
      std::accumulate(accepts.begin(), accepts.end(), 0.0f) / n_chains;

  // 精确解 (在 C++ 中除非引入 Eigen 做严格对角化，否则一般硬编码或通过外部
  // Python 脚本提供验证) 根据 OBC L=10, J=1.0, h=0.5 估算，大概在
  // -9.7左右，这里作为展示位。
  float exact_e_host =
      -9.73;  // 假定值，具体以你 Python 算出的 exact_e_val 为准

  std::cout << std::string(30, '-') << std::endl;
  std::cout << "Metropolis Acceptance Rate: " << avg_acc * 100.0f << "%"
            << std::endl;
  std::cout << "Untrained RBM Energy Mean:  " << std::fixed
            << std::setprecision(8) << avg_e << std::endl;
  std::cout << "Exact Ground State Energy:  " << exact_e_host
            << " (Approximation for L=10)" << std::endl;
  std::cout << "Energy Error (per site):    "
            << std::abs(avg_e - exact_e_host) / L << std::endl;

  // 6. 导出绘图数据
  save_to_csv(flat_energies, steps_per_chain, "vmc_parallel_results.csv");

  return 0;
}
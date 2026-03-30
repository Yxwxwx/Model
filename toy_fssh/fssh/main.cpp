// Eigen3
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
// progresscpp
#include <progresscpp/ProgressBar.hpp>
// std
#include <chrono>
#include <cmath>
#include <complex>
#include <ctime>
#include <format>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

std::random_device rd;  // 获取随机种子
std::mt19937 gen(rd()); // Mersenne Twister 生成器
std::uniform_real_distribution<double> dis(0.0, 1.0);

struct Neuclus {
  double mass;
  double x;
  double v;
  int state;
};

template <typename T>
Neuclus createNeuclus(double mass, double x, T numerator_for_v, int state) {
  return Neuclus(mass, x, static_cast<double>(numerator_for_v) / mass, state);
}

class Tully {
  /* J. Chem. Phys. 93, 1061 (1990); doi: 10.1063/1.459170 */
private:
  const double A{0.01};
  const double B{1.6};
  const double C{0.005};
  const double D{1.0};
  const std::complex<double> I{0.0, 1.0};

  bool quiet = true;

public:
  auto compute_H(double x) -> Eigen::MatrixXcd {
    /*
    Tully model A:
    V_{11} = A[ 1 - exp(-Bx)], x > 0
    V_{11} = -A[ 1 - exp(-Bx)], x < 0
    V_{12} = V_{21} =  C exp(-D x^2)
    V_{22} = -V_{11}
    */
    auto V11 = x > 0 ? A * (1 - std::exp(-B * x)) : -A * (1 - std::exp(B * x));
    auto V12 = C * std::exp(-D * std::pow(x, 2));

    Eigen::MatrixXcd H(2, 2);
    H(0, 0) = std::complex<double>(V11, 0.0); // 将 V11 转换为复数
    H(0, 1) = std::complex<double>(V12, 0.0); // 将 V12 转换为复数
    H(1, 0) = std::complex<double>(V12, 0.0);
    H(1, 1) = std::complex<double>(-V11, 0.0); // -V11 转换为复数

    return H;
  }

  auto get_adiabatic_state(double x)
      -> std::tuple<double, Eigen::MatrixXcd, double, Eigen::MatrixXcd> {
    auto H = compute_H(x);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigensolver(H);

    if (eigensolver.info() != Eigen::Success) {
      throw std::runtime_error("Eigenvalue decomposition failed");
    }

    Eigen::VectorXcd E = eigensolver.eigenvalues();
    Eigen::MatrixXcd phi = eigensolver.eigenvectors();

    return {E[0].real(), phi.col(0), E[1].real(), phi.col(1)};
  }

  auto get_derivative_t(Neuclus &nuc, Eigen::MatrixXcd &den)
      -> Eigen::VectorXcd {
    auto [E1, phi1, E2, phi2] = get_adiabatic_state(nuc.x);
    const double dx = 1e-5;
    Eigen::MatrixXcd dH = (compute_H(nuc.x + dx) - compute_H(nuc.x)) / dx;

    std::complex<double> F1 = (phi1.adjoint() * dH * phi1)(0, 0);
    std::complex<double> F2 = (phi2.adjoint() * dH * phi2)(0, 0);
    std::complex<double> d12 = (phi1.adjoint() * dH * phi2)(0, 0) / (E2 - E1);

    Eigen::VectorXcd du(5);
    du(0) = nuc.v;
    du(1) = nuc.state == 1 ? -F1 / nuc.mass : -F2 / nuc.mass;
    du(2) = -2.0 * (std::conj(den(0, 1)) * nuc.v * d12).real();
    du(3) = 2.0 * (den(0, 1) * nuc.v * std::conj(d12)).real();
    du(4) = -I * den(0, 1) * (E1 - E2) + den(0, 0) * nuc.v * d12 -
            den(1, 1) * nuc.v * d12;

    return du;
  }

  auto get_state(Neuclus &nuc_old, Eigen::MatrixXcd &den_old,
                 Eigen::VectorXcd &k, double dt, double step = 1.0)
      -> std::tuple<Neuclus, Eigen::MatrixXcd> {
    Neuclus nuc_new{nuc_old.mass, nuc_old.x + step * k(0).real() * dt,
                    nuc_old.v + step * k(1).real() * dt, nuc_old.state};

    auto den_new = den_old;
    den_new(0, 0) += step * k(2) * dt;
    den_new(1, 1) += step * k(3) * dt;
    den_new(0, 1) += step * k(4) * dt;
    den_new(1, 0) = std::conj(den_new(0, 1));

    return std::make_tuple(nuc_new, den_new);
  }

  auto rk4(Neuclus &nuc, Eigen::MatrixXcd &den, double dt)
      -> std::tuple<Neuclus, Eigen::MatrixXcd> {
    Eigen::VectorXcd k1 = get_derivative_t(nuc, den);
    auto [nuc2, den2] = get_state(nuc, den, k1, dt, 0.5);
    Eigen::VectorXcd k2 = get_derivative_t(nuc2, den2);
    auto [nuc3, den3] = get_state(nuc2, den2, k2, dt, 0.5);
    Eigen::VectorXcd k3 = get_derivative_t(nuc3, den3);
    auto [nuc4, den4] = get_state(nuc3, den3, k3, dt);
    Eigen::VectorXcd k4 = get_derivative_t(nuc4, den4);
    Eigen::VectorXcd k_tot = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

    return get_state(nuc, den, k_tot, dt);
  }

  auto test_hop(double dt, Neuclus &nuc, Eigen::MatrixXcd &den) -> bool {
    auto du = get_derivative_t(nuc, den);
    auto b12 = du(2).real();
    auto b21 = du(3).real();

    auto random_val = dis(gen);

    auto [E1, phi1, E2, phi2] = get_adiabatic_state(nuc.x);

    if (nuc.state == 1) {
      auto prob = dt * b21 / den(0, 0).real();
      auto energy_check = E1 + nuc.mass * std::pow(nuc.v, 2) / 2.0 - E2 >= 0.0;

      auto hop = prob >= random_val && energy_check;

      if (hop && !quiet) {
        std::cout << std::format("Hop attempt from state 1->2: prob={:.4f}, "
                                 "random={:.4f}, energy_check={}\n",
                                 prob, random_val, energy_check);
      }
      return hop;
    }

    auto prob = dt * b12 / den(1, 1).real();
    auto hop = prob >= random_val;

    if (hop && !quiet) {
      std::cout << std::format(
          "Hop attempt from state 2->1: prob={:.4f}, random={:.4f}\n", prob,
          random_val);
    }
    return hop;
  }

  auto step_upgrade(double dt, Neuclus &nuc, Eigen::MatrixXcd &den)
      -> std::tuple<Neuclus, Eigen::MatrixXcd> {
    auto [E1, phi1, E2, phi2] = get_adiabatic_state(nuc.x);
    auto [nuc_new, den_new] = rk4(nuc, den, dt);

    if (test_hop(dt, nuc_new, den_new)) {
      auto old_state = nuc_new.state;
      auto old_v = nuc_new.v;
      double v{0.0};

      if (nuc_new.state == 1) {
        nuc_new.state = 2;
        v = std::sqrt(2.0 *
                      (E1 + nuc_new.mass * std::pow(old_v, 2) / 2.0 - E2) /
                      nuc_new.mass);
      } else {
        nuc_new.state = 1;
        v = std::sqrt(2.0 *
                      (E2 + nuc_new.mass * std::pow(old_v, 2) / 2.0 - E1) /
                      nuc_new.mass);
      }

      nuc_new.v = (old_v > 0) ? v : -v;
      if (!quiet) {
        std::cout << std::format(
            "Hop successful: state {}->{}, velocity {:.4f}->{:.4f}\n",
            old_state, nuc_new.state, old_v, nuc_new.v);
      }
    }

    return std::make_tuple(nuc_new, den_new);
  }

  auto simulate(double x0, double v0, double dt = 0.5, int Ntraj = 1000)
      -> std::tuple<int, int, int> {
    int reflections{0};
    int hops{0};
    int transmissions{0};

    std::cout << std::format(
        "\nStarting simulation with x0={}, v0={:.5f}, dt={}, Ntraj={}\n", x0,
        v0, dt, Ntraj);

    // progresscpp::ProgressBar progressBar(Ntraj, 70); // 70是进度条的宽度

    for (auto i = 0; i < Ntraj; i++) {

      // progressBar.display(); // 显示进度条

      auto nuc = Neuclus{2000, x0, v0, 1};
      Eigen::MatrixXcd den = Eigen::MatrixXcd::Zero(2, 2);
      den(0, 0) = std::complex<double>(1.0, 0.0);
      den(1, 1) = std::complex<double>(0.0, 0.0);
      den(0, 1) = std::complex<double>(0.0, 0.0);
      den(1, 0) = std::complex<double>(0.0, 0.0);

      while (-10 <= nuc.x && nuc.x <= 10) {
        std::tie(nuc, den) = step_upgrade(dt, nuc, den);
      }

      if (nuc.state == 2) {
        hops += 1; // hop
      } else if (nuc.x < 0) {
        reflections += 1; // reflection
      } else {
        transmissions += 1; // transmission
      }

      //  ++progressBar; // 更新进度条
    }

    // progressBar.done(); // 完成进度条

    std::cout << std::format(
        "\nSimulation complete: Reflection={},  Hops={}, Transmission={}\n",
        reflections, hops, transmissions);
    return std::make_tuple(reflections, hops, transmissions);
  }
};

auto test() {
  std::vector<double> xlist;
  std::vector<double> elist;
  auto nuc = createNeuclus(2000, -10, 25, 1);
  Eigen::MatrixXcd den =
      (Eigen::MatrixXcd(2, 2) << 1.0, 0.0, 0.0, 0.0).finished();
  auto model = Tully();

  while (-10.0 <= nuc.x && nuc.x <= 10.0) {
    std::tie(nuc, den) = model.step_upgrade(0.5, nuc, den);
    xlist.emplace_back(nuc.x);

    auto [E1, phi1, E2, phi2] = model.get_adiabatic_state(nuc.x);

    auto e = (nuc.state == 1) ? E1 : E2;
    elist.emplace_back(e);
  }
}

auto tully_test() {
  auto start = std::chrono::high_resolution_clock::now();
  auto model = Tully();
  std::vector<double> pspan{1.42,   4.12,   4.539,  4.550,  5.550,
                            5.687,  7.566,  7.811,  8.389,  9.38389,
                            9.810,  11.115, 12.940, 14.360, 15.924,
                            19.621, 22.607, 25.308, 27.726, 30.0};

  // 存储结果的向量，预先分配空间
  std::vector<std::tuple<double, int, int, int>> results(pspan.size());

  // 创建互斥锁用于同步输出
  std::mutex cout_mutex;

  // 确定线程数量（使用可用的硬件线程数，但不超过数据点数量）
  const int num_threads =
      std::min(static_cast<int>(std::thread::hardware_concurrency()),
               static_cast<int>(pspan.size()));
  std::vector<std::thread> threads;

  // 计算每个线程处理的数据范围
  auto worker = [&](int start, int end) {
    for (int i = start; i < end; ++i) {
      auto sim_result = model.simulate(-10.0, pspan[i] / 2000.0);
      // 存储结果，包含对应的p值
      results[i] =
          std::make_tuple(pspan[i], std::get<0>(sim_result),
                          std::get<1>(sim_result), std::get<2>(sim_result));

      // 线程安全地输出结果
      {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << std::format(
            "P={:.3f}, Reflection={}, Hop={}, Transmission={}\n", pspan[i],
            std::get<0>(sim_result), std::get<1>(sim_result),
            std::get<2>(sim_result));
      }
    }
  };

  // 启动线程
  int batch_size = pspan.size() / num_threads;
  for (int i = 0; i < num_threads; ++i) {
    int start = i * batch_size;
    int end = (i == num_threads - 1) ? pspan.size() : (i + 1) * batch_size;
    threads.emplace_back(worker, start, end);
  }

  // 等待所有线程完成
  for (auto &thread : threads) {
    thread.join();
  }

  // 将结果保存到文件
  std::ofstream outfile("tully_results.txt");
  if (outfile.is_open()) {
    outfile << "P\tReflection\tHop\tTransmission\n";
    for (const auto &result : results) {
      outfile << std::format("{:.3f}\t{}\t{}\t{}\n",
                             std::get<0>(result),  // P value
                             std::get<1>(result),  // Reflection
                             std::get<2>(result),  // Hop
                             std::get<3>(result)); // Transmission
    }
    outfile.close();
    std::cout << "Results have been saved to tully_results.txt\n";
  } else {
    std::cerr << "Failed to open output file\n";
  }
  auto end = std::chrono::high_resolution_clock::now();

  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  std::cout << "Elapsed time: " << elapsed.count() << "s\n";
}

auto tully_test_old() {
  auto model = Tully();
  // std::vector<double> pspan {
  //     1.42, 4.12, 4.539, 4.550, 5.550, 5.687, 7.566, 7.811, 8.389, 9.38389,
  //     9.810, 11.115, 12.940, 14.360, 15.924, 19.621, 22.607, 25.308, 27.726, 30.0
  // };
  std::vector<double> pspan{30.0};
  // 预分配结果向量的大小，避免动态扩容
  std::vector<std::tuple<double, int, int, int>> results(pspan.size());
  auto idx{0};

  // 打开输出文件
  std::ofstream outfile("tully_results.txt");
  if (!outfile.is_open()) {
    std::cerr << "Failed to open output file\n";
    return results;
  }

  // 写入表头
  outfile << "P\tReflection\tHop\tTransmission\n";

  for (auto &&p : pspan) {
    auto sim_result = model.simulate(-10.0, p / 2000);

    // 存储结果，包含对应的p值
    results[idx] =
        std::make_tuple(p, std::get<0>(sim_result), std::get<1>(sim_result),
                        std::get<2>(sim_result));

    // 控制台输出
    std::cout << std::format("Simulation complete: P={:.3f}, Reflection={}, "
                             "Hop={}, Transmission={}\n",
                             p, std::get<0>(sim_result),
                             std::get<1>(sim_result), std::get<2>(sim_result));

    // 写入文件
    outfile << std::format("{:.3f}\t{}\t{}\t{}\n", p, std::get<0>(sim_result),
                           std::get<1>(sim_result), std::get<2>(sim_result));

    ++idx;
  }

  outfile.close();
  std::cout << "Results have been saved to tully_results.txt\n";

  return results;
}
auto main() -> int {

  tully_test();

  return 0;
}

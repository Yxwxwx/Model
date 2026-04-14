import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只允许看到 GPU 1
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # 按需分配显存

import time
from functools import partial
import jax
import jax.numpy as jnp
from flax import nnx
import matplotlib.pyplot as plt
import optax

from model import RBM
from sampler import metropolis_chain
from exact import exact_tfim_energy_obc
from driver import train_step


def main():
    # --- 1. 物理参数与训练配置 ---
    L = 10  # 格点数
    J = 1.0  # 耦合
    h = 0.5  # 横场

    num_epochs = 400  # 训练轮数
    n_chains = 4000  # 并行链数量
    steps_per_epoch = 500  # 每轮采集样本数
    learning_rate = 0.01  # 学习率

    device = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
    print(f"✅ 使用设备: {device}")

    # --- 2. 初始化模型与优化器 ---
    key = jax.random.PRNGKey(42)
    model_key, sampler_key, init_state_key = jax.random.split(key, 3)

    # 初始化 RBM
    model = RBM(L, num_hidden=8 * L, rngs=nnx.Rngs(model_key))
    graphdef, state = nnx.split(model)

    # 初始化 Optax 优化器
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(state)  # type: ignore

    # --- 3. 初始化采样器 ---
    # 每条链的初始构型
    configs = jax.random.choice(
        init_state_key, jnp.array([-1.0, 1.0]), shape=(n_chains, L)
    )
    # 为每条链分配独立随机种子
    mcmc_keys = jax.random.split(sampler_key, n_chains)

    # 使用 vmap 包装采样器
    # 采样器输入: (graphdef, state, configs, keys)
    vmap_sampler = jax.vmap(
        partial(metropolis_chain, n_samples=steps_per_epoch, burn_in=200, n_sweeps=L),
        in_axes=(None, None, 0, 0),
    )

    # --- 计算精确解 (强制在 CPU 上执行) ---
    print(f"🧮 正在 CPU 上计算理论精确解...")

    # 方式 A：使用 jax.default_device 装饰器
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        # 确保计算逻辑在这里被调用
        exact_e_val = exact_tfim_energy_obc(L, J, h)

    # 转换为 Python float 彻底脱离 JAX 数组生态，方便后续打印
    exact_e = float(exact_e_val)
    print(f"🎯 理论精确能量: {exact_e:.8f}")

    # --- 4. 训练循环 ---
    energy_history = []
    print("\n🚀 开始变分参数训练...")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # [采样阶段]
        # 使用当前最新的 state 进行采样
        # samples 形状: (n_chains, steps_per_epoch, L)
        # configs 是下一轮采样的起点 (Persistent Chain)
        samples, accepts, configs = vmap_sampler(graphdef, state, configs, mcmc_keys)

        # 展平样本以供训练: (Total_Samples, L)
        flat_samples = samples.reshape(-1, L)

        # [更新阶段]
        state, opt_state, current_energy = train_step(
            graphdef, state, opt_state, flat_samples, J, h, optimizer
        )

        # 更新采样种子，防止采样序列相关
        mcmc_keys = jax.random.split(mcmc_keys[0], n_chains)

        t1 = time.time()

        # 统计
        energy_history.append(current_energy)
        if epoch % 10 == 0 or epoch == 1:
            error = abs(current_energy - exact_e) / L
            print(
                f"Epoch {epoch:3d} | Energy: {current_energy:10.6f} | Error/site: {error:.4e} | Accept: {jnp.mean(accepts):.1%} | Time: {t1 - t0:.3f}s"
            )

    # --- 5. 结果可视化 ---
    plt.figure(figsize=(10, 6))
    plt.plot(energy_history, label="VMC Energy")
    plt.axhline(exact_e, color="black", linestyle="--", label="Exact Ground State")
    plt.xlabel("Training Epochs")
    plt.ylabel("Energy")
    plt.title(f"TFIM L={L} Ground State Search (RBM)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("vmc_training_curve.png", dpi=200)
    print("\n📈 训练曲线已保存为 vmc_training_curve.png")


if __name__ == "__main__":
    main()

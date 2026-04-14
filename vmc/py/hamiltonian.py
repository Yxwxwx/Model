import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=("log_psi_fn",))
def local_energy(log_psi_fn, sigma: jax.Array, J: float, h: float) -> jax.Array:
    """
    计算 TFIM 模型的局部能量。
    支持单构型 (L,) 或 批量构型 (..., L)。
    """
    # 保持计算精度
    sigma = sigma.astype(jnp.float32)
    L = sigma.shape[-1]

    # 1. 相互作用项 (对角项): -J * sum(sigma_i * sigma_{i+1})
    # 使用 jnp.sum 时保持维度一致性
    interaction = -J * jnp.sum(sigma[..., :-1] * sigma[..., 1:], axis=-1)

    # 2. 横场项 (非对角项): -h * sum(Psi(sigma^i) / Psi(sigma))

    # 构造翻转矩阵
    # flip_mask: (L, L) 的单位阵
    flip_mask = jnp.eye(L, dtype=sigma.dtype)

    # 利用广播机制生成 L 个翻转构型
    # 如果 sigma 是 (B, L)，那么 sigma_flipped 就是 (B, L, L)
    # 其中中间的 L 维度表示“第几个位点被翻转了”
    sigma_flipped = sigma[..., None, :] * (1.0 - 2.0 * flip_mask)

    # 计算原始对数幅度和翻转后的对数幅度
    log_psi_sigma = log_psi_fn(sigma)  # (B,)
    log_psi_flipped = log_psi_fn(sigma_flipped)  # (B, L)

    # 计算比值: exp(log_psi_flipped - log_psi_sigma)
    # 注意 log_psi_sigma 需要增加一个轴来匹配 (B, L) 的减法
    ratios = jnp.exp(log_psi_flipped - jnp.expand_dims(log_psi_sigma, axis=-1))

    field = -h * jnp.sum(ratios, axis=-1)

    return interaction + field

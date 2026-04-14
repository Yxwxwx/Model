import time
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import matplotlib.pyplot as plt
from functools import partial


from hamiltonian import local_energy


@partial(jax.jit, static_argnames=("graphdef", "optimizer"))
def train_step(graphdef, state, opt_state, samples, J, h, optimizer):
    """
    计算能量梯度并更新参数。
    使用 Surrogate Loss: 2 * E[ (E_loc - <E_loc>) * log_Psi ]
    """

    def surrogate_loss(current_state):
        # 定义当前的 log_psi 纯函数
        def log_psi_fn(s):
            out, _ = graphdef.apply(current_state)(s)
            return out

        # 计算局部能量 (stop_gradient 保证不计算 eloc 内部对参数的导数)
        eloc = local_energy(log_psi_fn, samples, J, h)
        eloc_fixed = jax.lax.stop_gradient(eloc)
        avg_eloc = jnp.mean(eloc_fixed)

        # 计算当前样本的 log_psi 值用于求导
        log_psi_val = log_psi_fn(samples)

        # 核心公式：变分梯度的替代损失
        loss = 2.0 * jnp.mean((eloc_fixed - avg_eloc) * log_psi_val)
        return loss, avg_eloc

    # 计算梯度
    (loss_val, energy_val), grads = jax.value_and_grad(surrogate_loss, has_aux=True)(
        state
    )

    # 优化器更新
    updates, opt_state = optimizer.update(grads, opt_state, state)
    new_state = optax.apply_updates(state, updates)

    return new_state, opt_state, energy_val

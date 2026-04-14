import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx


@partial(jax.jit, static_argnames=("graphdef", "n_samples", "burn_in", "n_sweeps"))
def metropolis_chain(
    graphdef: nnx.GraphDef,
    state: nnx.State,
    init_state: jax.Array,
    key: jax.Array,
    *,
    n_samples: int,
    burn_in: int = 100,
    n_sweeps: int = 0,
):
    L = init_state.shape[-1]
    sweep_len = n_sweeps if n_sweeps > 0 else L

    # ==========================================
    # 核心：定义 log_psi 计算逻辑
    # 直接使用 graphdef.apply，跳过中间层
    # ==========================================
    def get_log_psi(s):
        # NNX 的 apply 返回 (output, updates)，我们只取 output
        res, _ = graphdef.apply(state)(s)
        return res

    # ==========================================
    # 1. 单次自旋翻转 (Atomic Flip)
    # ==========================================
    def single_flip_step(carry, random_vals):
        sigma, logp, _ = carry
        key_site, key_acc = random_vals

        # 随机挑选位点并翻转
        site = jax.random.randint(key_site, (), 0, L)
        proposal = sigma.at[site].set(-sigma[site])

        # 计算新的 log_psi
        logp_new = get_log_psi(proposal)

        # Metropolis 判据: 2.0 * Re(log_psi' - log_psi)
        log_ratio = 2.0 * (logp_new - logp)
        log_u = jnp.log(jax.random.uniform(key_acc))
        accept = log_u < log_ratio

        # 状态更新
        sigma_next = jnp.where(accept, proposal, sigma)
        logp_next = jnp.where(accept, logp_new, logp)

        return (sigma_next, logp_next, None), accept

    # ==========================================
    # 2. 扫视步 (Sweep)
    # ==========================================
    def sweep_step(carry, rng):
        rng_sites, rng_accs = jax.random.split(rng, 2)
        keys_sites = jax.random.split(rng_sites, sweep_len)
        keys_accs = jax.random.split(rng_accs, sweep_len)

        (sigma_next, logp_next, _), accepts = jax.lax.scan(
            single_flip_step, carry, (keys_sites, keys_accs)
        )
        return (sigma_next, logp_next, None), jnp.mean(accepts)

    # ==========================================
    # 3. 运行逻辑
    # ==========================================
    init_state = init_state.astype(jnp.float32)
    init_logp = get_log_psi(init_state)
    carry = (init_state, init_logp, None)

    key_burn, key_sample = jax.random.split(key)

    # --- Burn-in ---
    burn_keys = jax.random.split(key_burn, burn_in)
    carry, _ = jax.lax.scan(sweep_step, carry, burn_keys)

    # --- Sampling ---
    sample_keys = jax.random.split(key_sample, n_samples)

    def sample_op(c, rng):
        next_c, acc = sweep_step(c, rng)
        return next_c, (next_c[0], acc)

    _, (samples, accepts) = jax.lax.scan(sample_op, carry, sample_keys)

    return samples, accepts, carry[0]

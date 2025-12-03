import jax
import jax.numpy as jnp


class RBM:
    """RBM with complex parameters using JAX (functional)."""

    def __init__(self, n_visible, n_hidden, key):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.params = self.reset(key)

    def reset(self, key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

        b = jax.random.normal(k1, (self.n_visible,)) + 1j * jax.random.normal(
            k2, (self.n_visible,)
        )
        c = jax.random.normal(k3, (self.n_hidden,)) + 1j * jax.random.normal(
            k4, (self.n_hidden,)
        )
        W = jax.random.normal(
            k5, (self.n_hidden, self.n_visible)
        ) + 1j * jax.random.normal(k6, (self.n_hidden, self.n_visible))

        return (b / 10, c / 10, W / 10)

    def unpack(self):
        return self.params


@jax.jit
def p(b, c, W, v):
    term1 = jnp.exp(jnp.vdot(b, v))
    hidden = jnp.prod(jnp.cosh(c + W @ v))
    return term1 * hidden * (2 ** W.shape[0])


@jax.jit
def p_ratio(b, c, W, v1, v2):
    f1 = jnp.cosh(c + W @ v1)
    f2 = jnp.cosh(c + W @ v2)
    log_diff = jnp.vdot(b, v2 - v1) + jnp.sum(jnp.log(f2 / f1))
    return jnp.exp(log_diff)


@jax.jit
def p_ratios(b, c, W, v1, v_list):
    return jax.vmap(lambda v: p_ratio(b, c, W, v1, v))(v_list)


def flip(x, idx):
    return x.at[idx].set(1 - x[idx])


def all_flips(x):
    return jax.vmap(lambda i: flip(x, i))(jnp.arange(x.shape[0]))


@jax.jit
def local_energy(x, b, c, W, J, B):
    couplings = jnp.where(x[:-1] == x[1:], 1, -1)
    e_interaction = J * jnp.sum(couplings)

    flips = all_flips(x)
    ratios = p_ratios(b, c, W, x, flips)
    e_field = B * jnp.sum(ratios)

    return e_interaction + e_field


@jax.jit
def mcmc_step(state, key, b, c, W, J, B):
    """One Metropolis step."""
    key, k1, k2 = jax.random.split(key, 3)

    # random site
    idx = jax.random.randint(k1, shape=(), minval=0, maxval=state.shape[0])
    new_state = flip(state, idx)

    # acceptance probability
    accept_prob = jnp.abs(p_ratio(b, c, W, state, new_state)) ** 2
    rand = jax.random.uniform(k2)

    # Metropolis accept/reject
    state = jax.lax.cond(
        rand < accept_prob, lambda _: new_state, lambda _: state, operand=None
    )

    # local energy
    energy = jnp.real(local_energy(state, b, c, W, J, B))

    return state, key, energy


@jax.jit(static_argnums=(7,))
def mcmc_chain(state, key, b, c, W, J, B, n_steps):
    """Run full MCMC chain using lax.scan."""

    def body(carry, _):
        state, key = carry
        state, key, E = mcmc_step(state, key, b, c, W, J, B)
        return (state, key), E

    (final_state, final_key), energies = jax.lax.scan(
        body, (state, key), xs=None, length=n_steps
    )
    return final_state, energies


if __name__ == "__main__":
    n = 10
    J, B = -2, -1
    n_visible = n
    n_hidden = 2 * n

    key = jax.random.PRNGKey(7)

    psi = RBM(n_visible, n_hidden, key)
    b, c, W = psi.unpack()

    key, k1 = jax.random.split(key)
    state = jax.random.randint(k1, shape=(n,), minval=0, maxval=2)

    states, energies = [], []
    n_samples = 50000
    n_flips = 1

    final_state, energies = mcmc_chain(state, key, b, c, W, J, B, int(n_samples))

    import matplotlib.pyplot as plt

    # collapse_input
    plt.figure(figsize=(12, 5))
    plt.plot(energies[:1000])  # Plot some
    plt.grid()
    plt.tick_params(labelsize=15)
    plt.ylabel("Local energy", fontsize=20)
    plt.xlabel("Sample", fontsize=20)
    plt.title(
        f"E = {jnp.mean(jnp.array(energies)):.4f} +- {jnp.std(jnp.array(energies)) / jnp.sqrt(n_samples):.4f}"
    )
    # plt.show()

    def tensor_prod(idx, s, size=10):
        """Tensor product of `s` acting on indices `idx`."""
        Id = jnp.eye(2)

        # convert idx to 1D JAX array
        idx = jnp.atleast_1d(jnp.array(idx))
        s = jnp.array(s)

        # Boolean mask for each position
        # is_target[k] = True if k in idx
        is_target = jnp.isin(jnp.arange(size), idx)

        # Construct list of matrices
        matrices = [jnp.where(is_target[k], s, Id) for k in range(size)]

        # Sequential Kronecker product
        prod = matrices[0]
        for k in range(1, size):
            prod = jnp.kron(prod, matrices[k])

        return prod

    sx = jnp.array([[0, 1], [1, 0]])
    sz = jnp.array([[1, 0], [0, -1]])

    H_int = sum(tensor_prod([k, k + 1], sz, size=n) for k in range(n - 1))
    H_field = sum(tensor_prod([k], sx, size=n) for k in range(n))

    H = J * H_int + B * H_field

    e_vals, e_vecs = jnp.linalg.eigh(H)
    print(f"Exact ground state energy: {e_vals[0]:.4f}")

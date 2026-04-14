import jax
import jax.numpy as jnp


def _kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = jnp.kron(out, op)
    return out


def exact_tfim_energy_obc(L: int, J: float, h: float) -> jax.Array:
    """
    Exact ground-state energy for small L by dense diagonalization.

    This is only for verification, not for production use.
    """
    sx = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    sz = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.float32)
    I = jnp.eye(2, dtype=jnp.float32)

    dim = 1 << L
    H = jnp.zeros((dim, dim), dtype=jnp.float32)

    for i in range(L - 1):
        ops = [I for _ in range(L)]
        ops[i] = sz
        ops[i + 1] = sz
        H = H + (-J) * _kron_all(ops)

    for i in range(L):
        ops = [I for _ in range(L)]
        ops[i] = sx
        H = H + (-h) * _kron_all(ops)

    return jnp.min(jnp.linalg.eigvalsh(H))

import jax
import jax.numpy as jnp
from flax import nnx

# jax.config.update("jax_enable_x64", True)


class RBM(nnx.Module):
    """
    Real-valued RBM wavefunction for the TFIM in the σ^z basis.

    \Psi(\sigma) = exp(a · \sigma) \prod_j 2 cosh(b_j + W_j · \sigma)

    Therefore

    log \Psi(\sigma) = a · \sigma + \sum_j log(2 cosh(b_j + W_j · \sigma))

    Here \sigma_i = ±1.
    """

    def __init__(self, num_visible: int, num_hidden: int, *, rngs: nnx.Rngs):
        self.visible_bias = nnx.Param(jnp.zeros((num_visible,), dtype=jnp.float32))
        self.hidden = nnx.Linear(
            num_visible,
            num_hidden,
            param_dtype=jnp.float32,
            kernel_init=jax.nn.initializers.normal(stddev=0.01),
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, sigma: jax.Array) -> jax.Array:
        """
        sigma:
            shape (..., L), values ±1
        return:
            log Psi(sigma), shape (...,)
        """
        sigma = sigma.astype(jnp.float32)
        visible_term = jnp.sum(sigma * self.visible_bias.value, axis=-1)
        hidden_pre = self.hidden(sigma)
        hidden_term = jnp.sum(jnp.logaddexp(hidden_pre, -hidden_pre), axis=-1)
        return visible_term + hidden_term


if __name__ == "__main__":
    model = RBM(10, 20, rngs=nnx.Rngs(42))

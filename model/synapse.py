import jax
import jax.numpy as jnp
import equinox as eqx  # type: ignore[import]
from functools import partial


class Synapse(eqx.Module):
    """Model of a synapse.

    The synapse state keeps track of
    0: post-synaptic potential
    1: derivative of psp with respect to time

    The synapse state is expected to have shape (2,) or (2, N)
    """

    gain: float
    time_constant: float
    dt: float

    @staticmethod
    def init_state(like: jax.Array) -> jax.Array:
        return jnp.array(2 * [jnp.zeros_like(like)])

    @partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        state: jax.Array,
        input: jax.Array,
    ) -> jax.Array:
        """Simulate one timestep of the model.

        state: [psp, psp_prime]; shape (2,) or (2, N)
        - psp: post-synaptic potential
        - psp_prime: derivative of psp with respect to time

        input: presynaptic spike density; shape () or (N)

        returns updated state
        """

        psp, psp_prime = state

        psp_double_prime = (
            self.gain / self.time_constant * input
            - 2 / self.time_constant * psp_prime
            - psp / self.time_constant**2
        )

        psp_prime += psp_double_prime * self.dt
        psp += psp_prime * self.dt
        new_state = jnp.array([psp, psp_prime])

        return new_state

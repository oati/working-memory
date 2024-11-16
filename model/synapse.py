import jax
import jax.numpy as jnp
import equinox as eqx  # type: ignore[import]
from typing import NamedTuple


class SynapseState(NamedTuple):
    # post-synaptic potential
    psp: jax.Array
    # derivative of psp with respect to time
    psp_prime: jax.Array


class Synapse(eqx.Module):
    dt: float
    gain: float
    time_constant: float

    def __init__(self, gain: float, time_constant: float, dt: float):
        self.dt = dt
        self.gain = gain
        self.time_constant = time_constant

    @staticmethod
    def init_state(like: jax.Array) -> SynapseState:
        s = jnp.zeros_like(like)
        return SynapseState(s, s)

    @eqx.filter_jit
    def __call__(self, state: SynapseState, input: jax.Array) -> SynapseState:
        psp, psp_prime = state

        psp_double_prime = (
            self.gain / self.time_constant * input
            - 2 / self.time_constant * psp_prime
            - psp / self.time_constant**2
        )

        psp_prime += psp_double_prime * self.dt
        psp += psp_prime * self.dt
        new_state = SynapseState(psp, psp_prime)

        return new_state

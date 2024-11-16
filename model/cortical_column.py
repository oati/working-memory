import equinox as eqx  # type: ignore[import]
import jax
import jax.numpy as jnp
from .synapse import Synapse, SynapseState
from typing import NamedTuple


class CorticalColumnState(NamedTuple):
    pyramidal_synapse: SynapseState
    excitatory_synapse: SynapseState
    slow_inhibitory_synapse: SynapseState
    fast_inhibitory_synapse: SynapseState
    l_fast_inhibitory_synapse: SynapseState

    # used by other parts of the model
    pyramidal_firing_rate: jax.Array


class CorticalColumnHyperparameters(NamedTuple):
    # activation function
    e0: float
    r: float
    s0: float

    # synapses
    excitatory_gain: float
    excitatory_time_constant: float
    slow_inhibitory_gain: float
    slow_inhibitory_time_constant: float
    fast_inhibitory_gain: float
    fast_inhibitory_time_constant: float

    # intra-column connections
    c_ep: float
    c_pe: float
    c_sp: float
    c_ps: float
    c_pp: float
    c_fp: float
    c_fs: float
    c_pf: float
    c_ff: float

    # input variance
    var_p: float
    var_f: float


class CorticalColumn(eqx.Module):
    dt: float
    hparams: CorticalColumnHyperparameters
    excitatory_synapse: Synapse
    slow_inhibitory_synapse: Synapse
    fast_inhibitory_synapse: Synapse

    def activation_function(self, input: jax.Array) -> jax.Array:
        return (
            2
            * self.hparams.e0
            / (1 + jnp.exp(self.hparams.r * (self.hparams.s0 - input)))
        )

    def __init__(
        self,
        hparams: CorticalColumnHyperparameters,
        dt: float,
    ):
        self.dt = dt
        self.hparams = hparams

        self.excitatory_synapse = Synapse(
            hparams.excitatory_gain, hparams.excitatory_time_constant, dt
        )
        self.slow_inhibitory_synapse = Synapse(
            hparams.slow_inhibitory_gain, hparams.slow_inhibitory_time_constant, dt
        )
        self.fast_inhibitory_synapse = Synapse(
            hparams.fast_inhibitory_gain, hparams.fast_inhibitory_time_constant, dt
        )

    def init_state(self, like: jax.Array) -> CorticalColumnState:
        s = Synapse.init_state(like)
        return CorticalColumnState(
            s, s, s, s, s, self.activation_function(jnp.zeros_like(like))
        )

    @eqx.filter_jit
    def __call__(
        self,
        state: CorticalColumnState,
        long_range_excitatory_input: jax.Array,
        long_range_inhibitory_input: jax.Array,
        excitatory_input: jax.Array,
        inhibitory_input: jax.Array,
        key: jax.Array,
        layer_wm: bool = False,
    ) -> CorticalColumnState:
        (p_state, e_state, s_state, f_state, l_state, _) = state

        yp, ye, ys, yf, yl = [y for y, _ in state[:-1]]

        # apply random noise to the inputs
        p_key, f_key = jax.random.split(key)
        excitatory_input += (
            jax.random.normal(p_key, excitatory_input.shape) * self.hparams.var_p**0.5
        )
        inhibitory_input += (
            jax.random.normal(f_key, inhibitory_input.shape) * self.hparams.var_f**0.5
        )

        # pyramidal neurons
        membrane_potential = (
            self.hparams.c_pe * ye
            + jnp.where(
                jnp.logical_and(layer_wm, excitatory_input == 0),
                self.hparams.c_pp * yp,
                0,
            )
            - self.hparams.c_ps * ys
            - self.hparams.c_pf * yf
            + long_range_excitatory_input
        )
        firing_rate = self.activation_function(membrane_potential)
        new_p_state = self.excitatory_synapse(p_state, firing_rate)
        pyramidal_firing_rate = firing_rate

        # excitatory interneurons
        membrane_potential = self.hparams.c_ep * yp
        firing_rate = self.activation_function(membrane_potential)
        new_e_state = self.excitatory_synapse(
            e_state, firing_rate + excitatory_input / self.hparams.c_pe
        )

        # slow inhibitory interneurons
        membrane_potential = self.hparams.c_sp * yp
        firing_rate = self.activation_function(membrane_potential)
        new_s_state = self.slow_inhibitory_synapse(s_state, firing_rate)

        # fast inhibitory interneurons
        membrane_potential = (
            self.hparams.c_fp * yp
            - self.hparams.c_fs * ys
            - self.hparams.c_ff * yf
            + yl
            + long_range_inhibitory_input
        )
        firing_rate = self.activation_function(membrane_potential)
        new_f_state = self.fast_inhibitory_synapse(f_state, firing_rate)
        new_l_state = self.excitatory_synapse(l_state, inhibitory_input)

        new_state = CorticalColumnState(
            new_p_state,
            new_e_state,
            new_s_state,
            new_f_state,
            new_l_state,
            pyramidal_firing_rate,
        )
        return new_state

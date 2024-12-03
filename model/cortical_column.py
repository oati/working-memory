import equinox as eqx
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple
from .synapse import Synapse


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
    c_pp: float  # this value changes depending on the layer and model phase
    c_fp: float
    c_fs: float
    c_pf: float
    c_ff: float

    # input variance
    var_p: float
    var_f: float


class CorticalColumn(eqx.Module):
    """Neural mass model of a single cortical column with 4 neural populations.

    There are 6 synapses simulated in the model:
    0: pyramidal neurons
    1: excitatory interneurons
    2: slow inhibitory interneurons
    3: fast inhibitory interneurons
    4: excitatory input synapse
    5: inhibitory input synapse
    """

    dt: float
    hparams: CorticalColumnHyperparameters
    connectivity_matrix: list[list[float]]
    synapses: list[Synapse]

    def activation_function(self, input: jax.Array) -> jax.Array:
        """Sigmoidal activation function."""
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

        # this matrix describes the connectivity of synapses between populations
        self.connectivity_matrix = [
            [hparams.c_pp, hparams.c_pe, -hparams.c_ps, -hparams.c_pf, 1, 0],
            [hparams.c_ep, 0, 0, 0, 0, 0],
            [hparams.c_sp, 0, 0, 0, 0, 0],
            [hparams.c_fp, 0, -hparams.c_fs, -hparams.c_ff, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]

        excitatory_synapse_hparams = (
            hparams.excitatory_gain,
            hparams.excitatory_time_constant,
        )
        slow_inhibitory_synapse_hparams = (
            hparams.slow_inhibitory_gain,
            hparams.slow_inhibitory_time_constant,
        )
        fast_inhibitory_synapse_hparams = (
            hparams.fast_inhibitory_gain,
            hparams.fast_inhibitory_time_constant,
        )

        # initialize Synapses
        self.synapses = [
            Synapse(*excitatory_synapse_hparams, dt),
            Synapse(*excitatory_synapse_hparams, dt),
            Synapse(*slow_inhibitory_synapse_hparams, dt),
            Synapse(*fast_inhibitory_synapse_hparams, dt),
            Synapse(*excitatory_synapse_hparams, dt),
            Synapse(*excitatory_synapse_hparams, dt),
        ]

    @staticmethod
    def init_state(like: jax.Array) -> jax.Array:
        return jnp.array(6 * [Synapse.init_state(like)])

    @partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        state: jax.Array,
        long_range_excitatory_input: jax.Array,
        long_range_inhibitory_input: jax.Array,
        excitatory_input: jax.Array,
        inhibitory_input: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Simulate one timestep of the model.

        state: stacked synapse states of the 6 synapses; shape (6, 2) or (6, 2, N)
        inputs: inputs to the cortical column; shape () or (N,)
        key: PRNG key

        returns (updated state, pyramidal firing rate)

        the shape of pyramidal firing rate is () or (N,)
        """

        # apply random noise to the inputs
        p_key, f_key = jax.random.split(key)
        excitatory_input += (
            jax.random.normal(p_key, excitatory_input.shape) * self.hparams.var_p**0.5
        )
        inhibitory_input += (
            jax.random.normal(f_key, inhibitory_input.shape) * self.hparams.var_f**0.5
        )

        # get post-synaptic potentials
        psp = state[:, 0]

        # calculate firing rates
        # this is analogous to a vectorized matrix-vector multiplication
        # along psp's axes 0 (synapse types)
        membrane_potentials = jnp.tensordot(
            jnp.array(self.connectivity_matrix), psp, axes=(1, 0)
        )

        # apply inputs
        membrane_potentials += jnp.array(
            [
                long_range_excitatory_input,
                jnp.zeros_like(excitatory_input),
                jnp.zeros_like(excitatory_input),
                long_range_inhibitory_input,
                excitatory_input,
                inhibitory_input,
            ]
        )

        # calculate firing rates
        firing_rates = self.activation_function(membrane_potentials)

        # simulate synapses
        new_state = jnp.array(
            [
                synapse(*args)
                for synapse, *args in zip(self.synapses, state, firing_rates)
            ]
        )

        # extract pyramidal firing rate
        pyramidal_firing_rate = firing_rates[0]

        return new_state, pyramidal_firing_rate

    @partial(jax.jit, static_argnames=["self"])
    def get_pyramidal_firing_rate(
        self, state: jax.Array, long_range_excitatory_input: jax.Array
    ) -> jax.Array:
        """Compute pyramidal firing rates. This function is needed for type A synapses."""
        psp = state[:, 0]
        pyramidal_membrane_potential = jnp.tensordot(
            jnp.array(self.connectivity_matrix[0]), psp, axes=(0, 0)
        )
        pyramidal_firing_rate = self.activation_function(pyramidal_membrane_potential)
        return pyramidal_firing_rate

import equinox as eqx
import jax
import jax.numpy as jnp
from functools import partial
from .cortical_column import (
    CorticalColumn,
    CorticalColumnHyperparameters,
)
from typing import NamedTuple


class ModelParameters(NamedTuple):
    w_l1_l1: jax.Array
    k_l2_l2: jax.Array
    a_l2_l2: jax.Array
    k_l3_l3: jax.Array
    a_l3_l3: jax.Array
    w_l2_l3: jax.Array


class ModelHyperparameters(NamedTuple):
    w_l1_wm: float
    w_wm_l1: float
    w_l2_l1: float
    w_l3_l2: float
    t: float
    r: float


class Model(eqx.Module):
    """Full model of working memory.

    There are 4 layers in the model:
    0: layer WM
    1: layer 1
    2: layer 2
    3: layer 3

    The model state is expected to have shape (4, 6, 2, N)
    where the dimensions correspond to (layer, synapse, psp/psp_prime, feature)
    """

    hparams: ModelHyperparameters
    cortical_column: CorticalColumn
    wm_maintenance_cortical_column: CorticalColumn

    def __init__(
        self,
        cortical_column_hparams: CorticalColumnHyperparameters,
        hparams: ModelHyperparameters,
        dt: float,
    ):
        self.hparams = hparams

        # initialize CorticalColumns to be vectorized
        self.cortical_column = CorticalColumn(
            cortical_column_hparams._replace(c_pp=0), dt
        )
        self.wm_maintenance_cortical_column = CorticalColumn(
            cortical_column_hparams, dt
        )

    def init_params(self, like: jax.Array) -> ModelParameters:
        """Initialize parameters based on the number of features in an input pattern."""
        n = jnp.size(like)
        m = jnp.zeros((n, n))
        return ModelParameters(*6 * [m])

    def init_state(self, like: jax.Array) -> jax.Array:
        """Initialize the model state based on the number of features in an input pattern."""

        s = CorticalColumn.init_state(like)
        return jnp.array(4 * [s])

    @partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        state: jax.Array,
        params: ModelParameters,
        excitatory_inputs: jax.Array,
        inhibitory_inputs: jax.Array,
        key: jax.Array,
        wm_reset: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Simulate one timestep of the model.

        params: trained parameters of the model
        state: stacked cortical column states of the 4 layers; shape (4, 6, 2, N)
        inputs: inputs to the model; shape (4, N)
        key: PRNG key
        wm_reset: working memory reset signal

        returns (updated state, pyramidal firing rates)

        the shape of pyramidal firing rates is (4, N)
        """

        # determine cortical columns depending on the wm_reset signal
        cortical_columns = [
            partial(
                jax.lax.cond,
                wm_reset,
                self.cortical_column,
                self.wm_maintenance_cortical_column,
            ),
            self.cortical_column,
            self.cortical_column,
            self.cortical_column,
        ]
        get_pyramidal_firing_rates = [
            partial(
                jax.lax.cond,
                wm_reset,
                self.cortical_column.get_pyramidal_firing_rate,
                self.wm_maintenance_cortical_column.get_pyramidal_firing_rate,
            ),
            self.cortical_column.get_pyramidal_firing_rate,
            self.cortical_column.get_pyramidal_firing_rate,
            self.cortical_column.get_pyramidal_firing_rate,
        ]

        # get post-synaptic potentials of pyramidal neurons
        ppsp = state[:, 0, 0]

        # compute long range excitatory inputs
        long_range_excitatory_inputs = jnp.array(
            [
                self.hparams.w_wm_l1 * ppsp[1],
                self.hparams.w_l1_wm * ppsp[0] + params.w_l1_l1 @ ppsp[1],
                self.hparams.w_l2_l1 * ppsp[1] + params.w_l2_l3 @ ppsp[3],
                self.hparams.w_l3_l2 * ppsp[2],
            ]
        )

        # compute pyramidal firing rates
        pfr = jnp.array(
            [
                get_pyramidal_firing_rate(*args)
                for get_pyramidal_firing_rate, *args in zip(
                    get_pyramidal_firing_rates, state, long_range_excitatory_inputs
                )
            ]
        )

        # compute long range inhibitory inputs
        inhibitor = self.hparams.r * jax.nn.relu(self.hparams.t - pfr[1].sum())
        long_range_inhibitory_inputs = jnp.array(
            [
                jnp.zeros_like(ppsp[0]),
                jnp.zeros_like(ppsp[0]),
                params.k_l2_l2 @ ppsp[2] + params.a_l2_l2 @ pfr[2] + inhibitor,
                params.k_l3_l3 @ ppsp[3] + params.a_l3_l3 @ pfr[3],
            ]
        )

        # simulate cortical columns
        keys = jax.random.split(key, 4)
        new_state = jnp.array(
            [
                cortical_column(*args)[0]
                for cortical_column, *args in zip(
                    cortical_columns,
                    state,
                    long_range_excitatory_inputs,
                    long_range_inhibitory_inputs,
                    excitatory_inputs,
                    inhibitory_inputs,
                    keys,
                )
            ]
        )

        return new_state, pfr

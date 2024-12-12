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
    w_wm_wm: float  # C_pp in the paper, only active during maintenance period
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

    def __init__(
        self,
        cortical_column_hparams: CorticalColumnHyperparameters,
        hparams: ModelHyperparameters,
        dt: float,
    ):
        self.hparams = hparams

        # initialize CorticalColumns
        self.cortical_column = CorticalColumn(cortical_column_hparams, dt)

    @staticmethod
    def init_params(shape: tuple[int, ...]) -> ModelParameters:
        """Initialize parameters based on the number of features in an input pattern."""
        n = shape[0]
        m = jnp.zeros((n, n))
        return ModelParameters(*6 * [m])

    @staticmethod
    def init_state(shape: tuple[int, ...]) -> jax.Array:
        """Initialize the model state based on the number of features in an input pattern."""
        s = CorticalColumn.init_state(shape)
        return jnp.array(4 * [s])

    @partial(jax.jit, static_argnames=["self"])
    def add_input_noise(self, *args, **kwargs):
        return self.cortical_column.add_input_noise(*args, **kwargs)

    @partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        state: jax.Array,
        params: ModelParameters,
        excitatory_inputs: jax.Array,
        inhibitory_inputs: jax.Array,
        wm_maintenance: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Simulate one timestep of the model.

        params: trained parameters of the model
        state: stacked cortical column states of the 4 layers; shape (4, 6, 2, N)
        inputs: inputs to the model; shape (4, N)
        wm_maintenance: working memory maintenance signal,
                        wm layer auto-excitation signal; shape (N)

        returns (updated state, pyramidal firing rates, fast inhibitory firing rates)

        the shape of the firing rates is (4, N)
        """

        # get pyramidal post-synaptic potentials
        ppsp = state[:, 0, 0]

        # determine layer wm auto-excitation based on wm_maintenance signal
        wm_auto_excitation = wm_maintenance * self.hparams.w_wm_wm * ppsp[0]
        # compute long range excitatory inputs
        long_range_excitatory_inputs = jnp.array(
            [
                wm_auto_excitation + self.hparams.w_wm_l1 * ppsp[1],
                self.hparams.w_l1_wm * ppsp[0] + params.w_l1_l1 @ ppsp[1],
                self.hparams.w_l2_l1 * ppsp[1] + params.w_l2_l3 @ ppsp[3],
                self.hparams.w_l3_l2 * ppsp[2],
            ]
        )

        # compute pyramidal firing rates
        pfr = jax.vmap(self.cortical_column.get_pyramidal_firing_rate)(
            state, long_range_excitatory_inputs
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
        new_state, _, fifr = jax.vmap(self.cortical_column)(
            state,
            long_range_excitatory_inputs,
            long_range_inhibitory_inputs,
            excitatory_inputs,
            inhibitory_inputs,
        )

        return new_state, pfr, fifr

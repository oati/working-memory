import equinox as eqx  # type: ignore[import]
import jax
import jax.numpy as jnp
from cortical_column import (
    CorticalColumn,
    CorticalColumnState,
    CorticalColumnHyperparameters,
)
from typing import NamedTuple, Optional
from enum import Enum


class ModelState(NamedTuple):
    wm_layer: CorticalColumnState
    layer1: CorticalColumnState
    layer2: CorticalColumnState
    layer3: CorticalColumnState


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


class TrainingHyperparameters(NamedTuple):
    theta_low1: float
    theta_low2: float
    theta_high2: float
    theta_low3: float
    gamma_w: float
    gamma_k: float
    gamma_a: float
    gamma_wb: float
    w_l1_l1_max: float
    k_l2_l2_max: float
    a_l2_l2_max: float
    k_l3_l3_max: float
    a_l3_l3_max: float
    w_l2_l3_max: float
    w_max_sum: float
    k_max_sum: float
    # a_max_sum is computed dynamically


SynapseType = Enum("SynapseType", ["W", "K", "A", "Wb"])


class UpdateRule(eqx.Module):
    synapse_type: SynapseType
    gamma: float
    e0: float
    theta_low: float
    theta_high: Optional[float]
    weights_max: float
    max_sum: Optional[float]

    def __call__(
        self,
        weights: jax.Array,
        firing_rates_1: jax.Array,
        firing_rates_2: jax.Array,
    ):
        # hebbian / anti-hebbian rule
        v = jax.nn.relu(firing_rates_1 / (2 * self.e0) - self.theta_low)

        # if type A, then use the anti-hebbian rule
        if self.synapse_type == SynapseType.A:
            w = jax.nn.relu(self.theta_high - firing_rates_2 / (2 * self.e0))
        else:
            w = jax.nn.relu(firing_rates_2 / (2 * self.e0) - self.theta_low)

        delta_weights = self.gamma * jnp.outer(v, w) * (self.weights_max - weights)
        weights += delta_weights

        # normalization to maximum
        if self.synapse_type != SynapseType.Wb:
            sum_weights = jnp.sum(weights, axis=1, keepdims=True)

            # if type A, then use min(sum_weights)
            if self.synapse_type == SynapseType.A:
                max_sum = sum_weights.min()
            else:
                max_sum = self.max_sum

            weights = jnp.where(
                sum_weights > max_sum, weights * max_sum / sum_weights, weights
            )

        return weights


class Model(eqx.Module):
    model_hparams: ModelHyperparameters
    training_hparams: TrainingHyperparameters

    cortical_column: CorticalColumn

    update_w_l1_l1: UpdateRule
    update_k_l2_l2: UpdateRule
    update_a_l2_l2: UpdateRule
    update_k_l3_l3: UpdateRule
    update_a_l3_l3: UpdateRule
    update_w_l2_l3: UpdateRule

    def __init__(
        self,
        cortical_column_hparams: CorticalColumnHyperparameters,
        model_hparams: ModelHyperparameters,
        training_hparams: TrainingHyperparameters,
        dt: float,
    ):
        self.model_hparams = model_hparams
        self.training_hparams = training_hparams
        self.cortical_column = CorticalColumn(cortical_column_hparams, dt)
        self.update_w_l1_l1 = UpdateRule(
            SynapseType.W,
            training_hparams.gamma_w,
            cortical_column_hparams.e0,
            training_hparams.theta_low1,
            None,
            training_hparams.w_l1_l1_max,
            training_hparams.w_max_sum,
        )
        self.update_k_l2_l2 = UpdateRule(
            SynapseType.K,
            training_hparams.gamma_k,
            cortical_column_hparams.e0,
            training_hparams.theta_low2,
            None,
            training_hparams.k_l2_l2_max,
            training_hparams.k_max_sum,
        )
        self.update_a_l2_l2 = UpdateRule(
            SynapseType.A,
            training_hparams.gamma_a,
            cortical_column_hparams.e0,
            training_hparams.theta_low2,
            training_hparams.theta_high2,
            training_hparams.a_l2_l2_max,
            None,
        )
        self.update_k_l3_l3 = UpdateRule(
            SynapseType.K,
            training_hparams.gamma_k,
            cortical_column_hparams.e0,
            training_hparams.theta_low2,
            None,
            training_hparams.k_l3_l3_max,
            training_hparams.k_max_sum,
        )
        self.update_a_l3_l3 = UpdateRule(
            SynapseType.A,
            training_hparams.gamma_a,
            cortical_column_hparams.e0,
            training_hparams.theta_low2,
            training_hparams.theta_high2,
            training_hparams.a_l3_l3_max,
            None,
        )
        self.update_w_l2_l3 = UpdateRule(
            SynapseType.Wb,
            training_hparams.gamma_wb,
            cortical_column_hparams.e0,
            training_hparams.theta_low3,
            None,
            training_hparams.w_l2_l3_max,
            None,
        )

    def init_params(self, like: jax.Array) -> ModelParameters:
        n = jnp.size(like)
        m = jnp.zeros((n, n))
        return ModelParameters(*[m] * 6)

    def init_state(self, like: jax.Array) -> ModelState:
        s = self.cortical_column.init_state(like)
        return ModelState(*[s] * 4)

    @eqx.filter_jit
    def __call__(
        self,
        params: ModelParameters,
        state: ModelState,
        input: jax.Array,
        mask: jax.Array,
        key: jax.Array,
    ) -> tuple[ModelParameters, ModelState]:
        wm_state, l1_state, l2_state, l3_state = state
        y_wm, y_l1, y_l2, y_l3 = [layer.pyramidal_synapse.psp for layer in state]
        z_l1, z_l2, z_l3 = [layer.pyramidal_firing_rate for layer in state[1:]]

        zero = jnp.zeros_like(input)
        key_wm, key_l1, key_l2, key_l3 = jax.random.split(key, 4)

        # working memory layer
        e_wm = self.model_hparams.w_wm_l1 * y_l1
        i_wm = zero
        new_wm_state = self.cortical_column(
            wm_state,
            e_wm,
            i_wm,
            mask[0][0] * input,
            mask[0][1] * input,
            key_wm,
            layer_wm=True,
        )

        # layer 1
        e_l1 = self.model_hparams.w_l1_wm * y_wm + params.w_l1_l1 @ y_l1
        i_l1 = zero
        new_l1_state = self.cortical_column(
            l1_state, e_l1, i_l1, mask[1][0] * input, mask[1][1] * input, key_l1
        )

        # layer 2
        inhibitor = self.model_hparams.r * jax.nn.relu(
            self.model_hparams.t - z_l1.sum()
        )
        e_l2 = self.model_hparams.w_l2_l1 * y_l1 + params.w_l2_l3 @ y_l3
        i_l2 = params.k_l2_l2 @ y_l2 + params.a_l2_l2 @ z_l2 + inhibitor
        new_l2_state = self.cortical_column(
            l2_state, e_l2, i_l2, mask[2][0] * input, mask[2][1] * input, key_l2
        )

        # layer 3
        e_l3 = self.model_hparams.w_l3_l2 * y_l2
        i_l3 = params.k_l3_l3 @ y_l3 + params.a_l3_l3 @ z_l3
        new_l3_state = self.cortical_column(
            l3_state, e_l3, i_l3, mask[3][0] * input, mask[3][1] * input, key_l3
        )

        # update parameters
        w_l1_l1 = self.update_w_l1_l1(params.w_l1_l1, z_l1, z_l1)
        k_l2_l2 = self.update_k_l2_l2(params.k_l2_l2, z_l2, z_l2)
        a_l2_l2 = self.update_a_l2_l2(params.a_l2_l2, z_l2, z_l2)
        k_l3_l3 = self.update_k_l3_l3(params.k_l3_l3, z_l3, z_l3)
        a_l3_l3 = self.update_a_l3_l3(params.a_l3_l3, z_l3, z_l3)
        w_l2_l3 = self.update_w_l2_l3(params.w_l2_l3, z_l2, z_l3)

        new_parameters = ModelParameters(
            w_l1_l1, k_l2_l2, a_l2_l2, k_l3_l3, a_l3_l3, w_l2_l3
        )
        new_state = ModelState(new_wm_state, new_l1_state, new_l2_state, new_l3_state)
        return new_parameters, new_state

import equinox as eqx
import jax
import jax.numpy as jnp
from functools import partial
from .model import ModelParameters
from typing import NamedTuple, Optional, Literal


class TrainingHyperparameters(NamedTuple):
    e0: float
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
    w_max_sum: Optional[float]
    k_max_sum: Optional[float]
    a_max_sum: Optional[float]


class UpdateRule(eqx.Module):
    """Hebbian / anti-hebbian update rule for one group of long range synapses"""

    e0: float
    gamma: float
    theta_low: float
    theta_high: Optional[float]
    weights_max: float
    max_sum: Optional[float]
    hebbian: bool
    normalization: bool

    @partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        weights: jax.Array,
        target_firing_rates: jax.Array,
        source_firing_rates: jax.Array,
    ) -> jax.Array:
        """Update weights based on firing rates."""
        # hebbian / anti-hebbian rule
        v = jax.nn.relu(target_firing_rates / (2 * self.e0) - self.theta_low)
        w = jax.lax.select(
            self.hebbian,
            jax.nn.relu(self.theta_high - source_firing_rates / (2 * self.e0)),
            jax.nn.relu(source_firing_rates / (2 * self.e0) - self.theta_low),
        )

        delta_weights = self.gamma * jnp.outer(v, w) * (self.weights_max - weights)
        weights += delta_weights

        # normalization to maximum
        if self.normalization:
            # get sum of weights for each row
            sum_weights = jnp.sum(weights, axis=1, keepdims=True)

            # if max_sum is undefined, then use min(sum_weights)
            max_sum = sum_weights.min() if self.max_sum is None else self.max_sum

            # apply normalization
            weights = jnp.where(
                sum_weights > max_sum, weights * max_sum / sum_weights, weights
            )

        return weights


class Trainer(eqx.Module):
    """This class trains our model parameters using hebbian and anti-hebbian update rules."""

    hparams: TrainingHyperparameters

    update_w_l1_l1: UpdateRule
    update_k_l2_l2: UpdateRule
    update_a_l2_l2: UpdateRule
    update_k_l3_l3: UpdateRule
    update_a_l3_l3: UpdateRule
    update_w_l2_l3: UpdateRule

    def __init__(self, hparams: TrainingHyperparameters):
        self.hparams = hparams

        # helper function for constructing UpdateRules
        def update_rule(
            synapse_type: Literal["W", "K", "A"],
            source_layer: int,
            weights_max: float,
            normalization: bool = True,
        ) -> UpdateRule:
            e0 = hparams.e0
            gamma, max_sum, hebbian = {
                "W": (hparams.gamma_w, hparams.w_max_sum, True),
                "K": (hparams.gamma_k, hparams.k_max_sum, True),
                "A": (hparams.gamma_a, hparams.a_max_sum, False),
            }[synapse_type]
            theta_low = {
                1: hparams.theta_low1,
                2: hparams.theta_low2,
                3: hparams.theta_low3,
            }[source_layer]
            theta_high = None if hebbian else hparams.theta_high2
            return UpdateRule(
                e0,
                gamma,
                theta_low,
                theta_high,
                weights_max,
                max_sum,
                hebbian,
                normalization,
            )

        self.update_w_l1_l1 = update_rule("W", 1, hparams.w_l1_l1_max)
        self.update_k_l2_l2 = update_rule("K", 2, hparams.k_l2_l2_max)
        self.update_a_l2_l2 = update_rule("A", 2, hparams.a_l2_l2_max)
        self.update_k_l3_l3 = update_rule("K", 3, hparams.k_l3_l3_max)
        self.update_a_l3_l3 = update_rule("A", 3, hparams.a_l3_l3_max)
        self.update_w_l2_l3 = update_rule("W", 3, hparams.w_l2_l3_max, False)

    @partial(jax.jit, static_argnames=["self"])
    def step1(
        self, params: ModelParameters, firing_rates: jax.Array
    ) -> ModelParameters:
        """Applys the update rule for step 1 of training."""
        w_l1_l1, k_l2_l2, a_l2_l2, k_l3_l3, a_l3_l3, w_l2_l3 = params
        _, l1, l2, l3 = firing_rates

        new_w_l1_l1 = self.update_w_l1_l1(w_l1_l1, l1, l1)
        new_k_l2_l2 = self.update_k_l2_l2(k_l2_l2, l2, l2)
        new_a_l2_l2 = self.update_a_l2_l2(a_l2_l2, l2, l2)
        new_k_l3_l3 = self.update_k_l3_l3(k_l3_l3, l3, l3)
        new_a_l3_l3 = self.update_a_l3_l3(a_l3_l3, l3, l3)

        new_params = ModelParameters(
            new_w_l1_l1,
            new_k_l2_l2,
            new_a_l2_l2,
            new_k_l3_l3,
            new_a_l3_l3,
            w_l2_l3,
        )

        return new_params

    @partial(jax.jit, static_argnames=["self"])
    def step2(
        self, params: ModelParameters, firing_rates: jax.Array
    ) -> ModelParameters:
        """Applys the update rule for step 2 of training.
        This step is only used in the "sequence-ordering working memory" modality.
        """
        w_l2_l3 = params.w_l2_l3
        _, _, l2, l3 = firing_rates

        new_w_l2_l3 = self.update_w_l2_l3(w_l2_l3, l2, l3)

        new_params = params._replace(w_l2_l3=new_w_l2_l3)

        return new_params

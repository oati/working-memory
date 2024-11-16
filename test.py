import jax
import jax.numpy as jnp

from model import (
    Model,
    ModelHyperparameters,
    ModelParameters,
    ModelState,
    TrainingHyperparameters,
)
from model.cortical_column import CorticalColumn, CorticalColumnHyperparameters
from model.synapse import Synapse
from model.hyperparameters import (
    dt,
    cortical_column_hyperparameters,
    sequence_model_hyperparameters,
    semantic_model_hyperparameters,
    training_hyperparameters,
    training_mask_1,
    training_mask_2,
    pattern_intensity,
)


def simulate_model(model, params, state, steps, dt, key):
    yield params, state
    for _ in range(steps):
        key, new_key = jax.random.split(key)
        params, state = model(params, state, jnp.array([1]), jnp.ones((2, 3)), new_key)
        yield params, state


def filter_data(params, state):
    return {
        "parameters": {
            "w_l1_l1": params.w_l1_l1,
            "k_l2_l2": params.k_l2_l2,
            "a_l2_l2": params.a_l2_l2,
            "k_l3_l3": params.k_l3_l3,
            "a_l3_l3": params.a_l3_l3,
            "w_l2_l3": params.w_l2_l3,
        },
        "pyramidal_firing_rates": {
            "wm_layer": state.wm_layer.pyramidal_firing_rate,
            "layer1": state.layer1.pyramidal_firing_rate,
            "layer2": state.layer2.pyramidal_firing_rate,
            "layer3": state.layer3.pyramidal_firing_rate,
        },
    }


def model_data(*args):
    return jax.tree.map(
        lambda *x: jnp.stack(x, axis=-1),
        *(filter_data(params, state) for params, state in simulate_model(*args))
    )


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    model = Model(
        cortical_column_hyperparameters,
        sequence_model_hyperparameters,
        training_hyperparameters,
        dt,
    )
    # column = CorticalColumn(cortical_column_hyperparameters, dt)

    params = model.init_params(jnp.array([0]))
    state = model.init_state(jnp.array([0]))
    # state = column.init_state(jnp.array(0))

    # t = jnp.arange(0, 1, dt)
    # data = model_data(model, params, state, 1, dt, key)
    data = model_data(model, params, state, 100, dt, key)
    print(data["pyramidal_firing_rates"]["wm_layer"])

import jax
import jax.numpy as jnp

from model import Model, ModelHyperparameters, ModelParameters
from model.training import TrainingHyperparameters

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


def simulate_model(model, state, params, steps, dt, key):
    for _ in range(steps):
        key, new_key = jax.random.split(key)
        state, pfr = model(state, params, jnp.ones((4,1)), jnp.ones((4,1)), new_key)
        yield pfr

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    model = Model(
        cortical_column_hyperparameters,
        semantic_model_hyperparameters,
        dt,
    )
    # column = CorticalColumn(cortical_column_hyperparameters, dt)

    params = model.init_params(jnp.array([0]))
    state = model.init_state(jnp.array([0]))
    # state = column.init_state(jnp.array(0))

    # t = jnp.arange(0, 1, dt)
    data = list(simulate_model(model, state, params, 100, dt, key))
    print(data)

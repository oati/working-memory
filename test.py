import jax
import jax.numpy as jnp
from model import (
    Model,
    ModelHyperparameters,
    ModelParameters,
    ModelState,
    TrainingHyperparameters,
)
from cortical_column import CorticalColumn, CorticalColumnHyperparameters
from synapse import Synapse
from hyperparameters import (
    dt,
    cortical_column_hyperparameters,
    sequence_model_hyperparameters,
    semantic_model_hyperparameters,
    training_hyperparameters,
    training_mask_1,
    training_mask_2,
    pattern_intensity,
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

    t = 0.0
    while t < 1:
        t += dt
        key, new_key = jax.random.split(key)
        # state = column(
        #     state, jnp.array(10), jnp.array(0), jnp.array(0), jnp.array(0), new_key
        # )
        params, state = model(params, state, jnp.array([0]), jnp.zeros((2, 3)), new_key)
    print(state)

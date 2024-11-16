import asyncio
import edifice as ed  # type: ignore[import]

import jax
import jax.np as jnp

from edifice.extra.matplotlib_figure import MatplotlibFigure
import matplotlib.pyplot as plt

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


@ed.component
def Application(self):
    with ed.View(layout="column"):
        with ed.View(layout="row"):
            ed.Button("Simulate")

@ed.component
def Main(self):
    with ed.Window("Edifice Application"):
        with ed.View():
            Application()

if __name__ == "__main__":
    ed.App(Main()).start()

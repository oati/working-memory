from pathlib import Path
import jax.numpy as jnp


dir = Path(__file__).parent

# create a global variable for each .npy file
for path in dir.glob("*.npy"):
    globals()[path.stem] = jnp.load(path)

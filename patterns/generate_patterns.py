from pathlib import Path
import jax
import jax.numpy as jnp


def string_to_pattern(string: str) -> jax.Array:
    """Convert a string of 1-indexed slices into an array representing a pattern."""
    pattern = jnp.zeros((400,))
    for slice in string.split():
        a, b = slice.split(":")
        pattern = pattern.at[int(a) - 1 : int(b)].set(1)
    return pattern


if __name__ == "__main__":
    dir = Path(__file__).parent

    # save each collection as a .npy file
    for path in dir.glob("*.txt"):
        with open(path, "r") as file:
            patterns = jnp.array([string_to_pattern(line) for line in file.readlines()])
            jnp.save(dir / path.stem, patterns)

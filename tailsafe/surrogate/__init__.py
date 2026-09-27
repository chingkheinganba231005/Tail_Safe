"""GNN surrogate for instant what-if estimates."""

from pathlib import Path

# The shipped weights (``.npz``) and their description (``.json``). Defined here,
# without importing JAX, so tools that only read the files do not need it.
DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "surrogate.npz"

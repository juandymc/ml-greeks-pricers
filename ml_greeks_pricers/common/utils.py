from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any


def save_variables(path: str | Path, **variables: Any) -> None:
    """Save the given variables dictionary to a pickle file.

    Parameters
    ----------
    path: str or Path
        Destination file path. Parent directories are created if needed.
    **variables: Any
        Variables to serialize. They will be stored under their given names.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(variables, fh)


def load_variables(path: str | Path, *names: str) -> Any:
    """Load variables from a pickle file.

    Parameters
    ----------
    path: str or Path
        Path to the pickle file.
    *names: str
        Optional names of variables to return. If omitted the entire
        dictionary is returned.
    """
    p = Path(path)
    with p.open("rb") as fh:
        data = pickle.load(fh)

    if names:
        return tuple(data[name] for name in names)
    return data

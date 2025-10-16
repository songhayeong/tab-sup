from pathlib import Path
from typing import Union

PROJ = Path(__file__).resolve().parent


def get_path(path: Union[str, Path]) -> Path:
    """
    Convert a possibly relative path into an absolute path rooted at the project.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJ / path
    return path


def get_relative_path(path: Union[str, Path]) -> Path:
    """
    Provide project-relative paths for logging/checkpoint helpers.
    """
    path = Path(path)
    return path if path.is_relative_to(PROJ) else path.relative_to(PROJ)

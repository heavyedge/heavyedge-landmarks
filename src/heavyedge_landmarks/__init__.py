"""Package to locate landmarks from edge profiles."""

from .mathematical import landmarks_type2
from .pseudo import pseudo_landmarks

__all__ = [
    "pseudo_landmarks",
    "landmarks_type2",
]

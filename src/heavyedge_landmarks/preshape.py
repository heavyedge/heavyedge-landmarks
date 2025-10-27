"""Converts configuration matrices to pre-shapes.

.. note::

    Helmert sub-matrices are LRU-cached.
    The number of most recent calls can be set by the environment variable
    `HEAVYEDGE_LANDMARKS_CACHE_SIZE`, which defaults to 4.
"""

import os
from functools import lru_cache

import numpy as np
from scipy.linalg import helmert

__all__ = [
    "preshape",
    "dual_preshape",
]


def preshape(Xs):
    """Convert configuration matrices to pre-shapes.

    Conversion is done using the Helmert sub-matrix.

    Parameters
    ----------
    Xs : array, shape (N, m, k)
        `N` configuration matrices of `k` landmarks in dimension `m`.

    Returns
    -------
    Zs : array, shape (N, m, k-1)
        `N` pre-shape matrices.

    See Also
    --------
    dual_preshape
        Pre-shape in configuration matrix space.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_landmarks import pseudo_landmarks, preshape
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> Zs = preshape(pseudo_landmarks(x, Ys, Ls, 10))
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(*Zs.transpose(1, 2, 0))
    """
    _, _, k = Xs.shape
    H = _helmert(k)
    HX = np.inner(Xs, H)
    scale = np.linalg.norm(HX, axis=(1, 2), keepdims=True)
    Zs = HX / scale
    return Zs


def dual_preshape(Xs):
    """Pre-shape in configuration matrix space.

    Conversion is done using the Helmert sub-matrix and its hat matrix.

    Parameters
    ----------
    Xs : array, shape (N, m, k)
        `N` configuration matrices of `k` landmarks in dimension `m`.

    Returns
    -------
    Zs : array, shape (N, m, k)
        `N` pre-shape matrices.

    See Also
    --------
    preshape
        Pre-shape in its original space.

    Notes
    -----
    Because location and scale information is lost during the pre-shaping process,
    *Zs* is rank-deficient and has unit norm.

    Examples
    --------
    >>> from heavyedge import get_sample_path, ProfileData
    >>> from heavyedge_landmarks import pseudo_landmarks, dual_preshape
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> Zs = dual_preshape(pseudo_landmarks(x, Ys, Ls, 10))
    >>> import matplotlib.pyplot as plt  # doctest: +SKIP
    ... plt.plot(*Zs.transpose(1, 2, 0))
    """
    _, _, k = Xs.shape
    H = _helmert(k)
    HX = np.inner(Xs, H)
    scale = np.linalg.norm(HX, axis=(1, 2), keepdims=True)
    Zs = HX / scale
    hat = _helmert_hat(k)
    return np.inner(Zs, hat)


CACHE_SIZE = os.environ.get("HEAVYEDGE_LANDMARKS_CACHE_SIZE")
if CACHE_SIZE is not None:
    CACHE_SIZE = int(CACHE_SIZE)
else:
    CACHE_SIZE = 4


@lru_cache(maxsize=CACHE_SIZE)
def _helmert(k):
    return helmert(k)


@lru_cache(maxsize=CACHE_SIZE)
def _helmert_hat(k):
    H = _helmert(k)
    return H.T @ np.linalg.inv(H @ H.T)

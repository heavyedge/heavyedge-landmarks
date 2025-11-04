.. HeavyEdge-Landmarks documentation master file, created by
   sphinx-quickstart on Wed Oct 15 09:22:36 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*********************************
HeavyEdge-Landmarks documentation
*********************************

.. module:: heavyedge_landmarks

.. plot:: plot-header.py
    :include-source: False

HeavyEdge-Landmarks is a Python package for locating landmarks from coating edge profiles and converting to pre-shapes.

=========
Tutorials
=========

Detecting landmarks requires profiles and lengths of each profile.
Here, we use preprocessed data distributed by :mod:`heavyedge` package.

.. plot::
    :context:

    >>> from heavyedge import get_sample_path, ProfileData
    >>> with ProfileData(get_sample_path("Prep-Type1.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> import matplotlib.pyplot as plt
    ... plt.plot(x, Ys.T)

Use :func:`pseudo_landmarks` to locate landmarks by equidistant sampling.
You need to specify the number of points `k` to sample.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import pseudo_landmarks
    >>> k = 10  # Number of landmarks
    >>> lm = pseudo_landmarks(x, Ys, Ls, k)
    >>> import matplotlib.pyplot as plt
    ... plt.plot(*lm.transpose(1, 2, 0))

Use :func:`landmarks_type2` to locate feature points as landmarks, assuming Type 2 shape which has heavy edge peak but no trough.
You need to specify the standad deviation `sigma` of Gaussian kernel for the function to internally smooth noises.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import landmarks_type2
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> sigma = 32  # Gaussian kernel std for noise smoothing
    >>> lm = landmarks_type2(x, Ys, Ls, sigma)
    >>> plt.plot(*lm.transpose(1, 2, 0))

Use :func:`landmarks_type3` to locate feature points as landmarks, assuming Type 2 shape which has heavy edge peak and trough.
Like :func:`landmarks_type2`, you need to specify `sigma`.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import landmarks_type3
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    ...     x = data.x()
    ...     Ys, Ls, _ = data[:]
    >>> sigma = 32  # Gaussian kernel std for noise smoothing
    >>> lm = landmarks_type3(x, Ys, Ls, sigma)
    >>> plt.plot(*lm.transpose(1, 2, 0))

=============
How-to Guides
=============

Determining the sigma value
---------------------------

What representation to use
--------------------------

Landmarks vs pre-shape vs dual pre-shape

Scaling the data
----------------

==========
Module API
==========

Landmark detection
==================

Acquires configuration matrices of pseudo-landmarks and mathematical landmarks.

.. autofunction:: pseudo_landmarks

.. autofunction:: landmarks_type2

.. autofunction:: landmarks_type3

Pre-shape conversion
====================

.. automodule:: heavyedge_landmarks.preshape
    :members:

Plataeu fitting
===============

.. automodule:: heavyedge_landmarks.plateau
    :members:

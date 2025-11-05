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

This section provides basic tutorials for beginners.

Preparing data
==============

Detecting landmarks requires profile and length data.
Here, we use preprocessed data distributed by :mod:`heavyedge` package.

.. plot::
    :context:

    >>> from heavyedge import get_sample_path, ProfileData
    >>> with ProfileData(get_sample_path("Prep-Type1.h5")) as data:
    ...     x1 = data.x()
    ...     Ys1, Ls1, _ = data[:]
    >>> with ProfileData(get_sample_path("Prep-Type2.h5")) as data:
    ...     x2 = data.x()
    ...     Ys2, Ls2, _ = data[:]
    >>> with ProfileData(get_sample_path("Prep-Type3.h5")) as data:
    ...     x3 = data.x()
    ...     Ys3, Ls3, _ = data[:]
    >>> import matplotlib.pyplot as plt
    ... plt.plot(x1, Ys1.T)
    ... plt.plot(x2, Ys2.T)
    ... plt.plot(x3, Ys3.T)

Locating landmarks
==================

Use :func:`pseudo_landmarks` to locate landmarks by equidistant sampling.
You need to specify the number of points `k` to sample.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import pseudo_landmarks
    >>> k = 10  # Number of landmarks
    >>> lm1 = pseudo_landmarks(x1, Ys1, Ls1, k)
    >>> import matplotlib.pyplot as plt
    ... plt.plot(x1, Ys1.T, color="gray", alpha=0.5)
    ... plt.plot(*lm1.transpose(1, 2, 0))

Use :func:`landmarks_type2` to locate feature points as landmarks, assuming Type 2 shape which has a heavy edge peak but no trough.
You need to specify the standard deviation `sigma` of the Gaussian kernel for the function to internally smooth noise.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import landmarks_type2
    >>> sigma = 32  # Gaussian kernel std for noise smoothing
    >>> lm2 = landmarks_type2(x2, Ys2, Ls2, sigma)
    >>> plt.plot(x2, Ys2.T, color="gray", alpha=0.5)
    ... plt.plot(*lm2.transpose(1, 2, 0))

Use :func:`landmarks_type3` to locate feature points as landmarks, assuming Type 3 shape which has a heavy edge peak and trough.
Like :func:`landmarks_type2`, you need to specify `sigma`.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import landmarks_type3
    >>> sigma = 32  # Gaussian kernel std for noise smoothing
    >>> lm3 = landmarks_type3(x3, Ys3, Ls3, sigma)
    >>> plt.plot(x3, Ys3.T, color="gray", alpha=0.5)
    ... plt.plot(*lm3.transpose(1, 2, 0))

Transforming to pre-shapes
==========================

In statistical shape analysis, a matrix of landmark coordinates from an object is called the *configuration matrix*.
Configuration matrices can be transformed to *pre-shapes*, where location and size information is removed, using :func:`preshape`.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import preshape
    >>> ps3 = preshape(lm3)
    >>> plt.plot(*ps3.transpose(1, 2, 0))

Pre-shapes exist in a different space from configuration matrices.
If you want to represent your pre-shape in the original space, use :func:`dual_preshape`.
Note that pre-shapes in the original space are rank-deficient.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import dual_preshape
    >>> dual_ps3 = dual_preshape(lm3)
    >>> plt.plot(*dual_ps3.transpose(1, 2, 0))

Fitting the plateau
===================

Plateaus detected by landmarks can be severely affected by noise or data artifacts.
For Type 2 profiles, :func:`plateau_type2` can be used for more robust plateau detection through nonlinear regression.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import plateau_type2
    >>> peaks, knees = lm2[:, 0, 1:].T
    >>> y0, m, xlast = plateau_type2(x2, Ys2, peaks, knees).T
    >>> x = np.stack([np.zeros(len(xlast)), xlast])
    >>> y = y0 + m * x
    >>> plt.plot(x2, Ys2.T, color="gray", alpha=0.5)
    ... plt.plot(x, y)

Likewise, :func:`plateau_type3` can be used for Type 3 profiles.

.. plot::
    :context: close-figs

    >>> from heavyedge_landmarks import plateau_type3
    >>> troughs, knees = lm3[:, 0, 2:].T
    >>> y0, m, xlast = plateau_type3(x3, Ys3, troughs, knees).T
    >>> x = np.stack([np.zeros(len(xlast)), xlast])
    >>> y = y0 + m * x
    >>> plt.plot(x3, Ys3.T, color="gray", alpha=0.5)
    ... plt.plot(x, y)


=============
How-to Guides
=============

Determining the sigma value
===========================

What representation to use
==========================

Landmarks vs pre-shape vs dual pre-shape

Scaling the data
================

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

Plateau fitting
===============

.. automodule:: heavyedge_landmarks.plateau
    :members:

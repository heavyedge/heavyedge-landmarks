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

=============
How-to Guides
=============

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

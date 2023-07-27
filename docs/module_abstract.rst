Geometry Base
^^^^^^^^^^^^^

``abstract`` module provides base classes for parametric curves, surfaces and volumes contained in this library and
therefore, it provides an easy way to extend the library in the most proper way.

Inheritance Diagram
===================

.. inheritance-diagram:: nurbs.abstract
    :top-classes: nurbs.abstract.nurbsBase

Abstract Curve
==============

.. autoclass:: nurbs.abstract.Curve
    :members:
    :inherited-members:
    :exclude-members: next
    :show-inheritance:

Abstract Surface
================

.. autoclass:: nurbs.abstract.Surface
    :members:
    :inherited-members:
    :exclude-members: next
    :show-inheritance:

Abstract Volume
===============

.. autoclass:: nurbs.abstract.Volume
    :members:
    :inherited-members:
    :exclude-members: next
    :show-inheritance:

Low Level API
=============

The following classes provide the low level API for the geometry abstract base.

* :py:class:`.nurbsBase`
* :py:class:`.Geometry`
* :py:class:`.SplineGeometry`

:py:class:`.Geometry` abstract base class can be used for implementation of any geometry object, whereas
:py:class:`.SplineGeometry` abstract base class is designed specifically for spline geometries, including basis splines.

.. autoclass:: nurbs.abstract.nurbsBase
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: nurbs.abstract.Geometry
    :members:
    :inherited-members:
    :exclude-members: next
    :show-inheritance:

.. autoclass:: nurbs.abstract.SplineGeometry
    :members:
    :inherited-members:
    :exclude-members: next
    :show-inheritance:

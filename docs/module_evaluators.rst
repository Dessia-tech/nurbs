Evaluators
^^^^^^^^^^

Evaluators allow users to change the evaluation algorithms that are used to evaluate curves, surfaces and volumes,
take derivatives and more. All geometry classes set an evaluator by default. Users may switch between the evaluation
algorithms at runtime. It is also possible to implement different algorithms (e.g. T-splines) or extend existing ones.

How to Use
==========

All geometry classes come with a default specialized ``evaluator`` class, the algorithms are generally different for
rational and non-rational geometries. The evaluator class instance can be accessed and/or updated using ``evaluator``
property. For instance, the following code snippet changes the evaluator of a B-Spline curve.

.. code-block:: python

    from nurbs import BSpline
    from nurbs import evaluators

    crv = BSpline.Curve()
    cevaltr = evaluators.CurveEvaluator2()
    crv.evaluator = cevaltr

    # Curve "evaluate" method will use CurveEvaluator2.evaluate() method
    crv.evaluate()

    # Get evaluated points
    curve_points = crv.evalpts


Implementing Evaluators
=======================

All evaluators should be extended from :class:`.evaluators.AbstractEvaluator` abstract base class. This class provides
a point evaluation and a derivative computation methods. Both methods take a *data* input which contains the geometry
data as a *dict* object (refer to :attr:`.BSpline.Surface.data` property as an example). The derivative computation
method also takes additional arguments, such as the parametric position and the derivative order.

Inheritance Diagram
===================

.. inheritance-diagram:: nurbs.evaluators

Abstract Base
=============

.. autoclass:: nurbs.evaluators.AbstractEvaluator
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Curve Evaluators
================

.. autoclass:: nurbs.evaluators.CurveEvaluator
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: nurbs.evaluators.CurveEvaluatorRational
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: nurbs.evaluators.CurveEvaluator2
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Surface Evaluators
==================

.. autoclass:: nurbs.evaluators.SurfaceEvaluator
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: nurbs.evaluators.SurfaceEvaluatorRational
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: nurbs.evaluators.SurfaceEvaluator2
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

Volume Evaluators
=================

.. autoclass:: nurbs.evaluators.VolumeEvaluator
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: nurbs.evaluators.VolumeEvaluatorRational
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

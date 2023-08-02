# cython: language_level=3
"""
.. module:: evaluators
    :platform: Unix, Windows
    :synopsis: Provides spline evaluator classes

.. moduleauthor:: Onur Rauf Bingol <orbingol@gmail.com>

"""

import abc
import copy

from nurbs import _utilities as utl
from nurbs import helpers, linalg

import cython
import numpy as np


@utl.add_metaclass(abc.ABCMeta)
class AbstractEvaluator(object):
    """Abstract base class for implementations of fundamental spline algorithms, such as evaluate and derivative.

    **Abstract Methods**:

    * ``evaluate`` is used for computation of the complete spline shape
    * ``derivative_single`` is used for computation of derivatives at a single parametric coordinate

    Please note that this class requires the keyword argument ``find_span_func`` to be set to a valid find_span
    function implementation. Please see :py:mod:`helpers` module for details.
    """

    def __init__(self, **kwargs):
        self._name = kwargs.get("name", self.__class__.__name__)
        self._span_func = kwargs.get("find_span_func", None)

    @property
    def name(self):
        """Evaluator name.

        :getter: Gets the name of the evaluator
        :type: str
        """
        return self._name

    @abc.abstractmethod
    def evaluate(self, datadict, **kwargs):
        """Abstract method for evaluation of points on the spline geometry.

        .. note::

            This is an abstract method and it must be implemented in the subclass.

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        """
        pass

    @abc.abstractmethod
    def derivatives(self, datadict, parpos, deriv_order=0, **kwargs):
        """Abstract method for evaluation of the n-th order derivatives at the input parametric position.

        .. note::

            This is an abstract method and it must be implemented in the subclass.

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :param parpos: parametric position where the derivatives will be computed
        :type parpos: list, tuple
        :param deriv_order: derivative order; to get the i-th derivative
        :type deriv_order: int
        """
        pass


@utl.export
class CurveEvaluator(AbstractEvaluator):
    """Sequential curve evaluation algorithms.

    This evaluator implements the following algorithms from **The NURBS Book**:

    * Algorithm A3.1: CurvePoint
    * Algorithm A3.2: CurveDerivsAlg1

    Please note that knot vector span finding function may be changed by setting ``find_span_func`` keyword argument
    during the initialization. By default, this function is set to :py:func:`.helpers.find_span_linear`.
    Please see :doc:`Helpers Module Documentation <module_utilities>` for more details.
    """

    def __init__(self, **kwargs):
        super(CurveEvaluator, self).__init__(**kwargs)
        self._span_func = kwargs.get("find_span_func", helpers.find_span_linear)

    def evaluate(self, datadict, **kwargs):
        """Evaluates the curve.

        Keyword Arguments:
            * ``start``: starting parametric position for evaluation
            * ``stop``: ending parametric position for evaluation

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :return: evaluated points
        :rtype: list
        """
        # Geometry data from datadict
        cdef int degree = datadict["degree"][0]
        cdef list knotvector = datadict["knotvector"][0]
        cdef tuple ctrlpts = datadict["control_points"]
        cdef int size = datadict["size"][0]
        cdef int sample_size = datadict["sample_size"][0]
        cdef int dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]
        cdef int precision = datadict["precision"]

        # Keyword arguments
        cdef double start = kwargs.get("start", 0.0)
        cdef double stop = kwargs.get("stop", 1.0)

        # Algorithm A3.1
        cdef list knots = linalg.linspace(start, stop, sample_size, decimals=precision)
        cdef list spans = helpers.find_spans(degree, knotvector, size, knots, self._span_func)
        cdef list basis = helpers.basis_functions(degree, knotvector, spans, knots)

        cdef list eval_points = []
        cdef i, idx
        cdef list crvpt
        cdef double crv_p, ctl_p
        for idx in range(len(knots)):
            crvpt = [0.0 for _ in range(dimension)]
            for i in range(0, degree + 1):
                crvpt[:] = [
                    crv_p + (basis[idx][i] * ctl_p) for crv_p, ctl_p in zip(crvpt, ctrlpts[spans[idx] - degree + i])
                ]

            eval_points.append(crvpt)

        return eval_points

    def derivatives(self, int degree, list knotvector, list ctrlpts, int size, int dimension, double parpos,
                    int deriv_order):
        """Evaluates the n-th order derivatives at the input parametric position.

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :param parpos: parametric position where the derivatives will be computed
        :type parpos: list, tuple
        :param deriv_order: derivative order; to get the i-th derivative
        :type deriv_order: int
        :return: evaluated derivatives
        :rtype: list
        """
        # Geometry data from datadict
        # cdef int degree = datadict["degree"][0]
        # cdef list knotvector = datadict["knotvector"][0]
        # cdef tuple ctrlpts = datadict["control_points"]
        # cdef int size = datadict["size"][0]
        # cdef int dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]

        # Algorithm A3.2
        cdef int du = min(degree, deriv_order)

        cdef list CK = [[0.0 for _ in range(dimension)] for _ in range(deriv_order + 1)]

        cdef int span = self._span_func(degree, knotvector, size, parpos)
        cdef list bfunsders = helpers.basis_function_ders(degree, knotvector, span, parpos, du)

        cdef int k, j
        cdef double drv, ctl_pt
        for k in range(0, du + 1):
            for j in range(0, degree + 1):
                CK[k][:] = [drv + (bfunsders[k][j] * ctl_pt) for drv, ctl_pt in zip(CK[k], ctrlpts[span - degree + j])]

        # Return the derivatives
        return CK


@utl.export
class CurveEvaluatorRational(CurveEvaluator):
    """Sequential rational curve evaluation algorithms.

    This evaluator implements the following algorithms from **The NURBS Book**:

    * Algorithm A3.1: CurvePoint
    * Algorithm A4.2: RatCurveDerivs

    Please note that knot vector span finding function may be changed by setting ``find_span_func`` keyword argument
    during the initialization. By default, this function is set to :py:func:`.helpers.find_span_linear`.
    Please see :doc:`Helpers Module Documentation <module_utilities>` for more details.
    """

    def __init__(self, **kwargs):
        super(CurveEvaluatorRational, self).__init__(**kwargs)
        self._span_func = kwargs.get("find_span_func", helpers.find_span_linear)

    def evaluate(self, datadict, **kwargs):
        """Evaluates the rational curve.

        Keyword Arguments:
            * ``start``: starting parametric position for evaluation
            * ``stop``: ending parametric position for evaluation

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :return: evaluated points
        :rtype: list
        """
        cdef int dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]

        # Algorithm A4.1
        cdef list crvptw = super(CurveEvaluatorRational, self).evaluate(datadict, **kwargs)

        # Divide by weight
        cdef list eval_points = []
        cdef list pt, cpt
        for pt in crvptw:
            cpt = [float(c / pt[-1]) for c in pt[0 : (dimension - 1)]]
            eval_points.append(cpt)

        return eval_points

    def derivatives(self, int degree, list knotvector, list ctrlpts, int size, int dimension, double parpos,
                    int deriv_order):
        """Evaluates the n-th order derivatives at the input parametric position.

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :param parpos: parametric position where the derivatives will be computed
        :type parpos: list, tuple
        :param deriv_order: derivative order; to get the i-th derivative
        :type deriv_order: int
        :return: evaluated derivatives
        :rtype: list
        """

        # Call the parent function to evaluate A(u) and w(u) derivatives
        cdef list CKw = super(CurveEvaluatorRational, self).derivatives(degree, knotvector, ctrlpts, size,
                                                                        dimension, parpos, deriv_order)

        # Algorithm A4.2
        cdef list CK = [[0.0 for _ in range(dimension - 1)] for _ in range(deriv_order + 1)]
        cdef int k
        cdef list v

        for k in range(0, deriv_order + 1):
            v = [val for val in CKw[k][0 : (dimension - 1)]]
            for i in range(1, k + 1):
                v[:] = [tmp - (linalg.binomial_coefficient(k, i) * CKw[i][-1] * drv) for tmp, drv in zip(v, CK[k - i])]
            CK[k][:] = [tmp / CKw[0][-1] for tmp in v]

        # Return C(u) derivatives
        return CK

@utl.export
class SurfaceEvaluator(AbstractEvaluator):
    """Sequential surface evaluation algorithms.

    This evaluator implements the following algorithms from **The NURBS Book**:

    * Algorithm A3.5: SurfacePoint
    * Algorithm A3.6: SurfaceDerivsAlg1

    Please note that knot vector span finding function may be changed by setting ``find_span_func`` keyword argument
    during the initialization. By default, this function is set to :py:func:`.helpers.find_span_linear`.
    Please see :doc:`Helpers Module Documentation <module_utilities>` for more details.
    """

    def __init__(self, **kwargs):
        super(SurfaceEvaluator, self).__init__(**kwargs)
        self._span_func = kwargs.get("find_span_func", helpers.find_span_linear)

    def evaluate(self, datadict, **kwargs):
        """Evaluates the rational curve.

        Keyword Arguments:
            * ``start``: starting parametric position for evaluation
            * ``stop``: ending parametric position for evaluation

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :return: evaluated points
        :rtype: list
        """
        # Geometry data from datadict
        cdef tuple sample_size = datadict["sample_size"]
        cdef int[2] degree = datadict["degree"]
        cdef tuple knotvector = datadict["knotvector"]
        cdef tuple ctrlpts = datadict["control_points"]
        cdef tuple size = datadict["size"]
        cdef int dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]
        cdef int pdimension = datadict["pdimension"]
        cdef int precision = datadict["precision"]

        # Keyword arguments
        start = kwargs.get("start", [0.0 for _ in range(pdimension)])
        stop = kwargs.get("stop", [1.0 for _ in range(pdimension)])

        # Algorithm A3.5
        cdef list knots
        cdef list spans = [[] for _ in range(pdimension)]
        cdef list basis = [[] for _ in range(pdimension)]
        cdef int idx
        for idx in range(pdimension):
            knots = linalg.linspace(start[idx], stop[idx], sample_size[idx], decimals=precision)
            spans[idx] = helpers.find_spans(degree[idx], knotvector[idx], size[idx], knots, self._span_func)
            basis[idx] = helpers.basis_functions(degree[idx], knotvector[idx], spans[idx], knots)

        cdef list eval_points = []
        cdef int i, j, k, m
        cdef int idx_u, idx_v
        cdef list spt, temp
        cdef double pt, tmp
        for i in range(len(spans[0])):
            idx_u = spans[0][i] - degree[0]
            for j in range(len(spans[1])):
                idx_v = spans[1][j] - degree[1]
                spt = [0.0 for _ in range(dimension)]
                for k in range(0, degree[0] + 1):
                    temp = [0.0 for _ in range(dimension)]
                    for m in range(0, degree[1] + 1):
                        temp[:] = [tmp + (basis[1][j][m] * cp) for tmp, cp in
                                   zip(temp, ctrlpts[idx_v + m + (size[1] * (idx_u + k))])]
                    spt[:] = [pt + (basis[0][i][k] * tmp) for pt, tmp in zip(spt, temp)]

                eval_points.append(spt)
        return eval_points

    def derivatives(self, tuple degree, tuple knotvector, tuple ctrlpts, tuple size, int dimension,
                    tuple parpos, int deriv_order):
        """
        Evaluates the n-th order derivatives at the input parametric position.

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :param parpos: parametric position where the derivatives will be computed
        :type parpos: list, tuple
        :param deriv_order: derivative order; to get the i-th derivative
        :type deriv_order: int
        :return: evaluated derivatives
        :rtype: list
        """
        # Geometry data from datadict
        # cdef int[2] degree = datadict["degree"]
        # cdef tuple knotvector = datadict["knotvector"]
        # cdef tuple ctrlpts = datadict["control_points"]
        # cdef tuple size = datadict["size"]
        # cdef int dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]
        cdef int degree_u = degree[0], degree_v = degree[1]
        cdef int size_u = size[0], size_v = size[1]
        cdef double u = parpos[0], v = parpos[1]
        cdef list knotvector_u = knotvector[0]
        cdef list knotvector_v = knotvector[1]
        cdef int pdimension = 2

        cdef int k, li, s, r, i, dd, cu, cv
        # Algorithm A3.6
        cdef int[2] d = [min(degree_u, deriv_order), min(degree_u, deriv_order)]

        cdef list SKL = [[[0.0 for _ in range(dimension)] for _ in range(deriv_order + 1)] for _ in range(deriv_order + 1)]

        # span = [0 for _ in range(pdimension)]
        # cdef int[2] span = [0, 0]
        # cdef list basisdrv = [[] for _ in range(pdimension)]
        # for idx in range(pdimension):
        #     span[idx] = self._span_func(degree[idx], knotvector[idx], size[idx], parpos[idx])
        #     basisdrv[idx] = helpers.basis_function_ders(degree[idx], knotvector[idx], span[idx], parpos[idx], d[idx])
        cdef int span_u = helpers.find_span_linear(degree_u, knotvector_u, size_u, u)
        cdef int span_v = helpers.find_span_linear(degree_v, knotvector_v, size_v, v)
        cdef list basisdrv_u = helpers.basis_function_ders(degree_u, knotvector_u, span_u, u, d[0])
        cdef list basisdrv_v = helpers.basis_function_ders(degree_v, knotvector_v, span_v, v, d[1])
        cdef list t = [0.0] * dimension
        cdef list cp = [0.0] * dimension
        cdef list tmp = [0.0] * dimension
        cdef list temp = [[0.0 for _ in range(dimension)] for _ in range(degree_v + 1)]
        for k in range(0, d[0] + 1):
            temp = [[0.0 for _ in range(dimension)] for _ in range(degree_v + 1)]
            for s in range(0, degree_v + 1):
                tmp = temp[s]
                for r in range(0, degree_u + 1):
                    cu = span_u - degree_u + r
                    cv = span_v - degree_v + s
                    cp = ctrlpts[cv + (size_v * cu)]
                    for i in range(dimension):
                        t[i] = tmp[i] + (basisdrv_u[k][r] * cp[i])
                    temp[s][:] = t

            dd = min(deriv_order, d[1])
            for li in range(0, dd + 1):
                for s in range(0, degree_v + 1):
                    elem = SKL[k][li]
                    tmp = temp[s]
                    for i in range(dimension):
                        t[i] = elem[i] + (basisdrv_v[li][s] * tmp[i])
                    SKL[k][li][:] = t
        return SKL


@utl.export
class SurfaceEvaluatorRational(SurfaceEvaluator):
    """Sequential rational surface evaluation algorithms.

    This evaluator implements the following algorithms from **The NURBS Book**:

    * Algorithm A4.3: SurfacePoint
    * Algorithm A4.4: RatSurfaceDerivs

    Please note that knot vector span finding function may be changed by setting ``find_span_func`` keyword argument
    during the initialization. By default, this function is set to :py:func:`.helpers.find_span_linear`.
    Please see :doc:`Helpers Module Documentation <module_utilities>` for more details.
    """

    def __init__(self, **kwargs):
        super(SurfaceEvaluatorRational, self).__init__(**kwargs)
        self._span_func = kwargs.get("find_span_func", helpers.find_span_linear)

    def evaluate(self, datadict, **kwargs):
        """Evaluates the rational surface.

        Keyword Arguments:
            * ``start``: starting parametric position for evaluation
            * ``stop``: ending parametric position for evaluation

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :return: evaluated points
        :rtype: list
        """
        cdef int dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]

        # Algorithm A4.3
        cdef list cptw = super(SurfaceEvaluatorRational, self).evaluate(datadict, **kwargs)

        # Divide by weight
        cdef list eval_points = []
        cdef double c
        cdef list pt, cpt
        for pt in cptw:
            cpt = [float(c / pt[-1]) for c in pt[0 : (dimension - 1)]]
            eval_points.append(cpt)

        return eval_points

    def derivatives(self, tuple degree, tuple knotvector, tuple ctrlpts, tuple size, int dimension,
                    tuple parpos, int deriv_order):
        """Evaluates the n-th order derivatives at the input parametric position.

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :param parpos: parametric position where the derivatives will be computed
        :type parpos: list, tuple
        :param deriv_order: derivative order; to get the i-th derivative
        :type deriv_order: int
        :return: evaluated derivatives
        :rtype: list
        """
        # cdef int dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]

        # Call the parent function to evaluate A(u) and w(u) derivatives
        cdef list SKLw = super(SurfaceEvaluatorRational, self).derivatives(degree, knotvector, ctrlpts, size,
                                                                           dimension, parpos, deriv_order)
        # Generate an empty list of derivatives
        cdef list SKL = [[[0.0 for _ in range(dimension)] for _ in range(deriv_order + 1)] for _ in range(deriv_order + 1)]

        # Algorithm A4.4


        cdef int i, j, k, li, ii

        cdef list tmp = [0.0] * (dimension - 1)
        cdef list drv = [0.0] * (dimension - 1)
        cdef list v = [0.0] * (dimension - 1)
        cdef list v2 = [0.0] * (dimension - 1)
        cdef list res = [0.0] * (dimension - 1)
        # Algorithm A4.4
        for k in range(0, deriv_order + 1):
            for li in range(0, deriv_order + 1):
                # Deep copying might seem a little overkill but we also want to avoid same pointer issues too
                # v = copy.deepcopy(SKLw[k][l])
                v = [value for value in SKLw[k][li]]
                for j in range(1, li + 1):
                    drv = SKL[k][li - j]
                    for ii in range(dimension - 1):
                        tmp[ii] = v[ii] - (linalg.binomial_coefficient(li, j) * SKLw[0][j][-1] * drv[ii])
                    v[:] = tmp
                for i in range(1, k + 1):
                    drv = SKL[k - i][li]
                    for ii in range(dimension - 1):
                        tmp[ii] = v[ii] - (linalg.binomial_coefficient(k, i) * SKLw[i][0][-1] * drv[ii])
                    v[:] = tmp
                    v2 = [0.0 for _ in range(dimension - 1)]
                    for j in range(1, li + 1):
                        drv = SKL[k - i][li - j]
                        for ii in range(dimension - 1):
                            tmp[ii] = v2[ii] + (linalg.binomial_coefficient(li, j) * SKLw[i][j][-1] * drv[ii])
                        v2[:] = tmp
                    for ii in range(dimension - 1):
                        tmp[ii] = v[ii] - (linalg.binomial_coefficient(k, i) * v2[ii])
                    v[:] = tmp

                for i in range(dimension - 1):
                    res[i] = v[i] / SKLw[0][0][-1]

                SKL[k][li][:] = res
        # Return S(u,v) derivatives
        return SKL


@utl.export
class VolumeEvaluator(AbstractEvaluator):
    """Sequential volume evaluation algorithms.

    Please note that knot vector span finding function may be changed by setting ``find_span_func`` keyword argument
    during the initialization. By default, this function is set to :py:func:`.helpers.find_span_linear`.
    Please see :doc:`Helpers Module Documentation <module_utilities>` for more details.
    """

    def __init__(self, **kwargs):
        super(VolumeEvaluator, self).__init__(**kwargs)
        self._span_func = kwargs.get("find_span_func", helpers.find_span_linear)

    def evaluate(self, datadict, **kwargs):
        """Evaluates the volume.

        Keyword Arguments:
            * ``start``: starting parametric position for evaluation
            * ``stop``: ending parametric position for evaluation

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :return: evaluated points
        :rtype: list
        """
        # Geometry data from datadict
        sample_size = datadict["sample_size"]
        degree = datadict["degree"]
        knotvector = datadict["knotvector"]
        ctrlpts = datadict["control_points"]
        size = datadict["size"]
        dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]
        pdimension = datadict["pdimension"]
        precision = datadict["precision"]

        # Keyword arguments
        start = kwargs.get("start", [0.0 for _ in range(pdimension)])
        stop = kwargs.get("stop", [1.0 for _ in range(pdimension)])

        # Algorithm A3.5 (modified)
        spans = [[] for _ in range(pdimension)]
        basis = [[] for _ in range(pdimension)]
        for idx in range(pdimension):
            knots = linalg.linspace(start[idx], stop[idx], sample_size[idx], decimals=precision)
            spans[idx] = helpers.find_spans(degree[idx], knotvector[idx], size[idx], knots, self._span_func)
            basis[idx] = helpers.basis_functions(degree[idx], knotvector[idx], spans[idx], knots)

        eval_points = []
        for i in range(len(spans[0])):
            iu = spans[0][i] - degree[0]
            for j in range(len(spans[1])):
                iv = spans[1][j] - degree[1]
                for k in range(len(spans[2])):
                    iw = spans[2][k] - degree[2]
                    spt = [0.0 for _ in range(dimension)]
                    for du in range(0, degree[0] + 1):
                        temp2 = [0.0 for _ in range(dimension)]
                        for dv in range(0, degree[1] + 1):
                            temp = [0.0 for _ in range(dimension)]
                            for dw in range(0, degree[2] + 1):
                                # flattening algorithm 1: x + (WIDTH * y) + (WIDTH * DEPTH) * z
                                # flattening algorithm 2: x + (WIDTH * (y + (DEPTH * z))
                                temp[:] = [
                                    tmp + (basis[2][k][dw] * cp)
                                    for tmp, cp in zip(
                                        temp, ctrlpts[iv + dv + (size[1] * (iu + du + (size[0] * (iw + dw))))]
                                    )
                                ]
                            temp2[:] = [pt + (basis[1][j][dv] * tmp) for pt, tmp in zip(temp2, temp)]
                        spt[:] = [pt + (basis[0][i][du] * tmp) for pt, tmp in zip(spt, temp2)]
                    eval_points.append(spt)

        return eval_points

    def derivatives(self, datadict, parpos, deriv_order=0, **kwargs):
        """Evaluates the n-th order derivatives at the input parametric position.

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :param parpos: parametric position where the derivatives will be computed
        :type parpos: list, tuple
        :param deriv_order: derivative order; to get the i-th derivative
        :type deriv_order: int
        :return: evaluated derivatives
        :rtype: list
        """
        # Geometry data from datadict
        degree = datadict["degree"]
        knotvector = datadict["knotvector"]
        ctrlpts = datadict["control_points"]
        size = datadict["size"]
        dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]
        pdimension = datadict["pdimension"]

        # TO-DO: Complete volume derivatives
        return list()


@utl.export
class VolumeEvaluatorRational(VolumeEvaluator):
    """Sequential rational volume evaluation algorithms.

    Please note that knot vector span finding function may be changed by setting ``find_span_func`` keyword argument
    during the initialization. By default, this function is set to :py:func:`.helpers.find_span_linear`.
    Please see :doc:`Helpers Module Documentation <module_utilities>` for more details.
    """

    def __init__(self, **kwargs):
        super(VolumeEvaluatorRational, self).__init__(**kwargs)
        self._span_func = kwargs.get("find_span_func", helpers.find_span_linear)

    def evaluate(self, datadict, **kwargs):
        """Evaluates the rational volume.

        Keyword Arguments:
            * ``start``: starting parametric position for evaluation
            * ``stop``: ending parametric position for evaluation

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :return: evaluated points
        :rtype: list
        """
        dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]

        # Algorithm A4.3 (modified)
        cptw = super(VolumeEvaluatorRational, self).evaluate(datadict, **kwargs)

        # Divide by weight
        eval_points = []
        for pt in cptw:
            cpt = [float(c / pt[-1]) for c in pt[0 : (dimension - 1)]]
            eval_points.append(cpt)

        return eval_points

    def derivatives(self, datadict, parpos, deriv_order=0, **kwargs):
        """Evaluates the n-th order derivatives at the input parametric position.

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :param parpos: parametric position where the derivatives will be computed
        :type parpos: list, tuple
        :param deriv_order: derivative order; to get the i-th derivative
        :type deriv_order: int
        :return: evaluated derivatives
        :rtype: list
        """
        dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]

        # Call the parent function to evaluate A(u) and w(u) derivatives
        SKLw = super(VolumeEvaluatorRational, self).derivatives(datadict, parpos, deriv_order, **kwargs)

        # TO-DO: Complete rational volume derivatives
        return list()


# Don't export alternative curve evalutator
class CurveEvaluator2(CurveEvaluator):
    """Sequential curve evaluation algorithms (alternative).

    This evaluator implements the following algorithms from **The NURBS Book**:

    * Algorithm A3.1: CurvePoint
    * Algorithm A3.3: CurveDerivCpts
    * Algorithm A3.4: CurveDerivsAlg2

    Please note that knot vector span finding function may be changed by setting ``find_span_func`` keyword argument
    during the initialization. By default, this function is set to :py:func:`.helpers.find_span_linear`.
    Please see :doc:`Helpers Module Documentation <module_utilities>` for more details.
    """

    def __init__(self, **kwargs):
        super(CurveEvaluator2, self).__init__(**kwargs)
        self._span_func = kwargs.get("find_span_func", helpers.find_span_linear)

    def derivatives(self, int degree, list knotvector, list ctrlpts, int size, int dimension, double parpos,
                    int deriv_order):
        """Evaluates the n-th order derivatives at the input parametric position.

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :param parpos: parametric position where the derivatives will be computed
        :type parpos: list, tuple
        :param deriv_order: derivative order; to get the i-th derivative
        :type deriv_order: int
        :return: evaluated derivatives
        :rtype: list
        """
        # # Geometry data from datadict
        # degree = datadict["degree"][0]
        # knotvector = datadict["knotvector"][0]
        # ctrlpts = datadict["control_points"]
        # size = datadict["size"][0]
        # dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]

        # Algorithm A3.4
        cdef int du = min(degree, deriv_order)

        cdef list CK = [[0.0 for _ in range(dimension)] for _ in range(deriv_order + 1)]

        cdef int span = self._span_func(degree, knotvector, size, parpos)
        cdef list bfuns = helpers.basis_function_all(degree, knotvector, span, parpos)

        # Algorithm A3.3
        PK = helpers.curve_deriv_cpts(
            dimension, degree, knotvector, ctrlpts, rs=((span - degree), span), deriv_order=du
        )
        cdef int k, j
        cdef double elem, drv_ctl_p
        for k in range(0, du + 1):
            for j in range(0, degree - k + 1):
                CK[k][:] = [elem + (bfuns[j][degree - k] * drv_ctl_p) for elem, drv_ctl_p in zip(CK[k], PK[k][j])]

        # Return the derivatives
        return CK


# Don't export alternative surface evaluator
class SurfaceEvaluator2(SurfaceEvaluator):
    """Sequential surface evaluation algorithms (alternative).

    This evaluator implements the following algorithms from **The NURBS Book**:

    * Algorithm A3.5: SurfacePoint
    * Algorithm A3.7: SurfaceDerivCpts
    * Algorithm A3.8: SurfaceDerivsAlg2

    Please note that knot vector span finding function may be changed by setting ``find_span_func`` keyword argument
    during the initialization. By default, this function is set to :py:func:`.helpers.find_span_linear`.
    Please see :doc:`Helpers Module Documentation <module_utilities>` for more details.
    """

    def __init__(self, **kwargs):
        super(SurfaceEvaluator2, self).__init__(**kwargs)
        self._span_func = kwargs.get("find_span_func", helpers.find_span_linear)

    def derivatives(self, tuple degree, tuple knotvector, tuple ctrlpts, tuple size, int dimension,
                    tuple parpos, int deriv_order):
        """Evaluates the n-th order derivatives at the input parametric position.

        :param datadict: data dictionary containing the necessary variables
        :type datadict: dict
        :param parpos: parametric position where the derivatives will be computed
        :type parpos: list, tuple
        :param deriv_order: derivative order; to get the i-th derivative
        :type deriv_order: int
        :return: evaluated derivatives
        :rtype: list
        """
        # Geometry data from datadict
        # degree = datadict["degree"]
        # knotvector = datadict["knotvector"]
        # ctrlpts = datadict["control_points"]
        # size = datadict["size"]
        # dimension = datadict["dimension"] + 1 if datadict["rational"] else datadict["dimension"]
        cdef int pdimension = 2

        SKL = [[[0.0 for _ in range(dimension)] for _ in range(deriv_order + 1)] for _ in range(deriv_order + 1)]

        d = (min(degree[0], deriv_order), min(degree[1], deriv_order))

        span = [0 for _ in range(pdimension)]
        basis = [[] for _ in range(pdimension)]
        for idx in range(pdimension):
            span[idx] = self._span_func(degree[idx], knotvector[idx], size[idx], parpos[idx])
            basis[idx] = helpers.basis_function_all(degree[idx], knotvector[idx], span[idx], parpos[idx])

        # Algorithm A3.7
        # rs: (minimum, maximum) span on the u-direction., ss: (minimum, maximum) span on the v-direction
        PKL = helpers.surface_deriv_cpts(
            dimension,
            degree,
            knotvector,
            ctrlpts,
            size,
            rs=(span[0] - degree[0], span[0]),
            ss=(span[1] - degree[1], span[1]),
            deriv_order=deriv_order,
        )

        # Evaluating the derivative at parameters (u,v) using its control points
        for k in range(0, d[0] + 1):
            dd = min(deriv_order - k, d[1])

            for l in range(0, dd + 1):
                SKL[k][l] = [0.0 for _ in range(dimension)]

                for i in range(0, degree[1] - l + 1):
                    temp = [0.0 for _ in range(dimension)]

                    for j in range(0, degree[0] - k + 1):
                        temp[:] = [
                            elem + (basis[0][j][degree[0] - k] * drv_ctl_p)
                            for elem, drv_ctl_p in zip(temp, PKL[k][l][j][i])
                        ]

                    SKL[k][l][:] = [
                        elem + (basis[1][i][degree[1] - l] * drv_ctl_p) for elem, drv_ctl_p in zip(SKL[k][l], temp)
                    ]

        return SKL

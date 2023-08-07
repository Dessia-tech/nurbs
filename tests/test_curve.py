"""
    Tests for the NURBS-Python package
    Released under The MIT License. See LICENSE file for details.
    Copyright (c) 2018-2019 Onur Rauf Bingol

    Requires "pytest" to run.
"""

from unittest import TestCase

import numpy as np

from nurbs import BSpline, convert, evaluators, fitting, helpers, operations

GEOMDL_DELTA = 0.001


class TestCurve(TestCase):
    def setUp(self):
        """Creates a spline Curve"""
        self.spline_curve = BSpline.Curve()
        self.spline_curve.degree = 3
        self.spline_curve.ctrlpts = [
            [5.0, 5.0],
            [10.0, 10.0],
            [20.0, 15.0],
            [35.0, 15.0],
            [45.0, 10.0],
            [50.0, 5.0],
        ]
        self.spline_curve.knotvector = [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0]

    def test_bspline_curve_name(self):
        self.spline_curve.name = "Curve Testing"
        self.assertEqual(self.spline_curve.name, "Curve Testing")

    def test_bspline_curve_degree(self):
        self.assertEqual(self.spline_curve.degree, 3)

    def test_bspline_curve_ctrlpts(self):
        expected_ctrlpts = [
            [5.0, 5.0],
            [10.0, 10.0],
            [20.0, 15.0],
            [35.0, 15.0],
            [45.0, 10.0],
            [50.0, 5.0],
        ]
        self.assertEqual(self.spline_curve.ctrlpts, expected_ctrlpts)
        self.assertEqual(self.spline_curve.dimension, 2)

    def test_bspline_curve_knot_vector(self):
        expected_knots = [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0]
        self.assertEqual(self.spline_curve.knotvector, expected_knots)

    def test_bspline_curve2d_eval(self):
        test_cases = [
            (0.0, (5.0, 5.0)),
            (0.3, (18.617, 13.377)),
            (0.5, (27.645, 14.691)),
            (0.6, (32.143, 14.328)),
            (1.0, (50.0, 5.0)),
        ]
        for param, res in test_cases:
            with self.subTest(param=param):
                evalpt = self.spline_curve.evaluate_single(param)
                self.assertAlmostEqual(evalpt[0], res[0], delta=GEOMDL_DELTA)
                self.assertAlmostEqual(evalpt[1], res[1], delta=GEOMDL_DELTA)

    def test_bspline_curve2d_deriv(self):
        der1 = self.spline_curve.derivatives(u=0.35, order=2)
        self.spline_curve.evaluator = evaluators.CurveEvaluator2()
        der2 = self.spline_curve.derivatives(u=0.35, order=2)

        for i in range(2):
            with self.subTest(order=i):
                self.assertAlmostEqual(der1[0][i], der2[0][i], delta=GEOMDL_DELTA)
                self.assertAlmostEqual(der1[1][i], der2[1][i], delta=GEOMDL_DELTA)
                self.assertAlmostEqual(der1[2][i], der2[2][i], delta=GEOMDL_DELTA)

    def test_bspline_curve2d_deriv_eval(self):
        evalpt = self.spline_curve.evaluate_single(0.35)
        der1 = self.spline_curve.derivatives(u=0.35)
        self.spline_curve.evaluator = evaluators.CurveEvaluator2()
        der2 = self.spline_curve.derivatives(u=0.35)

        for i in range(2):
            with self.subTest(order=i):
                self.assertAlmostEqual(der1[0][i], evalpt[i], delta=GEOMDL_DELTA)
                self.assertAlmostEqual(der2[0][i], evalpt[i], delta=GEOMDL_DELTA)

    def test_bspline_curve2d_insert_knot(self):
        test_cases = [
            (0.3, 1, (18.617, 13.377)),
            (0.6, 1, (32.143, 14.328)),
            (0.6, 2, (32.143, 14.328)),
        ]
        for param, num_insert, res in test_cases:
            with self.subTest(param=param, num_insert=num_insert):
                s_pre = helpers.find_multiplicity(param, self.spline_curve.knotvector)
                self.spline_curve.insert_knot(param, num=num_insert)
                s_post = helpers.find_multiplicity(param, self.spline_curve.knotvector)
                evalpt = self.spline_curve.evaluate_single(param)

                self.assertAlmostEqual(evalpt[0], res[0], delta=GEOMDL_DELTA)
                self.assertAlmostEqual(evalpt[1], res[1], delta=GEOMDL_DELTA)
                self.assertEqual(s_pre + num_insert, s_post)

    def test_bspline_curve2d_insert_knot_kv(self):
        self.spline_curve.insert_knot(0.66, num=2)
        s = helpers.find_multiplicity(0.66, self.spline_curve.knotvector)

        self.assertEqual(self.spline_curve.knotvector[5], 0.66)
        self.assertEqual(s, 3)

    def test_bspline_curve2d_remove_knot(self):
        test_cases = [(0.33, 1), (0.66, 1)]
        for param, num_remove in test_cases:
            with self.subTest(param=param, num_remove=num_remove):
                s_pre = helpers.find_multiplicity(param, self.spline_curve.knotvector)
                c_pre = self.spline_curve.ctrlpts_size
                self.spline_curve.remove_knot(param, num=num_remove)
                s_post = helpers.find_multiplicity(param, self.spline_curve.knotvector)
                c_post = self.spline_curve.ctrlpts_size

                self.assertEqual(c_pre - num_remove, c_post)
                self.assertEqual(s_pre - num_remove, s_post)

    def test_bspline_curve2d_remove_knot_kv(self):
        self.spline_curve.remove_knot(0.66, num=1)
        s = helpers.find_multiplicity(0.66, self.spline_curve.knotvector)

        self.assertNotIn(0.66, self.spline_curve.knotvector)

    def test_bspline_curve2d_knot_refine(self):
        test_cases = [
            (0, [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0]),
            (
                1,
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.165,
                    0.165,
                    0.165,
                    0.33,
                    0.33,
                    0.33,
                    0.495,
                    0.495,
                    0.495,
                    0.66,
                    0.66,
                    0.66,
                    0.830,
                    0.830,
                    0.830,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
            ),
        ]
        for density, kv in test_cases:
            with self.subTest(density=density, kv=kv):
                operations.refine_knotvector(self.spline_curve, [density])
                for a, b in zip(kv, self.spline_curve.knotvector):
                    self.assertAlmostEqual(a, b, delta=GEOMDL_DELTA)

    def test_bspline_curve2d_degree_elevate_degree(self):
        dops = 1
        degree_new = self.spline_curve.degree + dops
        operations.degree_operations(self.spline_curve, [dops])
        self.assertEqual(self.spline_curve.degree, degree_new)

    def test_bspline_curve2d_degree_elevate_ctrlpts_size(self):
        dops = 1
        ctrlpts_size = self.spline_curve.ctrlpts_size + dops
        operations.degree_operations(self.spline_curve, [dops])
        self.assertEqual(self.spline_curve.ctrlpts_size, ctrlpts_size)

    def test_bspline_curve2d_degree_reduce_degree(self):
        dops = -1
        degree_new = self.spline_curve.degree + dops
        operations.degree_operations(self.spline_curve, [dops])
        self.assertEqual(self.spline_curve.degree, degree_new)

    def test_bspline_curve2d_degree_reduce_ctrlpts_size(self):
        dops = -1
        ctrlpts_size_new = self.spline_curve.ctrlpts_size + dops
        operations.degree_operations(self.spline_curve, [dops])
        self.assertEqual(self.spline_curve.ctrlpts_size, ctrlpts_size_new)

    def test_bspline_curve3d_ctrlpts(self):
        curve3d = operations.add_dimension(self.spline_curve, offset=1.0)
        expected_ctrlpts = [
            [5.0, 5.0, 1.0],
            [10.0, 10.0, 1.0],
            [20.0, 15.0, 1.0],
            [35.0, 15.0, 1.0],
            [45.0, 10.0, 1.0],
            [50.0, 5.0, 1.0],
        ]
        self.assertEqual(curve3d.ctrlpts, expected_ctrlpts)
        self.assertEqual(curve3d.dimension, 3)

    def test_nurbs_curve2d_weights(self):
        curve = convert.bspline_to_nurbs(self.spline_curve)
        curve.weights = [0.5, 1.0, 0.75, 1.0, 0.25, 1.0]
        self.assertEqual(curve.weights, [0.5, 1.0, 0.75, 1.0, 0.25, 1.0])

    def test_nurbs_curve2d_eval(self):
        curve = convert.bspline_to_nurbs(self.spline_curve)
        curve.weights = [0.5, 1.0, 0.75, 1.0, 0.25, 1.0]
        test_cases = [
            (0.0, (5.0, 5.0)),
            (0.2, (13.8181, 11.5103)),
            (0.5, (28.1775, 14.7858)),
            (0.95, (48.7837, 6.0022)),
        ]
        for param, res in test_cases:
            with self.subTest(param=param):
                evalpt = curve.evaluate_single(param)

                self.assertAlmostEqual(evalpt[0], res[0], delta=GEOMDL_DELTA)
                self.assertAlmostEqual(evalpt[1], res[1], delta=GEOMDL_DELTA)

    def test_nurbs_curve2d_deriv(self):
        curve = convert.bspline_to_nurbs(self.spline_curve)
        curve.weights = [0.5, 1.0, 0.75, 1.0, 0.25, 1.0]
        test_cases = [
            (0.0, 1, ((5.0, 5.0), (90.9090, 90.9090))),
            (0.2, 2, ((13.8181, 11.5103), (40.0602, 17.3878), (104.4062, -29.3672))),
            (0.5, 3, ((28.1775, 14.7858), (39.7272, 2.2562), (-116.9254, -49.7367), (125.5276, 196.8865))),
            (0.95, 1, ((48.7837, 6.0022), (39.5178, -29.9962))),
        ]
        for param, order, res in test_cases:
            with self.subTest(param=param, order=order):
                deriv = curve.derivatives(u=param, order=order)

                for computed, expected in zip(deriv, res):
                    for c, e in zip(computed, expected):
                        self.assertAlmostEqual(c, e, delta=GEOMDL_DELTA)

    def test_bspline_curve2d_eval_kv_norm1(self):
        spline_curve_kv_norm1 = BSpline.Curve(normalize_kv=True)
        spline_curve_kv_norm1.degree = 3
        spline_curve_kv_norm1.ctrlpts = [
            [5.0, 5.0],
            [10.0, 10.0],
            [20.0, 15.0],
            [35.0, 15.0],
            [45.0, 10.0],
            [50.0, 5.0],
        ]
        spline_curve_kv_norm1.knotvector = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]

        test_cases = [
            (0.0, (5.0, 5.0)),
            (0.33333, (19.9998, 13.7499)),
            (0.5, (27.5, 14.687)),
            (0.66666, (34.9997, 13.75)),
            (1.0, (50.0, 5.0)),
        ]
        for param, res in test_cases:
            with self.subTest(param=param):
                evalpt = spline_curve_kv_norm1.evaluate_single(param)

                self.assertAlmostEqual(evalpt[0], res[0], delta=GEOMDL_DELTA)
                self.assertAlmostEqual(evalpt[1], res[1], delta=GEOMDL_DELTA)

    def test_bspline_curve2d_eval_kv_norm2(self):
        spline_curve_kv_norm2 = BSpline.Curve(normalize_kv=False)
        spline_curve_kv_norm2.degree = 3
        spline_curve_kv_norm2.ctrlpts = [
            [5.0, 5.0],
            [10.0, 10.0],
            [20.0, 15.0],
            [35.0, 15.0],
            [45.0, 10.0],
            [50.0, 5.0],
        ]
        spline_curve_kv_norm2.knotvector = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]

        test_cases = [
            (0.0, (5.0, 5.0)),
            (1.0, (19.9998, 13.7499)),
            (1.5, (27.5, 14.687)),
            (2.0, (34.9997, 13.75)),
            (3.0, (50.0, 5.0)),
        ]
        for param, res in test_cases:
            with self.subTest(param=param):
                evalpt = spline_curve_kv_norm2.evaluate_single(param)

                self.assertAlmostEqual(evalpt[0], res[0], delta=GEOMDL_DELTA)
                self.assertAlmostEqual(evalpt[1], res[1], delta=GEOMDL_DELTA)

    def test_interpolate_curve(self):
        # The NURBS Book Ex9.1
        points = [[0, 0], [3, 4], [-1, 4], [-4, 0], [-4, -3]]
        degree = 3  # cubic curve

        # Do global curve interpolation
        curve = fitting.interpolate_curve(points, degree)
        expected_ctrlpts = [
            [0.0, 0.0],
            [7.3169635171119936, 3.6867775257587367],
            [-2.958130565851424, 6.678276528176592],
            [-4.494953466891109, -0.6736915062424752],
            [-4.0, -3.0],
        ]
        for point, expected_point in zip(curve.ctrlpts, expected_ctrlpts):
            self.assertAlmostEqual(point[0], expected_point[0], delta=GEOMDL_DELTA)
            self.assertAlmostEqual(point[1], expected_point[1], delta=GEOMDL_DELTA)

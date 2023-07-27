"""
    Tests for the NURBS-Python package
    Released under The MIT License. See LICENSE file for details.
    Copyright (c) 2018-2019 Onur Rauf Bingol

    Requires "pytest" to run.
"""

import unittest
from geomdl import BSpline, convert, evaluators, helpers

GEOMDL_DELTA = 0.001

class TestSurface(unittest.TestCase):
    def setUp(self):
        """Create a B-spline surface instance as a fixture"""
        # Create a surface instance
        self.spline_surf = BSpline.Surface()

        # Set degrees
        self.spline_surf.degree_u = 3
        self.spline_surf.degree_v = 3

        ctrlpts = [
            [-25.0, -25.0, -10.0],
            [-25.0, -15.0, -5.0],
            [-25.0, -5.0, 0.0],
            [-25.0, 5.0, 0.0],
            [-25.0, 15.0, -5.0],
            [-25.0, 25.0, -10.0],
            [-15.0, -25.0, -8.0],
            [-15.0, -15.0, -4.0],
            [-15.0, -5.0, -4.0],
            [-15.0, 5.0, -4.0],
            [-15.0, 15.0, -4.0],
            [-15.0, 25.0, -8.0],
            [-5.0, -25.0, -5.0],
            [-5.0, -15.0, -3.0],
            [-5.0, -5.0, -8.0],
            [-5.0, 5.0, -8.0],
            [-5.0, 15.0, -3.0],
            [-5.0, 25.0, -5.0],
            [5.0, -25.0, -3.0],
            [5.0, -15.0, -2.0],
            [5.0, -5.0, -8.0],
            [5.0, 5.0, -8.0],
            [5.0, 15.0, -2.0],
            [5.0, 25.0, -3.0],
            [15.0, -25.0, -8.0],
            [15.0, -15.0, -4.0],
            [15.0, -5.0, -4.0],
            [15.0, 5.0, -4.0],
            [15.0, 15.0, -4.0],
            [15.0, 25.0, -8.0],
            [25.0, -25.0, -10.0],
            [25.0, -15.0, -5.0],
            [25.0, -5.0, 2.0],
            [25.0, 5.0, 2.0],
            [25.0, 15.0, -5.0],
            [25.0, 25.0, -10.0],
        ]

        # Set control points
        self.spline_surf.set_ctrlpts(ctrlpts, 6, 6)

        # Set knot vectors
        self.spline_surf.knotvector_u = [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0]
        self.spline_surf.knotvector_v = [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0]

        self.nurbs_surf = convert.bspline_to_nurbs(self.spline_surf)

    def test_bspline_surface_name(self):
        self.spline_surf.name = "Surface Testing"
        self.assertEqual(self.spline_surf.name, "Surface Testing")

    def test_bspline_surface_degree_u(self):
        self.assertEqual(self.spline_surf.degree_u, 3)

    def test_bspline_surface_degree_v(self):
        self.assertEqual(self.spline_surf.degree_v, 3)

    def test_bspline_surface_ctrlpts(self):
        self.assertEqual(self.spline_surf.ctrlpts2d[1][1], [-15.0, -15.0, -4.0])
        self.assertEqual(self.spline_surf.dimension, 3)

    def test_bspline_surface_knot_vector_u(self):
        self.assertEqual(self.spline_surf.knotvector_u, [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0])

    def test_bspline_surface_knot_vector_v(self):
        self.assertEqual(self.spline_surf.knotvector_v, [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0])

    def test_bspline_surface_eval(self):
        test_cases = [
                      ((0.0, 0.0), (-25.0, -25.0, -10.0)),
                      ((0.0, 0.2), (-25.0, -11.403, -3.385)),
                      ((0.0, 1.0), (-25.0, 25.0, -10.0)),
                      ((0.3, 0.0), (-7.006, -25.0, -5.725)),
                      ((0.3, 0.4), [-7.006, -3.308, -6.265]),
                      ((0.3, 1.0), [-7.006, 25.0, -5.725]),
                      ((0.6, 0.0), (3.533, -25.0, -4.224)),
                      ((0.6, 0.6), (3.533, 3.533, -6.801)),
                      ((0.6, 1.0), (3.533, 25.0, -4.224)),
                      ((1.0, 0.0), (25.0, -25.0, -10.0)),
                      ((1.0, 0.8), (25.0, 11.636, -2.751)),
                      ((1.0, 1.0), (25.0, 25.0, -10.0))]

        for param, res in test_cases:
            with self.subTest(param=param):
                evalpt = self.spline_surf.evaluate_single(param)
                self.assertAlmostEqual(evalpt[0], res[0], delta=GEOMDL_DELTA)
                self.assertAlmostEqual(evalpt[1], res[1], delta=GEOMDL_DELTA)
                self.assertAlmostEqual(evalpt[2], res[2], delta=GEOMDL_DELTA)

    def test_bspline_surface_deriv(self):
        der1 = self.spline_surf.derivatives(u=0.35, v=0.35, order=2)
        self.spline_surf.evaluator = evaluators.SurfaceEvaluator2()
        der2 = self.spline_surf.derivatives(u=0.35, v=0.35, order=2)
        for k in range(0, 3):
            for l in range(0, 3 - k):
                self.assertAlmostEqual(der1[k][l][0], der2[k][l][0], delta=GEOMDL_DELTA)
                self.assertAlmostEqual(der1[k][l][1], der2[k][l][1], delta=GEOMDL_DELTA)
                self.assertAlmostEqual(der1[k][l][2], der2[k][l][2], delta=GEOMDL_DELTA)

    def test_bspline_surface_insert_knot_eval(self):
        test_data = [
            # Test data for evaluation
            (dict(u=0.3, v=0.4), (0.3, 0.4), (-7.006, -3.308, -6.265)),
            (dict(u=0.3, num_u=2), (0.3, 0.4), (-7.006, -3.308, -6.265)),
            (dict(v=0.3, num_v=2), (0.3, 0.4), (-7.006, -3.308, -6.265)),
        ]

        for params, uv, res in test_data:
            self.spline_surf.insert_knot(**params)

            # Evaluate surface
            evalpt = self.spline_surf.evaluate_single(uv)

            self.assertAlmostEqual(evalpt[0], res[0], delta=GEOMDL_DELTA)
            self.assertAlmostEqual(evalpt[1], res[1], delta=GEOMDL_DELTA)
            self.assertAlmostEqual(evalpt[2], res[2], delta=GEOMDL_DELTA)

    def test_bspline_surface_insert_knot_kv_v(self):
        test_data = [(dict(v=0.3, num_v=2), 4, 0.3), (dict(v=0.33, num_v=2), 6, 0.33)]

        for params, idx, val in test_data:
            self.spline_surf.insert_knot(**params)

            self.assertEqual(self.spline_surf.knotvector_v[idx], val)

    def test_bspline_surface_insert_knot_kv_u(self):
        test_data = [(dict(u=0.33, num_u=2), 3, 0.0), (dict(u=0.66, num_u=1), 8, 0.66)]

        for params, idx, val in test_data:
            self.spline_surf.insert_knot(**params)

            self.assertEqual(self.spline_surf.knotvector_u[idx], val)

    def test_bspline_surface_remove_knot_u(self):
        test_data = [(0.33, 1), (0.66, 1)]
        for param, num_remove in test_data:
            s_pre = helpers.find_multiplicity(param, self.spline_surf.knotvector_u)
            c_pre = self.spline_surf.ctrlpts_size_u
            self.spline_surf.remove_knot(u=param, num_u=num_remove)
            s_post = helpers.find_multiplicity(param, self.spline_surf.knotvector_u)
            c_post = self.spline_surf.ctrlpts_size_u

            self.assertEqual(c_pre - num_remove, c_post)
            self.assertEqual(s_pre - num_remove, s_post)

    def test_bspline_surface_remove_knot_v(self):
        test_data = [(0.33, 1), (0.66, 1)]
        for param, num_remove in test_data:
            # with self.subTest(param=param, num_remove=num_remove):
            s_pre = helpers.find_multiplicity(param, self.spline_surf.knotvector_v)
            c_pre = self.spline_surf.ctrlpts_size_v
            self.spline_surf.remove_knot(v=param, num_v=num_remove)
            s_post = helpers.find_multiplicity(param, self.spline_surf.knotvector_v)
            c_post = self.spline_surf.ctrlpts_size_v

            self.assertEqual(c_pre - num_remove, c_post)
            self.assertEqual(s_pre - num_remove, s_post)

    def test_bspline_surface_remove_knot_kv_u(self):
        self.spline_surf.remove_knot(u=0.66, num_u=1)
        s = helpers.find_multiplicity(0.66, self.spline_surf.knotvector_u)

        self.assertNotIn(0.66, self.spline_surf.knotvector_u)
        self.assertEqual(s, 0)

    def test_bspline_surface_remove_knot_kv_v(self):
        self.spline_surf.remove_knot(v=0.33, num_v=1)
        s = helpers.find_multiplicity(0.33, self.spline_surf.knotvector_v)

        self.assertNotIn(0.33, self.spline_surf.knotvector_v)
        self.assertEqual(s, 0)

    def test_nurbs_weights(self):
        self.assertEqual(len(self.nurbs_surf.weights), self.nurbs_surf.ctrlpts_size)
        self.assertEqual(self.nurbs_surf.weights[5], 1.0)

    def test_nurbs_surface_eval(self):
        test_data =[
        ((0.0, 0.0), (-25.0, -25.0, -10.0)),
        ((0.0, 0.2), (-25.0, -11.403, -3.385)),
        ((0.0, 1.0), (-25.0, 25.0, -10.0)),
        ((0.3, 0.0), (-7.006, -25.0, -5.725)),
        ((0.3, 0.4), [-7.006, -3.308, -6.265]),
        ((0.3, 1.0), [-7.006, 25.0, -5.725]),
        ((0.6, 0.0), (3.533, -25.0, -4.224)),
        ((0.6, 0.6), (3.533, 3.533, -6.801)),
        ((0.6, 1.0), (3.533, 25.0, -4.224)),
        ((1.0, 0.0), (25.0, -25.0, -10.0)),
        ((1.0, 0.8), (25.0, 11.636, -2.751)),
        ((1.0, 1.0), (25.0, 25.0, -10.0)),
    ]
        for param, res in test_data:
            evalpt = self.nurbs_surf.evaluate_single(param)
            self.assertAlmostEqual(evalpt[0], res[0], delta=GEOMDL_DELTA)
            self.assertAlmostEqual(evalpt[1], res[1], delta=GEOMDL_DELTA)
            self.assertAlmostEqual(evalpt[2], res[2], delta=GEOMDL_DELTA)

    def test_nurbs_surface_deriv(self):
        test_data = [
        (
            (0.0, 0.25),
            1,
            [
                [[-25.0, -9.0771, -2.3972], [5.5511e-15, 43.6910, 17.5411]],
                [[90.9090, 0.0, -15.0882], [-5.9750e-15, 0.0, -140.0367]],
            ],
        ),
        (
            (0.95, 0.75),
            2,
            [
                [[20.8948, 9.3097, -2.4845], [-1.1347e-14, 43.7672, -15.0153], [-5.0393e-30, 100.1022, -74.1165]],
                [
                    [76.2308, -1.6965e-15, 18.0372],
                    [9.8212e-15, -5.9448e-15, -158.5462],
                    [4.3615e-30, -2.4356e-13, -284.3037],
                ],
                [
                    [224.5342, -5.6794e-14, 93.3843],
                    [4.9856e-14, -4.0400e-13, -542.6274],
                    [2.2140e-29, -1.88662e-12, -318.8808],
                ],
            ],
        ),
    ]
        for param, order, res in test_data:
            deriv = self.nurbs_surf.derivatives(*param, order=order)
            for computed, expected in zip(deriv, res):
                for idx in range(order + 1):
                    for c, e in zip(computed[idx], expected[idx]):
                        self.assertAlmostEqual(c, e, delta=GEOMDL_DELTA)
    #

    def test_surface_bounding_box(self):
        # Evaluate bounding box
        to_check = self.spline_surf.bbox

        # Evaluation result
        result = ((-25.0, -25.0, -10.0), (25.0, 25.0, 2.0))

        self.assertAlmostEqual(to_check[0][0], result[0][0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[0][1], result[0][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[0][2], result[0][2], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[1][0], result[1][0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[1][1], result[1][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[1][2], result[1][2], delta=GEOMDL_DELTA)


if __name__ == "__main__":
    unittest.main()

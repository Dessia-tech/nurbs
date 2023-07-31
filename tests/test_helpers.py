"""
    Tests for the NURBS-Python package
    Released under The MIT License. See LICENSE file for details.
    Copyright (c) 2018-2019 Onur Rauf Bingol

    Requires "pytest" to run.
"""
import unittest
from nurbs import helpers

GEOMDL_DELTA = 10e-8


class TestHelpers(unittest.TestCase):
    def test_find_span_binsearch(self):
        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        num_ctrlpts = len(knot_vector) - degree - 1
        knot = 5.0 / 2.0

        to_check = helpers.find_span_binsearch(degree, knot_vector, num_ctrlpts, knot)
        result = 4  # Value from The Nurbs Book p.68

        self.assertEqual(to_check, result)

    def test_find_span_linear(self):
        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        num_ctrlpts = len(knot_vector) - degree - 1
        knot = 5.0 / 2.0

        to_check = helpers.find_span_linear(degree, knot_vector, num_ctrlpts, knot)
        result = 4  # Value from The Nurbs Book p.68

        self.assertEqual(to_check, result)

    def test_basis_function(self):
        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        span = 4
        knot = 5.0 / 2.0

        to_check = helpers.basis_function(degree, knot_vector, span, knot)
        result = [1.0 / 8.0, 6.0 / 8.0, 1.0 / 8.0]  # Values from The Nurbs Book p.69

        self.assertAlmostEqual(to_check[0], result[0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[1], result[1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[2], result[2], delta=GEOMDL_DELTA)

    def test_basis_functions(self):
        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        spans = [3, 4, 5]
        knots = [5.0 / 2.0 for _ in range(len(spans))]

        to_check = helpers.basis_functions(degree, knot_vector, spans, knots)
        result = [
            helpers.basis_function(degree, knot_vector, spans[_], knots[_]) for _ in range(len(spans))
        ]  # Values from The Nurbs Book p.55

        self.assertAlmostEqual(to_check[0][0], result[0][0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[0][1], result[0][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[0][2], result[0][2], delta=GEOMDL_DELTA)

        self.assertAlmostEqual(to_check[1][0], result[1][0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[1][1], result[1][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[1][2], result[1][2], delta=GEOMDL_DELTA)

        self.assertAlmostEqual(to_check[2][0], result[2][0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[2][1], result[2][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[2][2], result[2][2], delta=GEOMDL_DELTA)

    def test_basis_function_all(self):
        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        span = 4
        knot = 5.0 / 2.0

        to_check = helpers.basis_function_all(degree, knot_vector, span, knot)
        interm = [
            helpers.basis_function(_, knot_vector, span, knot) + [None] * (degree - _) for _ in range(0, degree + 1)
        ]
        result = [list(_) for _ in zip(*interm)]  # Transposing to the same format as to_check

        self.assertAlmostEqual(to_check[0][0], result[0][0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[0][1], result[0][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[0][2], result[0][2], delta=GEOMDL_DELTA)

        self.assertEqual(to_check[1][0], result[1][0])  # NoneType can't be subtracted
        self.assertAlmostEqual(to_check[1][1], result[1][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[1][2], result[1][2], delta=GEOMDL_DELTA)

        self.assertEqual(to_check[2][0], result[2][0])  # None
        self.assertEqual(to_check[2][1], result[2][1])  # None
        self.assertAlmostEqual(to_check[2][2], result[2][2], delta=GEOMDL_DELTA)

    def test_basis_function_ders(self):
        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        span = 4
        knot = 5.0 / 2.0
        order = 2

        to_check = helpers.basis_function_ders(degree, knot_vector, span, knot, order)
        result = [
            [0.125, 0.75, 0.125],
            [-0.5, 0.0, 0.5],
            [1.0, -2.0, 1.0],
        ]  # Values and formulas from The Nurbs Book p.69 & p.72

        self.assertAlmostEqual(to_check[0][0], result[0][0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[0][1], result[0][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[0][2], result[0][2], delta=GEOMDL_DELTA)

        self.assertAlmostEqual(to_check[1][0], result[1][0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[1][1], result[1][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[1][2], result[1][2], delta=GEOMDL_DELTA)

        self.assertAlmostEqual(to_check[2][0], result[2][0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[2][1], result[2][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[2][2], result[2][2], delta=GEOMDL_DELTA)

    def test_find_multiplicity(self):
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        knots = [0.5, 2, 4, 5]

        to_check = [helpers.find_multiplicity(_, knot_vector) for _ in knots]
        result = [0, 1, 2, 3]

        self.assertEqual(to_check, result)

    def test_basis_function_one(self):
        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        span = 4
        knot = 5.0 / 2.0

        # Values from basis_function
        to_check = [helpers.basis_function_one(degree, knot_vector, span - _, knot) for _ in range(degree, -1, -1)]
        result = helpers.basis_function(degree, knot_vector, span, knot)

        self.assertAlmostEqual(to_check[0], result[0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[1], result[1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[2], result[2], delta=GEOMDL_DELTA)

    def test_basis_function_ders_one(self):
        degree = 2
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5]
        span = 4
        knot = 5.0 / 2.0
        order = 2

        to_check = [
            helpers.basis_function_ders_one(degree, knot_vector, span - _, knot, order) for _ in range(degree, -1, -1)
        ]
        interm = helpers.basis_function_ders(degree, knot_vector, span, knot, order)
        result = [list(_) for _ in zip(*interm)]

        self.assertAlmostEqual(to_check[0][0], result[0][0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[0][1], result[0][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[0][2], result[0][2], delta=GEOMDL_DELTA)

        self.assertAlmostEqual(to_check[1][0], result[1][0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[1][1], result[1][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[1][2], result[1][2], delta=GEOMDL_DELTA)

        self.assertAlmostEqual(to_check[2][0], result[2][0], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[2][1], result[2][1], delta=GEOMDL_DELTA)
        self.assertAlmostEqual(to_check[2][2], result[2][2], delta=GEOMDL_DELTA)


if __name__ == "__main__":
    unittest.main()

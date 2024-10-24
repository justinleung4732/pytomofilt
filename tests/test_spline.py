import unittest
import numpy as np
import numpy.testing as npt

import pytomofilt.spline as spl

class TestClampedCubicSpline(unittest.TestCase):

    def test_less_than_three(self):
        with self.assertRaises(AssertionError):
            spl.clamped_cubic_spline([1,2], [1,2])


    def test_not_ascending_order(self):
        with self.assertRaises(AssertionError):
            spl.clamped_cubic_spline([1,5,6,2,3], [1,2,3,4,5])


    def test_different_x_y_length(self):
        x = np.array([1,2,3,4,5])
        y = np.array([[3,2,1,5], [1,2,3,4]])

        with self.assertRaises(AssertionError):
            spl.clamped_cubic_spline(x,y)


    def test_spline(self):
        x = np.array([1,2,3,4,5])
        y = np.array([[1,2,3,4,5], [5,2,3,1,0]]).T
        ccs = spl.clamped_cubic_spline(x,y)

        npt.assert_array_equal(ccs.x, x)
        npt.assert_array_equal(ccs(x[0]), [1,5])


class TestCalculateSpline(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestCalculateSpline, self).__init__(*args, **kwargs)
        
        # Create a CubicSpline object
        knots = np.array([1,2,6,8,10.0,13.2,15.7,18.0])
        self.splines = spl.calculate_splines(knots)


    def test_knot_points(self):
        npt.assert_array_equal(self.splines(8), [0,0,0,1,0,0,0,0])
    
    
    def test_sum_to_one(self):
        self.assertEqual(sum(self.splines(3.1235)), 1)


class TestCubicSpline(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestCubicSpline, self).__init__(*args, **kwargs)
        
        # Create a CubicSpline object to use for testing cubic_spline function
        knots = np.array([1,2,6,8,10.0,13.2,15.7,18.0])
        self.splines = spl.calculate_splines(knots)


    def test_same_knots_and_depth(self):
        pts = np.array([1,2,6,8,10.0,13.2,15.7,18.0])
        spl_mtx = np.diag([1,3,5,2,3,5.5,2.3,8.0])
        coefs = spl.cubic_spline(spl_mtx, pts, self.splines)

        # Accurate to 10 decimal places
        npt.assert_array_almost_equal(coefs, spl_mtx, 10) 
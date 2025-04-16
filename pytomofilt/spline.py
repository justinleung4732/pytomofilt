import scipy.interpolate as spi
import scipy.linalg as spl
import numpy as np

# Function to calculate 3-point derivative clamped spline
def clamped_cubic_spline(x, y):
    """ 
    Creates a Scipy CubicSpline object with a 3-point clamped spline condition.
    The CubicSpline object can take an argument of a value x and output the value
    y of the spline at x.

    Parameters
    ----------
    x : array_like (n)
        1-D array containing values of the independent variable. Values must be real,
        finite and in strictly increasing order.
    y : array_like (n, k)
        Array containing values of the dependent variable. The length along the first 
        axis must match the length of x. Values must be finite.

    Returns
    -------
    scipy.interpolate.CubicSpline object
        A callable object that returns the value of the splines at the called location.
    """
    # Check that x must at least have 3 entries
    assert len(x) >= 3, "x must have 3 or more entries"
    # Check that entries in x are in ascending order
    assert np.all(sorted(x) == x), "List must be in ascending order" 
    # Check that length of x == length of y at axis = 0
    assert len(x) == y.shape[0], "length of y at axis 0 must be the same length as x"

    start_deriv = (y[0]*(2*x[0]-x[1]-x[2])*(x[2]-x[1]) + y[1]*(x[2]-x[0])**2 - y[2]*(x[1] - x[0])**2)/((x[1] - x[0]) * (x[2] - x[0]) * (x[2] - x[1]))
    end_deriv = (y[-1]*(2*x[-1]-x[-2]-x[-3])*(x[-3]-x[-2]) + y[-2]*(x[-3]-x[-1])**2 - y[-3]*(x[-2] - x[-1])**2)/((x[-2] - x[-1]) * (x[-3] - x[-1]) * (x[-3] - x[-2]))
    ccs = spi.CubicSpline(x, y, bc_type=((1, start_deriv),(1, end_deriv)), extrapolate=None)
    
    return ccs

# Cubic interpolation on spline knots (21)
def calculate_splines(knots):
    """
    Calculate_splines calculates the cubic splines based on the knot points provided

    Parameters
    ----------
    knots : list (n)
        A list of locations at which to anchor the splines at. Entries need to be in
        ascending order

    Returns
    -------
    scipy.interpolate.CubicSpline object
        A callable object that returns the weights of each of the splines at the called location.
    """
    spl = np.identity(len(knots))
    return clamped_cubic_spline(knots,spl)

def cubic_spline(coefs, depth, splines):
    """
    Cubic_spline calculates coefficients at the knot points

    Parameters
    ----------
    coefs : array (n,)
        An array containing the coefficients evaluated at each depth. The first axis
        must be the same length as depth.
    
    depth: list (n)
        A list containing values of the depth of each layer.

    splines: scipy.interpolate.CubicSpline object
        A callable CublineSpline object, which is the output from the calculate_splines
        function above. It returns the weights of each of the splines at the called 
        location.

    Returns
    -------
    coef_rts: np.ndarray
        An array of coefficients evaluated at each of the spline knots.
    """

    assert len(depth) == coefs.shape[0], "first axis of coefs array must be the \
                                          same length as depth"
    assert len(depth) >= len(splines.x), f"Number of depth points must be equal \
                                          or more than {len(splines.x)}"

    if coefs.ndim > 2:
        coefs_redim = coefs.reshape((len(depth), -1))
    else:
        coefs_redim = coefs

    wgt_layers = splines(depth)
    coef_rts = spl.lstsq(wgt_layers,coefs_redim)[0]

    if coef_rts.ndim == 1:
        coef_rts = coef_rts[:,None] # mandating that matrix must be 2 dimensional
    elif coefs.ndim > 2:
        coef_rts = coef_rts.reshape((wgt_layers.shape[-1],) + coefs.shape[1:])

    return coef_rts
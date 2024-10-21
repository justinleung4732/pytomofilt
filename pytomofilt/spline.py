import scipy.interpolate.CubicSpline as CubicSpline
import scipy.linalg

# Function to calculate 3-point derivative clamped spline
def clamped_cubic_spline(x, y):
    """ 
    Creates a Scipy CubicSpline object with a 3-point clamped spline condition.
    The CubicSpline object can take an argument of a value x and output the value
    y of the spline at x.

    Parameters
    ----------
    x : list (n)
        1-D array containing values of the independent variable. Values must be real,
        finite and in strictly increasing order.
    y : list (n, k)
        Array containing values of the dependent variable. The length along the first 
        axis must match the length of x. Values must be finite.

    Returns
    -------
    scipy.interpolate._cubic.CubicSpline object
        A callable object that returns the value of the splines at the called location.
    """
    start_deriv = (y[0]*(2*x[0]-x[1]-x[2])*(x[2]-x[1]) + y[1]*(x[2]-x[0])**2 - y[2]*(x[1] - x[0])**2)/((x[1] - x[0]) * (x[2] - x[0]) * (x[2] - x[1]))
    end_deriv = (y[-1]*(2*x[-1]-x[-2]-x[-3])*(x[-3]-x[-2]) + y[-2]*(x[-3]-x[-1])**2 - y[-3]*(x[-2] - x[-1])**2)/((x[-2] - x[-1]) * (x[-3] - x[-1]) * (x[-3] - x[-2]))
    ccs = CubicSpline(x, y, bc_type=((1, start_deriv),(1, end_deriv)), extrapolate=None)
    
    return ccs

# Cubic interpolation on spline knots (21)
def calculate_splines(knots):
    """
    Calculate_splines calculates the cubic splines based on the knot points provided

    Parameters
    ----------
    knots : list (n)
        A list of locations at which to anchor the splines at.

    Returns
    -------
    scipy.interpolate._cubic.CubicSpline object
        A callable object that returns the weights of each of the splines at the called location.
    """
    spl = np.zeros((len(knots),len(knots)))
    np.fill_diagonal(spl,1)
    return clamped_cubic_spline(knots,spl)

def cubic_spline(coefs, depth, splines):
    """
    Calculate_splines calculates the cubic splines based on the knot points provided

    Parameters
    ----------
    knots : list (n)
        A list of locations at which to anchor the splines at.

    Returns
    -------
    scipy.interpolate._cubic.CubicSpline object
        A callable object that returns the weights of each of the splines at the called location.
    """
    if coefs.ndim > 2:
        coefs_redim = coefs.reshape((len(depth), -1))
    else:
        coefs_redim = coefs

    wgt_layers = splines(depth)
    coef_rts = scipy.linalg.lstsq(wgt_layers,coefs_redim)[0]

    if coef_rts.ndim == 1:
        coef_rts = coef_rts[:,None] # mandating that matrix must be 2 dimensional
    elif coefs.ndim > 2:
        coef_rts = coef_rts.reshape((wgt_layers.shape[-1],) + coefs.shape[1:])

    return coef_rts
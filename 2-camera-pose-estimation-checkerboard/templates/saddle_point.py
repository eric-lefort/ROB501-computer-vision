import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """

    size_y, size_x = I.shape
    size_total = size_y * size_x
 
    X, Y = np.meshgrid(range(size_y), range(size_x))
    X = X.ravel()
    Y = Y.ravel()

    # one row from X: x^2, xy, y^2, x, y, 1
    X = np.column_stack((np.square(X), X * Y, np.square(Y), X, Y, np.ones(size_total)))
    
    I = I.ravel()

    # linear lstsq solve X * phi = I

    # normal equations
    theta = inv(X.T @ X) @ (X.T @ I)
    alpha, beta, gamma, delta, epsilon, zeta = theta

    A = -np.array([[2*alpha, beta], [beta, 2*gamma]])
    b = np.array([delta, epsilon]).T
    pt = inv(A) @ b
    pt = pt.reshape((2,1))

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt
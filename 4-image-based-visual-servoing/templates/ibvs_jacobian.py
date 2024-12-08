import numpy as np

def ibvs_jacobian(K, pt, z):
    """
    Determine the Jacobian for IBVS.

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K  - 3x3 np.array, camera intrinsic calibration matrix.
    pt - 2x1 np.array, image plane point. 
    z  - Scalar depth value (estimated).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian. The matrix must contain float64 values.
    """
    # Unpack intrinsic parameters
    fx = K[0, 0]  # Focal length in x
    fy = K[1, 1]  # Focal length in y
    s = K[0, 1]   # Skew (typically 0)
    cx = K[0, 2]  # Principal point x
    cy = K[1, 2]  # Principal point y

    # Subtract principal points to get normalized coordinates
    x = (pt[0,0] - cx)
    y = (pt[1,0] - cy)

    fy = fx

    # Jacobian w.r.t. translation
    J_t = np.array([
        [-fx / z,    0,          x / z],
        [0,          -fy / z,    y / z]
    ], dtype=np.float64)

    # Jacobian w.r.t. rotation
    J_r = np.array([
        [ x * y / fx,           -(fx**2 + x**2) / fx,      y],
        [(fy**2 + y**2) / fy,    -x * y / fy,             -x]
    ], dtype=np.float64)


    # Combine the Jacobians
    J = np.hstack((J_t, J_r))


    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J
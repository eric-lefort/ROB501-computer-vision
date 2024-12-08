import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_depth_finder(K, pts_obs, pts_prev, v_cam):
    """
    Compute estimated 

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    pts_obs  - 2xn np.array, observed (current) image plane points.
    pts_prev - 2xn np.array, observed (previous) image plane points.
    v_cam    - 6x1 np.array, camera velocity (last commmanded).

    Returns:
    --------
    zs_est - nx0 np.array, updated, estimated depth values for each point.
    """
    n = pts_obs.shape[1]
    J = np.zeros((2*n, 6))

    # start with 1 as guess for zs 
    zs_est = 1 * np.ones(n)

    v = v_cam[:3, 0]
    w = v_cam[3:, 0]

    J = np.zeros((2 * n, 6))

    for i in range(n):
        J = ibvs_jacobian(K, pts_obs[:, i].reshape(2,1), zs_est[i])
    
        # from the Corke text
        Jv = J[:, :3]
        Jw = J[:, 3:]
        
        # Jv v
        A = (Jv @ v).reshape((2,1))

        # u_dot, v_dot - Jw w
        b = (pts_obs[:, i] - pts_prev[:, i] - Jw @ w).reshape((2,1))

        # Solve lstsq, using pseudoinverse
        A_pinv = np.linalg.inv(A.T @ A) @ A.T
        sol = A_pinv @ b

        zs_est[i] = 1/sol[0][0]

    correct = isinstance(zs_est, np.ndarray) and \
        zs_est.dtype == np.float64 and zs_est.shape == (n,)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return zs_est
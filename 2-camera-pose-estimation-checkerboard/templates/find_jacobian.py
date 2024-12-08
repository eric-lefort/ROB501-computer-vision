import numpy as np
from numpy.linalg import inv

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Note that the homogeneous transformation matrix provided defines the
    transformation from the *camera frame* to the *world frame* (to 
    project into the image, you would need to invert this matrix).

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
         The Jacobian must contain float64 values.
    """

    # potential problems: Wpt - t, not correct, offset is not C^T * t
    # - dx_dv, dy_dv not correct/necessary
    # - misunderstand general process here p bar -> 3d homogeneous -> normalize by z (planar)

    # extract C and t from Twc
    Cwc = Twc[:3, :3]
    Ccw = Cwc.T
    t = Twc[:3, 3].reshape((3,1))
    Wpt = Wpt.reshape((3,1))

    # derivative dv/dtx, dv/dty, dv/dtz
    dv_dtx = -K @ Ccw[:, 0].reshape((3,1))
    dv_dty = -K @ Ccw[:, 1].reshape((3,1))
    dv_dtz = -K @ Ccw[:, 2].reshape((3,1))

    phi, theta, psi = rpy_from_dcm(Cwc)
    Cx = dcm_from_rpy((phi, 0, 0))
    Cy = dcm_from_rpy((0, theta, 0))
    Cz = dcm_from_rpy((0, 0, psi))

    # dv_dphi, dv_dtheta, dv_dpsi
    dv_dphi = K @ (Cz @ Cy @ dC_dangle('x', phi)).T @ (Wpt - t)
    dv_dtheta = K @ (Cz @ dC_dangle('y', theta) @ Cx).T @ (Wpt - t)
    dv_dpsi = K @ (dC_dangle('z', psi) @ Cy @ Cx).T @ (Wpt - t)

    dv_d_params = np.concatenate((dv_dtx, dv_dty, dv_dtz, dv_dphi, dv_dtheta, dv_dpsi), axis=1)

    # derivatives for homogeneous normalization dx/dv and dy/dv
    v = K @ Ccw @ (Wpt - t).reshape((3,))

    dx_dv = np.array([ 1/v[2], 0, -v[0]/(v[2]*v[2]) ])
    dy_dv = np.array([ 0, 1/v[2], -v[1]/(v[2]*v[2]) ])

    dx_d_params = dx_dv @ dv_d_params
    dy_d_params = dy_dv @ dv_d_params

    J = np.concatenate((dx_d_params, dy_d_params)).reshape((2,6))

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J









def dC_dangle(axis, a):
    assert axis in ('x', 'y', 'z')

    if axis == 'x':
        return np.array([
            [0, 0, 0],
            [0, -np.sin(a).item(), -np.cos(a).item()],
            [0, np.cos(a).item(), -np.sin(a).item()]
        ])

    if axis == 'y':
        return np.array([
            [-np.sin(a).item(), 0, np.cos(a).item()],
            [0, 0, 0],
            [-np.cos(a).item(), 0, -np.sin(a).item()]
        ])

    if axis == 'z':
        return np.array([
            [-np.sin(a).item(), -np.cos(a).item(), 0],
            [np.cos(a).item(), -np.sin(a).item(), 0],
            [0, 0, 0],
        ])
    raise ValueError("incorrect axis string: 'x', 'y', or 'z'")

def dcm_from_rpy(rpy):
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])

def rpy_from_dcm(R):
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy
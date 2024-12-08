import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---
    
    # Create the matrix A
    A = np.zeros((8, 9))
    for i in range(4):
        x1, y1 = I1pts[:, i]
        x2, y2 = I2pts[:, i]
        A[2*i, :] = [-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2]
        A[2*i+1, :] = [0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2]

    # Solve for the homography matrix H
    H = null_space(A).reshape(3, 3)
    # Normalize H
    H = H / H[2,2]

    return H, A
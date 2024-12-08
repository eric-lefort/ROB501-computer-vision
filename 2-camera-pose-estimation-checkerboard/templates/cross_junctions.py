import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path


def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """

    # assume target is 8x6

    # least squares homography projecting bounding quad to an axis-aligned rectangle
    eo = 0.03 # edge offset
    target_poly = np.array([[-eo,-eo], [1+eo,-eo], [1+eo,1+eo], [-eo,1+eo]]).T
    H, _ = dlt_homography(target_poly, bpoly)

    # # pixel size of kernel
    r = 9
    n = 19
    # x = np.linspace(-1, 1, n)
    # sigma = 2
    # gx = np.exp(-x**2/(2*sigma**2))
    # kern = np.outer(gx, gx)
    # # kern /= np.sum(kern) # normalize

    nx, ny = 8, 6
    targ_pt = np.ones((3,))
    pt = np.ones((3,))

    # approx grid locations in target space
    x_targ_coords = np.linspace(0, 1, nx+2)
    y_targ_coords = np.linspace(0, 1, ny+2)
    Ipts = []
    for y in range(ny):
        targ_pt[1] = y_targ_coords[y+1]
        for x in range(nx):
            targ_pt[0] = x_targ_coords[x+1]
            # apply homography
            pt = H @ targ_pt
            pt /= pt[2]

            # refine approximate location
            pt_x, pt_y, _ = map(int, np.round(pt))
            grid = I[pt_y-r : pt_y+r+1, pt_x-r : pt_x+r+1]
            grid = np.asarray(grid, dtype='float64')
            # grid *= kern # can add a kernel to prefer a centered saddle point

            # from matplotlib import pyplot as plt 
            # plt.imshow(grid, interpolation="nearest", origin="upper")
            # plt.show()

            pt_ref = saddle_point(grid).reshape((2,)) + np.array([pt_x-r, pt_y-r])
            Ipts.append(pt_ref)

    Ipts = np.array(Ipts).T

    # from matplotlib import pyplot as plt
    # from itertools import cycle

    # # Define a cycle of colors
    # colors = cycle(['r', 'g', 'b', 'y', 'c'])  # Red, Green, Blue, Yellow, Cyan

    # # Plotting
    # for i, point in enumerate(Ipts):
    #     plt.scatter(point[0], point[1], color=next(colors), s=100)  # s=100 makes the points larger

    # plt.title("2D Points with Repeating Colors")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.grid(True)
    # plt.show()

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts


def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """   
    A = np.zeros((8, 9))
    for i in range(4):
        x1, y1 = I1pts[:, i]
        x2, y2 = I2pts[:, i]
        A[2*i, :] = [-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2]
        A[2*i+1, :] = [0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2]
    H = null_space(A).reshape(3, 3)
    H = H / H[2,2]
    return H, A


def saddle_point(I):
    """
    Locate saddle point in an image patch.
    """
    size_y, size_x = I.shape
    size_total = size_y * size_x
    X, Y = np.meshgrid(range(size_y), range(size_x))
    X = X.ravel()
    Y = Y.ravel()
    X = np.column_stack((np.square(X), X * Y, np.square(Y), X, Y, np.ones(size_total)))
    I = I.ravel()
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
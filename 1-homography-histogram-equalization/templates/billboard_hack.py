# Billboard hack script file.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image - use if you find useful.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])
    bbox_x = np.array([np.min(bbox[0, :]), np.max(bbox[0, :])])
    bbox_y = np.array([np.min(bbox[1, :]), np.max(bbox[1, :])])

    print("Test")

    # Point correspondences.
    Iyd_pts = np.array([
        [416, 485, 488, 410], 
        [40,  61, 353, 349]])
    Ist_pts = np.array([
        [2, 218, 218, 2], 
        [2, 2, 409, 409]])

    def in_quad(quad, p):
        # assume points are ordered counterclockwise from lower left
        p0 = quad[:, 0]
        p1 = quad[:, 1]
        p2 = quad[:, 2]
        p3 = quad[:, 3]

        # consider lines 01, 12, 23, 30
        x, y = p
        # above 01
        cond0 = (p1[1] - p0[1]) * (x - p0[0]) - (p1[0] - p0[0]) * (y - p0[1]) < 0

        # left of 12
        cond1 = (p2[1] - p1[1]) * (x - p1[0]) - (p2[0] - p1[0]) * (y - p1[1]) < 0

        # below 23
        cond2 = (p3[1] - p2[1]) * (x - p2[0]) - (p3[0] - p2[0]) * (y - p2[1]) < 0

        # right of 30
        cond3 = (p0[1] - p3[1]) * (x - p3[0]) - (p0[0] - p3[0]) * (y - p3[1]) < 0

        return cond0 and cond1 and cond2 and cond3

    Iyd = imread('../images/yonge_dundas_square.jpg')
    Ist = imread('../images/uoft_soldiers_tower_light.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    # Let's do the histogram equalization first.
    Ist = histogram_eq(Ist)

    # Compute the perspective homography we need...
    H, _ = dlt_homography(Iyd_pts, Ist_pts)
    # H_inv = np.linalg.inv(H)
    # Main 'for' loop to do the warp and insertion - 
    # this could be vectorized to be faster if needed!

    v = np.zeros((3,))
    v_src = np.zeros((3,))

    for y in range(Ihack.shape[0]):
        for x in range(Ihack.shape[1]):
            if x < bbox_x[0] or x > bbox_x[1] or \
                y < bbox_y[0] or y > bbox_y[1]:
                
                # print(f"Changing pixel {x}, {y}, not between {bbox_x} {bbox_y}")
                continue
            
            if not in_quad(Iyd_pts, (x, y)):
                continue

            v[:] = np.array((x, y, 1))
            v_src = H @ v
            v_src /= v_src[2]

            # print(f"Changing pixel {x}, {y} ->", end=None)
            Ihack[y, x] = bilinear_interp(Ist, np.array([(v_src[0], v_src[1])]).reshape((2,1)))

    # print(Ihack.shape)
    # I_idx = np.indices(Ihack[:, :, 0].shape)
    # print(I_idx.shape)
    # print(np.all([
    #     np.greater(I_idx[, :, :], bbox_x[0]),
    #     np.less(I_idx[:, :], bbox_x[1]),
    #     np.greater(I_idx[:, :], bbox_y[0]),
    #     np.less(I_idx[:, :], bbox_y[1])
    # ], axis=0).shape)

    # I_hack_mask = np.zeros(Ihack.shape)

    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!

    #------------------

    plt.imshow(Ist)
    plt.plot(Ist_pts[0, :], Ist_pts[1, :], 'ro')
    plt.show()

    # Visualize the result, if desired...
    plt.imshow(Ihack)
    # plot points over the image
    plt.plot(Iyd_pts[0, :], Iyd_pts[1, :], 'ro')
    plt.show()
    imwrite('billboard_hacked.png', Ihack);

    return Ihack



if __name__ == "__main__":
    billboard_hack()
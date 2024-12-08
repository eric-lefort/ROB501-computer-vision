import numpy as np
from scipy.ndimage.filters import *

from matplotlib import pyplot as plt

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    # (size, %error)
    # (15, 0.14)
    # (13, 0.14)
    # (11, 0.14)
    # (9, 0.15)
    # (7, 0.17)
    # (5, 0.21)

    w_size = 11
    w_half = w_size // 2

    x_range = bbox[0, :]
    y_range = bbox[1, :]

    print(Il.shape)
    print(bbox)

    Id = np.zeros_like(Il)
    for i in range(x_range[0], x_range[1] + 1):
        for j in range(y_range[0], y_range[1] + 1):
            x_lo = i - w_half
            x_hi = i + w_half + 1
            y_lo = j - w_half
            y_hi = j + w_half + 1

            window_left = Il[y_lo:y_hi, x_lo:x_hi] / 255
            
            # scan for match in right window
            min_err = float('inf')
            for offset in range(maxd):
                # print((x_lo - offset), (x_hi - offset))
                if (x_lo - offset < 0):
                    break
                window_right = Ir[y_lo:y_hi, (x_lo - offset) : (x_hi - offset)] / 255
                cur_err = sum_absolute_diff(window_left, window_right)
                if cur_err < min_err:
                    d = offset
                    min_err = cur_err

                # err.append((offset, sum_absolute_diff(window_left, window_right)))

            Id[j, i] = d 
            # draw = Il
            # draw[j, i] = 0
            # plt.imshow(draw)
            # plt.show()
            # plt.imshow(Ir)
            # plt.show()
            # plt.scatter(np.array(err)[:, 0], np.array(err)[:, 1])
            # plt.show()

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape
    
    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id

def sum_absolute_diff(im1, im2):
    return np.sum(np.abs(im1 - im2))
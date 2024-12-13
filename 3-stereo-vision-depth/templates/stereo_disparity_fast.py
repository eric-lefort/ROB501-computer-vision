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

    # inital results with different window sizes
    # (size, %error)
    # (15, 0.14)
    # (13, 0.14)
    # (11, 0.14)
    # (9, 0.15)
    # (7, 0.17)
    # (5, 0.21)

    imshape = Il.shape

    w_size = 11
    w_half = w_size // 2

    # pad the image with half the window size, such that a window centered at any 
    # point on the original image is completely contained by the padded image
    pad = w_half
    x_range = bbox[0, :] + pad
    y_range = bbox[1, :] + pad

    Id = np.zeros_like(Il)
    Il = np.pad(Il, pad_width=((pad, pad), (pad, pad)), mode='edge') / 255
    Ir = np.pad(Ir, pad_width=((pad, pad), (pad, pad)), mode='edge') / 255

    for i in range(y_range[0], y_range[1] + 1):
        for j in range(x_range[0], x_range[1] + 1):
            x_lo = j
            x_hi = j + w_size
            y_lo = i
            y_hi = i + w_size

            window_left = Il[y_lo:y_hi, x_lo:x_hi]
            
            # generate all shifted windows (for all disparities)
            right_windows = np.stack([
                Ir[i:i+w_size, j-d:j-d+w_size] if j-d >= 0 else np.full((w_size, w_size), np.inf)
                for d in range(maxd + 1)
            ], axis=0)  # Shape: (maxd+1, w_size, w_size)

            # compute SAD for each window
            sad = np.sum(np.abs(right_windows - window_left), axis=(1, 2))

            Id[i, j] = np.argmin(sad)

    correct = isinstance(Id, np.ndarray) and Id.shape == imshape
    
    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id
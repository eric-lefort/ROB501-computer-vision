import numpy as np
from scipy.ndimage import *
from matplotlib import pyplot as plt

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

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


    ####################
    # ALGORITHM
    # This algorithm is similar to the basic matching strategy,
    # however it implements additional filters in order to improve
    # performance.
    # 
    # Filters I experimented with
    # - gaussian blur on input images
    # - gaussian blur on disparity map
    # - laplacian of gaussian on input images
    
    # Noticing many "holes" where the algorithm assigned a zero 
    # disparity, I tried some filters to reduce this issue
    # - mean filter on disparity map
    # - median filter on disparity map

    # FINAL ALGORITHM
    # Laplacian of gaussian with sigma = 0.5 on input images. This is inspired by general feature matching algorithms which perform well with edges.
    # Median filter with filter size 11 on disparity map.
    ####################


    imshape = Il.shape
    w_size = 11
    w_half = w_size // 2

    # pad the image with half the window size, such that a window centered at any 
    # point on the original image is completely contained by the padded image
    pad = w_half
    x_range = bbox[0, :]
    y_range = bbox[1, :]

    Id = np.zeros_like(Il)
    Il = np.pad(Il, pad_width=((pad, pad), (pad, pad)), mode='edge') / 255
    Ir = np.pad(Ir, pad_width=((pad, pad), (pad, pad)), mode='edge') / 255

    # laplacian of gaussian
    plt.imsave("input.png", Il, cmap="gray")
    
    prev = Il
    Il = gaussian_laplace(Il, 0.1)
    Ir = gaussian_laplace(Ir, 0.5)
    Il = gaussian_laplace(prev, 0.1)
    plt.imsave("output_01.png", Il, cmap="gray")
    Il = gaussian_laplace(prev, 0.5)
    plt.imsave("output_05.png", Il, cmap="gray")
    Il = gaussian_laplace(prev, 1)
    plt.imsave("output_10.png", Il, cmap="gray")

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
    
    # here we copy the edge pixels to the edge of the screen to prevent 
    # unwanted effects from the median filter at the edge of the bounding box
    xmin, xmax = x_range
    ymin, ymax = y_range
    region = Id[ymin:ymax+1, xmin:xmax+1]
    Id = np.pad(region, 
                pad_width=((ymin, Id.shape[0] - ymax - 1), 
                           (xmin, Id.shape[1] - xmax - 1)), 
                mode='edge')
    Id = median_filter(Id, (11,11))

    correct = isinstance(Id, np.ndarray) and Id.shape == imshape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id
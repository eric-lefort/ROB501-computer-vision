import numpy as np
from numpy.linalg import inv

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    four pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    # print(f"i: {pt[0]} \t j: {pt[1]}")
    i = int(pt[0])
    j = int(pt[1])
    dx = pt[0] - i
    dy = pt[1] - j

    b = I[j, i] * (1 - dx) * (1 - dy) + \
        I[j + 1, i] * (1 - dx) * dy + \
        I[j, i + 1] * dx * (1 - dy) + \
        I[j + 1, i + 1] * dx * dy

    #------------------

    return b
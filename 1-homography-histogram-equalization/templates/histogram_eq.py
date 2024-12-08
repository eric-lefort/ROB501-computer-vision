import numpy as np
import matplotlib.pyplot as plt
import imageio

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---
    #------------------

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    clr_range = (0, 255)

    # Compute histogram.
    hist, bins = np.histogram(I.flatten(), bins=256, range=clr_range)

    # Compute cumulative distribution.
    cdf = hist.cumsum()

    cdf = cdf / cdf[-1]

    # # plot histogram
    # plt.plot(hist)
    # plt.show()

    # plt.plot(cdf)
    # plt.show()

    J = np.empty_like(I)

    for i in range(*clr_range):
        # i: starting intensity value
        # cdf[i] * 255: ending intensity value
        J[I == i] = round(cdf[i] * 255)

    # # draw I and J
    # plt.imshow(I, cmap='gray')
    # plt.show()
    # plt.imshow(J, cmap='gray')
    # plt.show()

    return J

import numpy as np
import matplotlib.pyplot as plt
from imageio.v3 import imread
from stereo_disparity_score import stereo_disparity_score
from stereo_disparity_best import stereo_disparity_best

# Load the stereo images and ground truth.
# The cones and teddy datasets have ground truth that is 'stretched'
# to fill the range of available intensities - here we divide to undo.
Il = imread("../images/cones/cones_image_02.png", mode='F')
Ir = imread("../images/cones/cones_image_06.png", mode='F')
It = imread("../images/cones/cones_disp_02.png",  mode='F')/4.0
bbox = np.load("../data/cones_02_bounds.npy")

# Il = imread("../images/teddy/teddy_image_02.png", mode='F')
# Ir = imread("../images/teddy/teddy_image_06.png", mode='F')
# It = imread("../images/teddy/teddy_disp_02.png", mode='F')/4.0
# bbox = np.load("../data/teddy_02_bounds.npy")

# Il = imread("../images/kitti/image_0/000070_10.png", mode='F')
# Ir = imread("../images/kitti/image_1/000070_10.png", mode='F')
# It = imread("../images/kitti/disp_occ/000070_10.png", mode='F')
# bbox = np.load("../data/kitti_070_bounds.npy")


# Load the appropriate bounding box.

Id = stereo_disparity_best(Il, Ir, bbox, 55)
N, rms, pbad = stereo_disparity_score(It, Id, bbox)
print("Valid pixels: %d, RMS Error: %.2f, Percentage Bad: %.2f" % (N, rms, pbad))
plt.imshow(Id, cmap = "gray")
plt.show()
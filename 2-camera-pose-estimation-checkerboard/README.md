# ROB501: Computer Vision for Robotics

## Assignment 2: Camera Pose Estimation

### Fall 2024

---

### Overview

Camera pose estimation is a fundamental task in robotic vision, allowing us to determine the position and orientation of a camera relative to the environment. This assignment focuses on estimating the pose of a camera relative to a planar checkerboard calibration target. You will:

- Gain hands-on experience with image smoothing and subpixel feature extraction.
- Understand and implement nonlinear least squares optimization.

The checkerboard target consists of squares that are 63.5 mm on each side, and you will extract 2D-3D correspondences to compute the camera pose. The target is assumed to be planar, and intrinsic camera parameters are provided.

### Submission Details

- **Due Date**: Friday, October 18, 2024, by 11:59 p.m. EDT
- **Submission Platform**: Autolab
- **Programming Language**: Python 3
- You may submit multiple attempts until the deadline.

Ensure your code is properly commented and adheres to the provided templates. Submissions will be tested for accuracy and clarity.

---

### Tasks Overview

1. **Image Smoothing and Subpixel Feature Extraction**:
   Implement a function to compute saddle points of cross-junctions in a checkerboard image using the method described by Lucchese and Mitra.

2. **Cross-Junction Detection**:
   Extract and refine all cross-junctions in a target image to subpixel accuracy.

3. **Jacobian Computation**:
   Derive and implement a function to compute the Jacobian matrix for image plane observations with respect to camera pose parameters.

4. **Pose Estimation**:
   Solve the Perspective-n-Points (PnP) problem using nonlinear least squares to estimate the camera pose.

---

### Resources

1. Lecture Notes and Slides
2. Szeliski’s Textbook (Sections on Camera Models and Pose Estimation)
3. Lucchese and Mitra’s Paper on Subpixel Feature Detection (included in the assignment archive)

---

### Submission Checklist

- [ ] `saddle_point.py`
- [ ] `cross_junctions.py`
- [ ] `find_jacobian.py`
- [ ] `pose_estimate_nls.py`

Test your functions thoroughly using the provided sample images and ensure compatibility with the Autolab testing suite.


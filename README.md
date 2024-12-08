# ROB501 Assignments

This repository contains the code and solutions for the assignments given by Professor Jonathan Kelly in ROB501, in Fall 2024 semester. These assignments go over a number of computer vision concepts and algorithms, listed below.

## Assignment 1: Image Transformations & Billboard Hacking

1. **Perspective Transformations**  
   - Compute the perspective homography using the DLT algorithm to map points between images.

2. **Bilinear Interpolation**  
   - Implement interpolation for subpixel accuracy during image warping.

3. **Histogram Equalization**  
   - Enhance the contrast of grayscale images.

4. **Billboard Hacking**  
   - Combine all components to create a realistic composite image replacing the billboard.

---

## Assignment 2: Camera Pose Estimation

1. **Image Smoothing and Subpixel Feature Extraction**  
   - Implement a function to compute saddle points of cross-junctions in a checkerboard image using the method described by Lucchese and Mitra.

2. **Cross-Junction Detection**  
   - Extract and refine all cross-junctions in a target image to subpixel accuracy.

3. **Jacobian Computation**  
   - Derive and implement a function to compute the Jacobian matrix for image plane observations with respect to camera pose parameters.

4. **Pose Estimation**  
   - Solve the Perspective-n-Points (PnP) problem using nonlinear least squares to estimate the camera pose.

---

## Assignment 3: Stereo Correspondence Algorithms

1. **Your Secret Identifier**  
   - Create a function in `secret_id.py` that returns a polite secret ID string (max 32 characters). This ID will be used on the leaderboard tracking erformance.

2. **Fast Local Correspondence Algorithms**  
   - Implement a simple stereo correspondence algorithm using a fixed window size and the Sum of Absolute Differences (SAD) similarity measure.  
   - Match pixels with a winner-take-all strategy and ensure your function handles bounding boxes and maximum disparity constraints.  
   - Evaluate performance using the provided RMS error function and compare your disparity maps against ground truth data from the Middlebury Cones and Teddy datasets.

3. **A Different Approach**  
   - Develop and implement an alternative stereo correspondence algorithm, such as one using global information or an alternative local matching technique.  
   - Aim to exceed the performance of your Part 1 implementation, evaluated using the RMS error and percentage of correct disparities.

---

## Assignment 4: Image-Based Visual Servoing

1. **Image-Based Jacobian**  
   - Implement a function in `ibvs_jacobian.py` to compute the velocities of a point on the image plane, converting image coordinates to normalized image plane coordinates.

2. **Image-Based Servo Controller**  
   - Implement a proportional controller in `ibvs_controller.py` that calculates camera velocities using the Jacobian matrix and a Moore-Penrose pseudo-inverse.

3. **Depth Estimation**  
   - Implement a function in `ibvs_depth_finder.py` to estimate new depths for feature points based on camera velocity and changes in image feature positions.

4. **Performance Evaluation**  
   - Experiment with gain values and evaluate the IBVS controller's performance with exact vs. estimated depth values. Submit a report with optimal gain values and an analysis of the impact of depth errors.

---

## Assignment 5: Deep Visual Sun Sensing

1. **Network Setup and Baseline Model**  
   - Implement a CNN to classify images from the KITTI dataset into azimuth angle bins (8 bins, 45-degree intervals).  
   - The provided script `sun_cnn.py` will help you get started with the baseline model.  
   - Evaluate the model using the provided training, validation, and test datasets.  
   - Visualize training performance through loss and error curves.

2. **Improving Network Performance (Part 1)**  
   - Modify the network to improve validation set accuracy. Suggested improvements include:  
     - Adding more layers to increase network depth.  
     - Applying batch normalization to improve gradient flow.  
     - Zero-centering the input data.  
     - Implementing dropout for regularization.  
   - After training, save the best model (`best_model_45.pth`) and test it using `test_sun_cnn.py`.  
   - Submit the modified network file (`sun_cnn_45.py`) and the test predictions (`predictions_45.txt`).  
   - Achieve at least 70% accuracy (less than 30% top-1 error).

3. **Expanding Azimuth Bins (Part 2)**  
   - Modify the network to use 20-degree azimuth bins (18 total bins).  
   - Update the `sun_cnn.py` file and train the modified network.  
   - After training, evaluate it using `test_sun_cnn.py` and save predictions (`predictions_20.txt`).  
   - Submit the modified network file (`sun_cnn_20.py`) and the test predictions.  
   - Achieve at least 60% accuracy (less than 40% top-1 error).

---

### Compression Instructions

To compress your code files, use the following command:

```
tar cvf code.tar *.py
```

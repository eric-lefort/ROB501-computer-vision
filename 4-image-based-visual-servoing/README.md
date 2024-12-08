## ROB501: Computer Vision for Robotics  
### Assignment 4: Image-Based Visual Servoing  
**Fall 2024**

### Overview
In this assignment, you will implement an Image-Based Visual Servoing (IBVS) controller, which uses feedback from image plane features to control camera motion. The focus will be on Jacobian matrices, pseudo-inverse solutions, proportional control, and the impact of 3D point depth errors.

---

### Tasks Overview

1. **Part 1: Image-Based Jacobian**
   - Implement a function in `ibvs_jacobian.py` to compute the velocities of a point on the image plane, converting image coordinates to normalized image plane coordinates.

2. **Part 2: Image-Based Servo Controller**
   - Implement a proportional controller in `ibvs_controller.py` that calculates camera velocities using the Jacobian matrix and a Moore-Penrose pseudo-inverse.
   - ![Video](../assets/video/ibvs_jac_controller.webm)

3. **Part 3: Depth Estimation**
   - Implement a function in `ibvs_depth_finder.py` to estimate new depths for feature points based on camera velocity and changes in image feature positions.

4. **Part 4: Performance Evaluation**
   - Experiment with gain values and evaluate the IBVS controller's performance with exact vs. estimated depth values. Submit a report with optimal gain values and an analysis of the impact of depth errors.

---

### Submission Instructions
- Submit code via Autolab, with clear comments.
- Multiple submissions are allowed before the deadline.

---

### Resources
- Corke, R. (2024). *Robotics, Vision and Control: Fundamental Algorithms in MATLAB* (2nd ed.).
- Relevant resources on Jacobians, least squares, and pseudo-inverse techniques.

---  
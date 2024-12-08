# ROB501: Computer Vision for Robotics  
## Assignment 5: Deep Visual Sun Sensing  
**Fall 2024**

### Overview
In this assignment, you will apply deep learning (specifically Convolutional Neural Networks, or CNNs) to predict the azimuth angle of the sun from a single RGB image, using the KITTI dataset. This task introduces fundamental CNN concepts and provides hands-on experience with the PyTorch framework. The assignment involves training a CNN to classify images into discretized bins of sun azimuth angles, improving the network's performance, and experimenting with different azimuth bin sizes.

The project submission is due on **Wednesday, December 4, 2024, by 11:59 p.m. EST**.

Note: you must extract the `data.tar.gz` file before running the training.

Compress: 
```bash
tar -czvf data.tar.gz test.mat train.mat val.mat
split -b 100m data.tar.gz data.tar.gz.part_
```

Decompress:
```bash
cat data.tar.gz.part_* > data.tar.gz
tar -xzvf data.tar.gz
```

### Tasks Overview

1. **Network Setup and Baseline Model**  
   - Implement a CNN to classify images from the KITTI dataset into azimuth angle bins (8 bins, 45-degree intervals).  
   - The provided script `sun_cnn.py` will help you get started with the baseline model.  
   - Evaluate the model using the provided training, validation, and test datasets.  
   - You will visualize training performance through loss and error curves.  

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

### KITTI Dataset
The KITTI dataset consists of images and sensor data collected from a moving vehicle in Karlsruhe, Germany. It contains RGB images along with inertial and lidar measurements. For this project, you will work with a 4,541-image sequence, using 80% for training, 16% for validation, and 4% for testing.

### PyTorch Framework
PyTorch is an open-source machine learning library used for training neural networks. The assignment requires using PyTorch to implement and train the CNN. While training on a GPU is preferred for speed, it is not necessary for this assignment.

### Network Structure
The baseline network consists of:
- A convolutional layer (9x9 filter, 16 channels)
- A max-pooling layer (3x3 window, stride of 3)
- A rectified linear unit (ReLU) activation
- Another convolutional layer (5x8 filter, with output corresponding to the azimuth bins)

You can modify this architecture to improve performance.

### Performance Improvements
- **Depth**: Adding more convolutional layers and max-pooling layers.  
- **Batch Normalization**: Helps improve training stability, especially for deeper networks.  
- **Zero-Centering**: Preprocess the data by subtracting the mean of the training set.  
- **Dropout**: Prevent overfitting by randomly disabling connections during training.  
- **Learning Rate Tuning**: Experiment with different learning rates to improve convergence.  

### Submission
- **Part 1**: Submit `sun_cnn_45.py` and `predictions_45.txt` with at least 70% accuracy.  
- **Part 2**: Submit `sun_cnn_20.py` and `predictions_20.txt` with at least 60% accuracy.

### Additional Resources
- PyTorch documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)  
- KITTI dataset: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)  
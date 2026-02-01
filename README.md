# CIFAR-10 Image Classification with a CNN Built from Scratch

## Overview

This project implements a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset as part of a university course assignment. The model was built from scratch under a strict constraint of **at most 4 million parameters**. This limitation was imposed to encourage efficient architecture design and to ensure fair comparison across students, as evaluation was based on classification accuracy. The code was developed and executed using Google Colab.

## Data

This project uses the CIFAR-10 dataset, consisting of 60,000 32 $\times$ 32 color images across 10 object classes, with 6000 images per class. The dataset is publicly available at: <https://www.cs.toronto.edu/~kriz/cifar.html>

The dataset was split into 45,000 images for training, 5,000 images for validation, and 10,000 images for testing. Image pixel values were normalized to the range [0,1] to improve numerical stability during training.

## Methods

A CNN was built from scratch using TensorFlow to perform image classification on the CIFAR-10 dataset. The model construction followed these steps:

-   A sequential model was initialized, with the input defined as an RGB image of size 32 $\times$ 32 pixels.

-   The network consists of 10 convolutional layers for feature extraction, with increasing numbers of filters (64, 128, and 256). All convolutional layers use 3 $\times$ 3 kernels with ReLU activation, and L2 weight regularization is applied throughout. Max pooling layers with a 2 $\times$ 2 pool size and dropout layers with increasing dropout rates (0.1, 0.2, and 0.3) are added after the 3rd, 6th, 8th, 9th, and 10th convolutional layers.

-   The extracted feature maps are flattened using a flatten layer and passed to a fully connected (FC) layer with 1024 units and L2 weight regularization, followed by batch normalization, ReLU activation, and a dropout layer with a dropout rate of 0.4.

-   Finally, the output layer is defined as a FC layer with 10 units and softmax activation, producing class probability scores.

The resulting model contains a total of 2,898,250 parameters. It was trained using the Stochastic Gradient Descent (SGD) optimizer with a learning rate of 0.001 and momentum of 0.9, together with Sparse Categorical Cross-Entropy loss. Training was performed with a batch size of 64 for 30 epochs. A ReduceLROnPlateau callback helps decrease the learning rate by a factor of 0.6 whenever the validation loss failed to improve for two consecutive epochs. Model performance was monitored using accuracy and loss on both the training and validation sets across all epochs.

## Results

Training and validation loss and accuracy curves showed steadily decreasing loss and increasing accuracy across epochs, indicating stable convergence and reasonable generalization. Evaluation on the test set showed that the CNN model achieved an accuracy of 0.8373, demonstrating solid performance on the CIFAR-10 image classification task.

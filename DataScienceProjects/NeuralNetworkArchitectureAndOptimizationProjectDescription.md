## Neural Network Architecture and Optimization
Author: Valeria Calderon 
Created: 12/10/2025

(Project done for a DS501 class in UKY)

### Objective:
Hands on experience on designing, training and debugging neural networks. The project will use a basic Multi-Layer/
Perceptron (MLP) to a Convulutional Neural Network (CNN). This project is focused on the interpretation of training curves.

Dataset used: Fashion-MNIST
Access: Available directly via PyTorch (torchvision.datasets.FashionMNIST)

### Baseline MLP:
  Input layer: Flatten the 28x28 images into a vector size of 784
  Output Layer: 10 neurons for each class with a softmaz output.
  For training I used the stochastic gradient descent and the cross-entropy loss function and trained for 20 epochs.

### Overfitting Experiment & Regularization
Intentionally create an Overfitting scenario and the fix it.
+ Forcing the overfitting: create a new model with excessive capacity and train on a small.subset of the data.
+ Regularization: Apply dropout or L2 to regulate the large model and retrain
+ Compare

### Convulutional Neural Networks(CNNs)
Implemented a LeNet-Style CNN to use translational variance inductive bias in images

All rights reserved

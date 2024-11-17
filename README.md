# CIFAR-10 Image Classification Project

This repository explores the development and evaluation of Convolutional Neural Networks (CNNs) for the CIFAR-10 dataset. The project investigates various custom and pre-trained architectures to classify images into 10 categories, aiming to optimize accuracy and generalization while minimizing overfitting.

## Project Overview

### Dataset
- **CIFAR-10**: A dataset of 60,000 32x32 color images across 10 classes:
  - Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
  - **Training set**: 50,000 images.
  - **Test set**: 10,000 images.

### Objective
- Develop and evaluate different CNN architectures to classify CIFAR-10 images accurately.
- Compare custom CNN models with pre-trained architectures such as ResNet50, EfficientNet-B0, and VGG19.
- Visualize and analyze the learning process using feature maps, kernel weights, and confusion matrices.

## Key Features

### Custom CNN Models
1. **Custom Architectures**: 
   - Experimented with varying numbers of convolutional layers, activation functions (e.g., ReLU, Leaky ReLU, SiLU), and dropout rates.
   - **Best Custom Model**:
     - Four convolutional layers.
     - Batch normalization for stability.
     - SiLU (Swish) activation for smooth learning.
     - Optimized using AdamW optimizer with weight decay.
     - Achieved **75.36% test accuracy**.

### Pre-Trained Models
1. **ResNet50**:
   - Fine-tuned the final fully connected layer while freezing pre-trained layers.
   - Achieved **82.37% test accuracy** with 1709 seconds training time.
2. **EfficientNet-B0**:
   - Customized the classifier with additional fully connected layers.
   - Achieved **80.52% test accuracy** with efficient training.


### Visualizations
- **Feature Maps**: Visualized feature maps from convolutional layers to understand feature extraction.
- **Kernel Weights**: Displayed kernel weights to analyze convolutional filters.
- **Confusion Matrix**: Evaluated class-wise predictions and common misclassifications.

## Results and Findings
- The **best test accuracy** was achieved with the ResNet50 model at **82.37%**, demonstrating the effectiveness of transfer learning.
- EfficientNet-B0 and the custom CNN also showed strong performance, with test accuracies of **80.52%** and **75.36%**, respectively.
- Smaller batch sizes increased training time but did not significantly improve accuracy.
- Visualizations highlighted the model's ability to learn key features while also revealing areas of misclassification, particularly among visually similar classes (e.g., cats and dogs).


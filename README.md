# Simple & Vision Classification

This repository contains two classification experiments implemented in PyTorch: 
**(1)** a small synthetic multi-class classification task demonstrating a simple feedforward network and decision-boundary visualization, and **(2)** an image classification pipeline (FashionMNIST) using a convolutional neural network (CNN). The notebooks/scripts include training loops, evaluation (accuracy/loss curves), and plots used in the course report.

## Files

- project2_simple_classification.ipynb — synthetic classification notebook (data generation, model, training, plots, decision boundary).
- project2_vision.ipynb — FashionMNIST CNN notebook (data loaders, ConvModel, training/validation/test loops, plots).
- data/FashionMNIST/raw/ — downloaded raw FashionMNIST files (auto-created by torchvision).
- proj2_report.pdf — written report with figures and interpretation.
- README.md — this file.

## Project Overview

- **Synthetic Classification**:
    - **Dataset**: 10,000 2-D samples produced by sklearn.make_blobs (4 centers).
    - **Model**: SimpleModel (two hidden layers, ReLU); trained with Adam.
    - **Outputs**: train/test accuracy & loss curves and a decision-boundary plot.
      
- **Vision Classification**:
    - **Dataset**: FashionMNIST (downloaded to data/).
    - **Model**: ConvModel (2 conv layers + pooling, fully connected head, dropout).
    - **Training**: Adam optimizer, L2 regularization added explicitly to loss, save best validation model.
    - **Outputs**: train/validation/test accuracy & loss curves; final test accuracy near ~90%.
 
## Results

- **Synthetic Classification**:
    - Training and test accuracies increase steadily and reach 90% after 40 epochs.
    - Loss curves decrease smoothly; a small train/test gap suggests slight overfitting but overall good fit.

- **Vision Classification**:
    - Train, validation, and test accuracies converge well; validation accuracy peaks around ~91%.
    - Final test accuracy ≈ 90% when using saved best validation model.
    - Loss curves show steady decrease with small gaps between train/val/test, indicating good generalization.

## Interpretation

- The feedforward model separates well-structured synthetic clusters and produces a clear decision surface.
- The CNN architecture performs strongly on FashionMNIST with moderate regularization (dropout + L2) and standard training choices.
- Saving best validation weights and monitoring training/validation behavior is crucial for generalization.




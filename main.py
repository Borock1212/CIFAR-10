"""
Author: Alex Shevchenko
    Project: CIFAR-10 Image Classification with TensorFlow (Low-Level API)
    Description: 
    This project implements a neural network from scratch using TensorFlow (without Keras Sequential API).
    The network is trained to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images
    across 10 classes. The architecture consists of three fully connected layers with ReLU activations and a 
    softmax output layer.
"""

import matplotlib.pyplot as plt
from data_loader import load_data
from model import initialize_model
from train import train_model


# === Load datasets ===
train_dataset, val_dataset, test_dataset, steps_per_epoch = load_data()

# === Initialize model parameters ===
parameters = initialize_model()

# === Visualization of metrics ===
# === Display Training Loss, Training Accuracy, and Validation Accuracy ===
history = train_model(train_dataset, val_dataset, parameters, steps_per_epoch)

# === Extract values for plotting ===
loss_values, acc_values, val_acc_values = zip(*history)
plt.figure(figsize=(18, 5))

# === Plot Loss === 
plt.subplot(1, 3, 1)
plt.plot(loss_values, label='Training Loss')
plt.title('Training Loss')
plt.legend()

# === Plot Train Accuracy === 
plt.subplot(1, 3, 2)
plt.plot(acc_values, label='Training Accuracy')
plt.title('Training Accuracy')
plt.legend()

# === Plot Validation Accuracy ===
plt.subplot(1, 3, 3)
plt.plot(val_acc_values, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.show()
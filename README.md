# CIFAR-10 Image Classification with TensorFlow (Low-Level API)

## Project Description

This project implements a neural network from scratch using TensorFlow's low-level API (without Keras `Sequential`). The model is designed to classify images from the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 classes. The network consists of three fully connected layers with ReLU activations and a softmax output layer for multi-class classification.

---

## 📁 **Project Structure**

```
project/
│
├── data_loader.py      # Data loading and preprocessing
├── model.py            # Definition of model parameters and forward propagation
├── train.py            # Training loop with gradient descent
├── main.py             # Main script to run the pipeline
└── README.md           # Project documentation
```

---

## 📦 **Dependencies**

* TensorFlow
* NumPy
* scikit-learn
* Matplotlib (for visualization, optional)

Install all dependencies:

```bash
pip install tensorflow numpy scikit-learn matplotlib
```

---

## 🚀 **How to Run**

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/cifar10-classifier.git
   cd cifar10-classifier
   ```

2. Run the main script:

   ```bash
   python main.py
   ```

3. Training results and accuracy metrics will be displayed in the console.

---

## 📊 **Model Architecture**

* Input layer: 32x32x3 (flattened to 3072)
* Hidden layer 1: 512 neurons, ReLU activation
* Hidden layer 2: 265 neurons, ReLU activation
* Hidden layer 3: 128 neurons, ReLU activation
* Hidden layer 4: 64 neurons, ReLU activation
* Softmax activation (for multi-class classification)

---

## ✅ **Results**

* The model achieves around **50% accuracy** on the CIFAR-10 test set after training.


---

## 📌 **Future Improvements**

* Implement Data Augmentation for more robust training
* Add Early Stopping to prevent overfitting
* Experiment with deeper architectures

---

## 👤 **Author**

* **Name:** Alex Shevchenko
* **LinkedIn:** [Link](https://www.linkedin.com/in/alex-shevchenko-411510317)
* **GitHub:** [Link](https://github.com/Borock1212)

---

Happy Coding! 🚀

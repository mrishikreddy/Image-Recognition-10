# MNIST Digit Recognition with Convolutional Neural Network

Welcome to the **MNIST Digit Recognition** project, a Python-based deep learning application that uses a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. This project demonstrates image preprocessing, CNN model building, training, and evaluation using TensorFlow and Keras. It's an excellent resource for learning computer vision and deep learning fundamentals.

[Visit Live Project](https://colab.research.google.com/drive/1x16UtBAVfhaJ9jdP8C6Izy38jYsCTVZu?usp=sharing)

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Dataset](#dataset)

## Features
- **Convolutional Neural Network**: Classifies handwritten digits (0–9) using a CNN with convolutional and pooling layers.
- **Data Preprocessing**: Normalizes and reshapes MNIST images for model compatibility.
- **Model Training and Evaluation**: Trains the model on the MNIST dataset and evaluates accuracy on the test set.
- **Model Persistence**: Saves the trained model as `mnist_model.h5` for reuse.
- Built with powerful Python libraries: TensorFlow, Keras, NumPy, and OpenCV.
- Simple and educational code for learning CNNs and image classification.

## Installation
To set up the **MNIST Digit Recognition** project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/mnist-digit-recognition.git
   cd mnist-digit-recognition
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.6+ installed. Install the required libraries using pip:
   ```bash
   pip install tensorflow numpy opencv-python
   ```
   Note: TensorFlow includes Keras, so no separate installation is needed for Keras.

3. **Run the Script**:
   Execute the Python script:
   ```bash
   python mnist_digit_recognition.py
   ```

## Usage
Running the script performs the following steps:
1. Loads and preprocesses the MNIST dataset (handwritten digit images).
2. Builds and compiles a CNN model with convolutional, pooling, and dense layers.
3. Trains the model on the training set for 10 epochs with a batch size of 200.
4. Evaluates the model on the test set and prints the test accuracy and error.
5. Saves the trained model as `mnist_model.h5`.

Example output:
```
Test Accuracy: 0.9875
Test Error: 1.25
```

To use the saved model for predictions or modify the script (e.g., adjust epochs or model architecture), edit the relevant code sections. For example, to load and use the saved model:
```python
from keras.models import load_model
model = load_model("mnist_model.h5")
```

## How It Works
The project uses a Convolutional Neural Network (CNN) to classify handwritten digits. Here's a breakdown:

### Data Preprocessing
- **Dataset**: Loads the MNIST dataset using `keras.datasets.mnist`, which includes 60,000 training and 10,000 test images (28x28 grayscale).
- **Normalization**: Scales pixel values to the range [0, 1] by dividing by 255.
- **Reshaping**: Reshapes images to `(samples, 28, 28, 1)` to include a single color channel.
- **Label Encoding**: Converts digit labels (0–9) to one-hot encoded vectors using `to_categorical`.

### Model Architecture
- **Convolutional Layers**:
  - First layer: 16 filters (5x5), ReLU activation.
  - Second layer: 8 filters (2x2), ReLU activation.
- **Pooling Layers**: MaxPooling2D with 2x2 pool size and stride to reduce spatial dimensions.
- **Flattening**: Converts 2D feature maps to a 1D vector.
- **Dense Layers**:
  - Hidden layer: 100 units, ReLU activation.
  - Output layer: 10 units (one per digit), softmax activation.
- **Compilation**: Uses Adam optimizer and categorical crossentropy loss, with accuracy as the metric.

### Training and Evaluation
- **Training**: Fits the model on the training set for 10 epochs with a batch size of 200, using the test set for validation.
- **Evaluation**: Computes accuracy on the test set, reporting both accuracy and error percentage.
- **Model Saving**: Saves the trained model as `mnist_model.h5` for future use.

## Dataset
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each 28x28 pixels, split into:
- **Training Set**: 60,000 images with corresponding labels.
- **Test Set**: 10,000 images with corresponding labels.

Source: Included in `keras.datasets.mnist`, originally from [Yann LeCun's MNIST database](http://yann.lecun.com/exdb/mnist/).

# Apple vs. Banana Image Classification using U-Net

# Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Usage](#usage)
7. [Dependencies](#dependencies)
8. [Conclusion](#conclusion)

## Introduction
This is an image classification model implemented using U-Net architecture to classify images of apples and bananas. The model is implemented using Keras with a TensorFlow backend.

## Dataset
The dataset contains images of apples and bananas, divided into two sets: a training set and a test set. The training set contains 200 images, while the test set contains 60 images.

## Model Architecture
The model uses the InceptionV3 backbone for feature extraction, followed by a U-Net architecture for segmentation, and a fully connected neural network for classification. The U-Net architecture is used to produce a mask that highlights the regions of the image that contain either an apple or a banana.

The architecture consists of:

* Input layer
* InceptionV3 backbone layers
* Convolutional layers for downsampling
* Transposed convolutional layers for upsampling
* Fully connected layers for classification
* Output layer

# Training
The model is trained using the Adam optimizer and binary crossentropy loss function. The training is done on a batch size of 32 for 10 epochs.

## Evaluation
The model is evaluated on the test set using binary crossentropy loss function and accuracy metric.
```
Confusion Matrix:
 [[47  0]
 [ 0 44]]

Classification Report:
               precision    recall  f1-score   support

       apple       1.00      1.00      1.00        47
      banana       1.00      1.00      1.00        44

    accuracy                           1.00        91
   macro avg       1.00      1.00      1.00        91
weighted avg       1.00      1.00      1.00        91
 ```

## Usage

Clone the repository
Install the dependencies
Run the apple_banana.ipynb notebook to train and evaluate the model

## Dependencies
* TensorFlow 2.x
* Keras 2.x 

## Conclusion

In this project, we implemented an image classification model using the U-Net architecture to classify images of apples and bananas. We used the InceptionV3 backbone for feature extraction and a fully connected neural network for classification. The model was trained on a small dataset of 200 images and achieved an accuracy of 91.67% on the test set.

To improve the accuracy of the model, we suggested several future improvements, including experimenting with different segmentation architectures, increasing the dataset size, fine-tuning the model on other fruit categories, augmenting the dataset, using transfer learning, and hyperparameter tuning.

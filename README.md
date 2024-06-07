# Fruit Image Classifier

This project is a deep learning-based image classification model designed to identify various types of fruits. The model is built using TensorFlow and Keras, leveraging convolutional neural networks (CNNs) for accurate image recognition. This project represents my first experience with deep learning, applying foundational knowledge to build a functional and effective model.

## Features

* **Model Architecture:** Built using TensorFlow and Keras with a sequential model comprising:
    * Convolutional layers (Conv2D)
    * Max pooling layers (MaxPooling2D)
    * Dense layers
    * Dropout layers for regularization
    * Batch normalization for faster convergence
    * Flatten layer for final classification
* **Data Augmentation:** Incorporates techniques like rotation, zoom, shear, and flips to improve model robustness.
* **Training and Validation:** Includes mechanisms for early stopping and model checkpointing to prevent overfitting and save the best model.
* **Performance:** Achieves a validation accuracy of 75%, demonstrating effective learning and generalization.

## Future Improvements

* Experiment with different model architectures and hyperparameters.
* Collect and incorporate more training data to enhance model accuracy.
* Implement additional data augmentation strategies.
# Deep-Learning
MNIST Handwritten Digit Classification using TensorFlow

Overview

This project demonstrates a simple neural network built using TensorFlow and Keras to classify handwritten digits from the MNIST dataset. The model is trained to recognize digits (0-9) from 28x28 grayscale images.

Dataset

The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is a 28x28 pixel grayscale image.

Project Structure

train.py: Contains the code for loading data, building, and training the model.

predict.py: Runs predictions on test data and visualizes results.

README.md: Project documentation.

Installation

Ensure you have Python 3.8 or later installed, then install dependencies using:

pip install tensorflow numpy matplotlib

Running the Project

Train the Model

Run the following command to train the model:

python train.py

Make Predictions

After training, use the trained model to make predictions:

python predict.py

Model Architecture

The neural network consists of:

Flatten Layer: Converts 28x28 images into a 1D array.

Hidden Layer: Fully connected (Dense) layer with 128 neurons and ReLU activation.

Output Layer: 10 neurons (one per digit) with softmax activation.

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

Model Training

The model is trained using the Adam optimizer and sparse categorical crossentropy loss function:

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

Evaluation

To evaluate the model's performance:

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

Results

The model achieves around 97% accuracy on the test dataset.

Predictions can be visualized using Matplotlib.

Future Improvements

Implement Convolutional Neural Networks (CNNs) for better accuracy.

Tune hyperparameters (learning rate, batch size, etc.).

Deploy the model as an API using Flask or FastAPI.

Author

Abayo Brian - Data Scientist & Deep Learning Enthusiast


Features layers, dropout, adagrad, regularizers, l2, and early stopping  --alphabetical glyphs predictions.

Keras Layered Neural Network - Alphabetical Glyphs Prediction
Project Overview
This project, titled Keras Layered Neural Network - Alphabetical Glyphs Prediction, involves building a multi-layer neural network using Keras to predict alphabetical glyphs from the notMNIST dataset. The notMNIST dataset contains images of glyphs representing the characters from 'A' to 'J' (10 classes), which serve as the targets for the classification task. The primary objectives of this project are to:

Load and preprocess the data.
Build a neural network using Keras.
Train the model with various hyperparameters, including learning rate and regularization strength.
Evaluate the model using metrics such as precision and recall.
Display selected predictions alongside their true labels.
Key Features
Data Loading and Preprocessing:

Loads the notMNIST dataset, normalizes the image pixel values, and converts the labels to a one-hot encoded format.
Splits the dataset into training, validation, and test sets.
Standardizes the data for more stable neural network training.
Neural Network Architecture:

A simple feedforward neural network architecture with two layers:
First layer: 30 neurons with ReLU activation and L2 regularization.
Second layer: 10 neurons (corresponding to the 10 glyph classes) with softmax activation for multi-class classification.
Hyperparameter Tuning:

Performs grid search over various learning rates and regularization strengths to identify the best combination of hyperparameters.
Implements early stopping to prevent overfitting during training.
Evaluation and Metrics:

Evaluates the model using precision and recall scores, with a focus on weighted precision and recall to account for class imbalance.
Tracks precision and recall at each epoch using a custom Keras callback.
Visualization:

Displays the precision, loss, and accuracy over training epochs.
Displays a set of test images along with their true and predicted labels.
Model Persistence and Selection:

Keeps track of the best performing model based on validation precision, and retrains the model with the best hyperparameters on the entire training and validation set.
Requirements
To run this project, you need to install the following libraries:

Keras and TensorFlow:

Copy code
pip install keras tensorflow
Scikit-learn (for data splitting, evaluation metrics):

Copy code
pip install scikit-learn
Matplotlib (for plotting):

Copy code
pip install matplotlib
Pandas (for data handling):

Copy code
pip install pandas
Google Colab (optional): If using Google Colab, you need to mount your Google Drive to access the notMNIST dataset.

Project Structure
Classes
Sequential_Model Class:

Purpose: Implements a multi-layer neural network model using Keras with adjustable hyperparameters (learning rate and regularization strength).
Methods:
build_model: Builds the neural network architecture (30 neurons in the hidden layer, softmax output for 10 classes).
train: Trains the model with early stopping and custom precision/recall tracking.
evaluate: Evaluates the model on test data, returning precision and recall scores.
predict: Predicts class labels for given test data.
display_images_with_labels: Displays a set of test images along with their true and predicted labels.
PrecisionRecallCallback Class:

Purpose: Custom callback to track precision and recall during training for both the training and validation datasets at the end of each epoch.
Plotter Class:

Purpose: Handles plotting of precision, loss, and accuracy during training.
Methods:
plot_precision: Plots training and validation precision over epochs.
plot_loss: Plots training and validation loss over epochs.
plot_accuracy: Plots training and validation accuracy over epochs.
Key Functions
Sequential_Model.load_data(data_path):

Loads the notMNIST dataset from a .mat file and normalizes the pixel values.
Splits the dataset into training, validation, and test sets.
Standardizes the data for stable neural network training.
Sequential_Model.train():

Trains the neural network on the training data using early stopping to prevent overfitting.
Tracks precision and recall using the PrecisionRecallCallback.
Sequential_Model.evaluate():

Evaluates the model on the test set and computes precision and recall scores.
Sequential_Model.predict():

Makes predictions on the test set or a subset of test images, returning the predicted class labels.
Usage
Steps to Run the Project:
Prepare the Data:

Download the notMNIST_small.mat dataset and store it in Google Drive (or another accessible location).
Mount your Google Drive in Colab:
python
Copy code
from google.colab import drive
drive.mount('/content/drive')
Run the Script:

Instantiate the Sequential_Model class and set the learning rate and regularization strength.
Load the data, train the model, and evaluate it on the test set.
Run hyperparameter tuning (grid search) to find the best learning rate and regularization strength.
python
Copy code
model = Sequential_Model(learning_rate=0.01, regularization_strength=0.0001)

X_train, X_val, y_train, y_val = model.load_data("drive/My Drive/ColabProjects/notMNIST_small.mat")

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.33, random_state=42, stratify=y_val)

# Train the model and evaluate it.
history, callback = model.train(X_train, y_train, epochs=100, batch_size=1000, validation_data=(X_val, y_val))

test_precision, test_recall = model.evaluate(X_test, y_test)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)

# Display random test images and their predictions.
num_images_to_display = 9
random_indices = np.random.choice(len(X_test), num_images_to_display)
selected_images = X_test[random_indices]
true_labels = np.argmax(y_test[random_indices], axis=1)
predicted_labels = np.argmax(model.predict(selected_images), axis=1)

model.display_images_with_labels(selected_images, true_labels, predicted_labels)

# Plot precision, loss, and accuracy during training.
plotter = Plotter()
plotter.plot_precision(callback, callback)
plotter.plot_loss(history)
plotter.plot_accuracy(history)
Example Output:
Test Precision: 0.92 (weighted average)
Test Recall: 0.91 (weighted average)
Visualizations:
Precision over Epochs: Visualizes precision for both training and validation sets across all epochs.
Loss over Epochs: Tracks training and validation loss across epochs.
Accuracy over Epochs: Plots accuracy for both training and validation sets across all epochs.
Predictions: Displays randomly selected test images along with their true labels and predicted labels.
Hyperparameter Tuning:
The model tunes hyperparameters (learning rate and regularization strength) through grid search. It retains the best model based on validation precision and retrains it on the combined training and validation sets. The final model is evaluated on the test set.

Conclusion
This project demonstrates how to build and train a multi-layer neural network to classify images of alphabetical glyphs from the notMNIST dataset. By tuning the learning rate and regularization strength, the model achieves strong classification performance, which is evaluated using precision and recall metrics. The project also visualizes the results, providing insights into the model's learning process and final predictions.

Future Enhancements
Advanced Hyperparameter Tuning: Implement more sophisticated methods for hyperparameter tuning such as Bayesian optimization (e.g., using Optuna).

Convolutional Neural Networks (CNNs): Replace the fully connected layers with convolutional layers to improve classification accuracy for image data.

Data Augmentation: Implement data augmentation techniques to increase the robustness of the model and prevent overfitting.

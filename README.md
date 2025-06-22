
🧠 Image Classification using Sequential CNN Model
This repository contains a Convolutional Neural Network (CNN) built using TensorFlow/Keras to classify images of cats and dogs. The model is implemented using a Sequential architecture for simplicity and is designed for beginner to intermediate deep learning practitioners.

📁 Project Structure
bash
Copy
Edit
├── Img_classify_Sequential.ipynb
├── /content/images/train/         # Contains original training images
├── /content/images/train_subset/  # Subset for training/validation/testing
├── README.md
🚀 Model Architecture
The CNN model is built using:

Convolutional layers

MaxPooling layers

Dropout for regularization

Dense layers for classification

It classifies images into two categories: Cats and Dogs.

🛠️ Features
Data preparation and directory structure creation

Image preprocessing using Keras ImageDataGenerator

Model training and evaluation

Accuracy/loss visualization

Prediction on test data

📊 Requirements
Python 3.8+

TensorFlow 2.x

NumPy

Matplotlib

You can install the dependencies with:

bash
Copy
Edit
pip install tensorflow numpy matplotlib
🧪 Training
The notebook:

Loads a subset of images (train/validation/test)

Uses a batch size of 32

Trains the model using model.fit()

Evaluates performance on a validation and test set

📈 Results
Training and validation accuracy/loss curves are plotted at the end to monitor overfitting or underfitting. The model can be saved and reused for predictions.

📷 Sample Prediction
The model takes a test image, resizes it, and predicts whether it's a cat or a dog, showing both the image and prediction.

✅ Usage
Run the notebook from start to finish:

python
Copy
Edit
# Inside the notebook
model.fit(...)
model.evaluate(...)
model.predict(...)
You can modify make_subset() if you want to customize how many images are used in training/validation/test.

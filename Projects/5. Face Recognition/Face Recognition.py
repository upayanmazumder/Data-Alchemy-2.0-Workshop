# %% [markdown]
# ## Step 1: Install OpenCV
# In this step, we install the required libraries for OpenCV to work on Google Colab.

# %%
# Install OpenCV to work with images
# !pip install opencv-python
# !pip install opencv-python-headless  # Headless version of OpenCV (no GUI, necessary for Colab)
# !pip install opencv-contrib-python

# %% [markdown]
# ## Step 2: Unzip the Dataset
# Once the dataset is uploaded, we will unzip it into a directory for further use.

# %%
# Step 3: Unzip the Dataset
import zipfile
import os

zip_file_path = 'dataset.zip'  # Replace with the actual zip file name

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall('dataset')  # Extracts to a folder named 'dataset'


# %% [markdown]
# ## Step 3: Import Required Libraries
# We import the necessary libraries to work with OpenCV and handle arrays.

# %%
# Import OpenCV for image processing
import cv2

# Import NumPy to handle arrays and matrices
import numpy as np


# %% [markdown]
# ## Step 4: Load and Preprocess Images
# We load images from the unzipped dataset folder and associate them with labels based on folder names.

# %%
# Function to load images from a folder and map them to labels
def load_images_from_folder(folder):
    images = []  # List to store the images
    labels = []  # List to store corresponding labels
    label_map = {}  # Dictionary to map label numbers to person names

    # Loop through each subfolder (each person's folder) in the dataset
    for label, person_name in enumerate(os.listdir(folder)):
        person_folder = os.path.join(folder, person_name)  # Path to the person's folder
        label_map[label] = person_name  # Map label number to person's name

        # Loop through each image file in the person's folder
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)  # Full path to the image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale

            # If image is successfully loaded, add it to the list
            if img is not None:
                images.append(img)  # Add the image to the images list
                labels.append(label)  # Add the corresponding label

    return images, labels, label_map  # Return the images, labels, and label map

# Load the training images and their corresponding labels from the 'train_data' folder
train_images, train_labels, label_map = load_images_from_folder('dataset/dataset/train_data')


# %% [markdown]
# ## Step 5: Train the Face Recognition Model
# We will use the LBPH (Local Binary Patterns Histograms) face recognizer to train the model.

# %%
model = cv2.face_LBPHFaceRecognizer.create()

# Train the model on the training images and labels
model.train(train_images, np.array(train_labels))


# %% [markdown]
# ## Step 6: Testing the Model
# Finally, we test the trained model on images from the test dataset.

# %%
# Function to test the face recognition model with images from a test folder
def test_model(model, test_folder):
    # Loop through each image in the test folder
    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)  # Full path to the test image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale

        # If image is successfully loaded, predict the label
        if img is not None:
            label, confidence = model.predict(img)  # Predict the person's label and get confidence score
            print(f"Image: {img_name} is recognized as: {label_map[label]} with confidence: {confidence}")

# Test the model with images from the 'test_data' folder
test_model(model, 'dataset/dataset/test_data')




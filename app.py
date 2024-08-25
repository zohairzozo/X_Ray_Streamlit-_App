import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

# Set the directory paths for normal and pneumonia training images
TRAIN_NORMAL = "chest_xray/train/NORMAL"
TRAIN_PNEUMONIA = "chest_xray/train/PNEUMONIA"

# Load your trained model
model_path = 'model.h5'
model = tf.keras.models.load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((128, 128))  # Resize image to match model's expected sizing
    
    # Check if image has 3 channels (RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')  # Convert to RGB if not already
    
    img = np.asarray(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to check if an image is a chest X-ray
def is_chest_xray(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((256, 256))  # Resize image for consistency
        
        # Convert to RGB if not already in RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.asarray(img)
        
        # Check if image is grayscale
        if len(img_array.shape) < 3:
            return False
        
        # Check if image is mostly black and white
        grayscale_img = rgb2gray(img_array)
        thresh = threshold_otsu(grayscale_img)
        binary_img = grayscale_img > thresh
        return np.mean(binary_img) > 0.5  # Adjust threshold based on your images
    
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False

# Function to display a random normal and pneumonia image
def display_comparison_images(user_image_path):
    # Open user uploaded image
    user_image = Image.open(user_image_path)
    
    # Choose a random image from the normal and pneumonia directories
    random_normal_image = random.choice(os.listdir(TRAIN_NORMAL))
    random_pneumonia_image = random.choice(os.listdir(TRAIN_PNEUMONIA))

    # Open the selected images
    normal_image = Image.open(os.path.join(TRAIN_NORMAL, random_normal_image))
    pneumonia_image = Image.open(os.path.join(TRAIN_PNEUMONIA, random_pneumonia_image))

    # Create a figure for displaying the images
    figure = plt.figure(figsize=(20, 10))

    # Display the user uploaded image in the first subplot
    subplot1 = figure.add_subplot(1, 3, 1)
    plt.imshow(user_image)
    subplot1.set_title("Uploaded Image")

    # Display the normal image in the second subplot
    subplot2 = figure.add_subplot(1, 3, 2)
    plt.imshow(normal_image)
    subplot2.set_title("Normal")

    # Display the pneumonia image in the third subplot
    subplot3 = figure.add_subplot(1, 3, 3)
    plt.imshow(pneumonia_image)
    subplot3.set_title("Pneumonia")

    # Show the figure
    st.pyplot(figure)

# Function to predict pneumonia based on user-uploaded image
def predict_pneumonia(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Streamlit app
def main():
    st.title('Pneumonia Detection App')
    st.write('Upload a chest X-ray image for pneumonia detection')

    # File upload
    uploaded_file = st.file_uploader("Choose a chest X-ray image ...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Temporarily save the uploaded image
        user_image_path = './user_uploaded_image.png'
        with open(user_image_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Check if uploaded image is a chest X-ray
        if not is_chest_xray(user_image_path):
            st.error('Please upload a valid chest X-ray image only.')
            return

        # Display the uploaded image
        user_image = Image.open(user_image_path)
        st.image(user_image, caption='Uploaded Image', use_column_width=True)

        # Display normal and pneumonia images for comparison
        display_comparison_images(user_image_path)

        # Predict pneumonia based on the uploaded image
        prediction = predict_pneumonia(user_image_path)
        
        # Determine and display prediction result
        pneumonia_probability = prediction[0][0]
        if pneumonia_probability > 0.5:
            st.error('High probability of Pneumonia. Please consult a doctor for further evaluation.')
        else:
            st.success('Low probability of Pneumonia. Consider regular check-ups.')

        # Remove the temporarily saved uploaded image
        os.remove(user_image_path)

# Run the app
if __name__ == '__main__':
    main()
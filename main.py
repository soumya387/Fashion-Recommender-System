import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load precomputed feature list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Initialize the ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

# Create upload directory if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def recommend(features, feature_list):
    try:
        neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return None

# File upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        # Display the uploaded image
        display_image = Image.open(file_path)
        st.image(display_image, caption='Uploaded Image', use_column_width=True)

        # Feature extraction
        features = feature_extraction(file_path, model)
        if features is not None:
            # Recommendation
            indices = recommend(features, feature_list)
            if indices is not None:
                # Display recommended images
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    img_path = filenames[indices[0][i]]
                    col.image(img_path, use_column_width=True)
            else:
                st.header("Could not generate recommendations.")
        else:
            st.header("Could not extract features from the image.")
    else:
        st.header("Some error occurred in file upload.")

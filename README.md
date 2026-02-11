# 👗 Fashion Recommendation System

A deep learning–based fashion recommender system that suggests visually similar clothing items based on an uploaded image. The system uses a pre-trained ResNet50 model for feature extraction and K-Nearest Neighbors (KNN) for similarity matching. The application is deployed using Streamlit.

---

## 🚀 Features

- Image-based fashion recommendations
- Deep learning feature extraction using ResNet50
- Similarity search using KNN
- Interactive web interface with Streamlit

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- ResNet50  
- NumPy  
- Scikit-learn  
- Streamlit  
- Pillow (PIL)  
- Pickle  

---

## 📁 Project Structure




> ⚠️ Dataset images and generated `.pkl` files are not included due to GitHub size limitations.

---

## 📦 Dataset

Dataset used: **Fashion Product Images Dataset (Kaggle)**

Create a folder named `images/` and place all dataset images inside it.

---

## 🧠 Generating Embeddings (One-Time Setup)

Before running the application, you must generate:

- `embeddings.pkl`
- `filenames.pkl`

These files contain extracted image feature vectors and corresponding image paths.

To generate them:
1. Place dataset images inside `images/`
2. Run your feature extraction code (used to create embeddings locally)
3. Ensure `embeddings.pkl` and `filenames.pkl` are in the project root directory

---

## ⚙️ Installation

### 1️⃣ Clone Repository




### 2️⃣ Install Dependencies


---

## ▶️ Run the Application


Open the URL shown in the terminal (usually http://localhost:8501).

---

## 🖼️ How It Works

1. User uploads an image.
2. The system extracts features using ResNet50.
3. The feature vector is compared with stored embeddings.
4. KNN finds the most visually similar fashion items.
5. Recommended images are displayed.

---

## ❌ Excluded Files

The following files are excluded from GitHub:

- `images/`
- `embeddings.pkl`
- `filenames.pkl`
- `uploads/`
- Virtual environment folders

---

## ✅ Conclusion

This project demonstrates the application of computer vision and deep learning in building a real-world content-based fashion recommendation system.

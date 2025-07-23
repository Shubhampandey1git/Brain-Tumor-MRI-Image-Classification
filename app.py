import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Loading the chosen custom CNN model
model = tf.keras.models.load_model('brain_tumor_cnn_model.h5')

# Defining class names in order
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# Image preprocessing
def preprocess_image(image, target_size=(180,180)):
    image = image.resize(target_size)
    image = image.convert('RGB')
    image_arr = np.array(image) / 255.0   # Rescaling
    image_arr = np.expand_dims(image, axis=0)  # Adding batch dim
    return image_arr

# Streamlit App UI
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")
st.title("Brain Tumor MRI Classification")
st.write("Upload an MRI image to predict the type of tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    st.write("Classifying...")
    processed_img = preprocess_image(image)

    predictions = model.predict(processed_img)
    confidence_scores = tf.nn.softmax(predictions[0]).numpy()
    predicted_class = class_names[np.argmax(confidence_scores)]

    st.success(f"### Predicted Tumor Type: **{predicted_class.upper()}**")
    st.write("#### Confidence Scores:")
    
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {confidence_scores[i]*100:.2f}%")
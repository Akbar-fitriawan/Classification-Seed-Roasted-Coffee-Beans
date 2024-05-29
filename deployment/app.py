import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np


        


# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model_genaral.h5')
    return model

model = load_model()

# Define the class names (replace with your actual class names)
class_names = ['Dark','Green','Light','Medium']

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust this according to your model input size
    image = np.array(image)
    if image.shape[2] == 4:  # Check if the image has an alpha channel
        image = image[:, :, :3]
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)
    return image

# Main function to run the Streamlit app
def main():

    st.title("Welcome to my project")
    st.markdown('**Introduction**')
    st.write("Name\t: Akbar Fitriawan")
    st.write("Batch\t: hacktiv8-15")

    with st.expander('Tujuan Milestone'):
        st.write('Membuat model klasifikasi gambar Coffee Beans Roast')
    st.divider()

    st.header("Image Classification App")
    st.subheader('Coffee Beans Roast')
    st.write('Selamat datang di aplikasi klasifikasi biji kopi berbasis computer vision . Aplikasi ini membantu Anda mengklasifikasikan gambar biji kopi berdasarkan tingkat kematangan, termasuk dark roasted, medium roasted, light roasted, dan raw (green). Silakan tekan tombol Browse files untuk mengunggah gambar dan melakukan prediksi')

    st.divider()
    st.write("Upload an image to classify")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make predictions
        predictions = model.predict(processed_image)
        score = tf.nn.softmax(predictions[0])
        
        # Display the results
        st.write(f"Prediction: {class_names[np.argmax(score)]}")
        st.write(f"Confidence: {100 * np.max(score):.2f}%")


if __name__ == "__main__":
    main()
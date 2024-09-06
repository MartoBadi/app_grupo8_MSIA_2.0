import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('fashion_CNN_new.h5')

# Define class labels for Fashion MNIST dataset
class_labels = [
    'remera', 'pantalón', 'cartera', 'vestido', 'sobretodo', 'sandalia', 'camisa', 'zapatilla', 'pulóver', 'bota'
]

# Function to preprocess uploaded image
def preprocess_image(image):
    image = image.resize((28, 28))
    image = image.convert('L')
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 28, 28, 1)

# Main function for Streamlit app
def main():
    st.text("Esta aplicación web ha sido creada por Martín Badino para la materia Modelizado de Sistemas de IA de la carrera de Ciencias de Datos del IFTS 18. Esta aplicación web tiene integrado un modelo de IA que usa una red neuronal artificial para clasificar imágenes de prendas de ropa en las siguientes categorías: remera, pantalón, cartera, vestido, sobretodo, sandalia', camisa, zapatilla, pulóver y bota.\n\n La idea es que subas una imagen de una prenda de ropa que pertenezca a las cateogorías mencionadas y la aplicación te dirá a qué categoría pertenece (scrollear hacia abajo hasta donde dice: drag and drop file here).")
    

    uploaded_file = st.file_uploader("Adjunta una imagen jpg, jpeg o png.", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image with reduced size
        st.image(uploaded_file, caption='Adjuntá una imagen...', use_column_width=False, width=200)
        
        # Preprocess and predict
        image = Image.open(uploaded_file)
        processed_image = preprocess_image(image)
        
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction[0])
            
        if class_labels[predicted_class_index] in ["remera", "cartera", "sandalia", "camisa",  "zapatilla",  "bota"]: # Si la prenda en un sustantivo femenino se usa una: 
                st.write(f"En la imagen hay una {class_labels[predicted_class_index]}")
        else:
            st.write(f"En la imagen hay un {class_labels[predicted_class_index]}")
        
if __name__ == "__main__":
    main()
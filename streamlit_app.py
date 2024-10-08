import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('fashion_CNN_new.h5')

# Define class labels for Fashion MNIST dataset
class_labels = [
    'remera', 'pantalón', 'pulóver', 'vestido', 'sobretodo', 'sandalia', 'camisa', 'zapatilla', 'cartera', 'bota'
]

# Function to preprocess uploaded image
def preprocess_image(image):
    image = image.resize((28, 28))
    image = image.convert('L')
    image_array = np.array(image) / 255.0
    return image_array.reshape(1, 28, 28, 1)

# Main function for Streamlit app
def main():
    st.markdown("""
    Esta aplicación web ha sido creada por Martín Badino para la materia Modelizado de Sistemas de IA de la carrera de Ciencias de Datos del IFTS 18. Esta aplicación web tiene integrado un modelo de IA que usa una red neuronal artificial para clasificar imágenes de prendas de ropa en las siguientes categorías: remera, pantalón, cartera, vestido, sobretodo, sandalia, camisa, zapatilla, pulóver y bota.

    La idea es que subas una imagen jpg, jpeg o png de una prenda de ropa que pertenezca a las categorías mencionadas y la aplicación te dirá a qué categoría pertenece (si estás viendo esta página desde un celular scrolleá hacia abajo hasta donde dice: browse files). Puedes sacarle una foto a tu prenda o buscar una imagen en Pinterest (ahí podés encontrar fácilmente fotos de prendas de ropa solas con un fondo blanco, o sea sin que en las fotos haya un modelo con la ropa puesta).
    """)
    
    uploaded_file = st.file_uploader("Adjuntá una imagen jpg, jpeg o png.", type=["jpg", "jpeg", "png"])
    
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
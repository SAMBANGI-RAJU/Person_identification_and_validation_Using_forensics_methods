import streamlit as st
import numpy as np
from numpy import asarray
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import tensorflow as tf
# import numpy as np
from tensorflow.keras.preprocessing import image
model = load_model("Person_identification_iris.h5")

image_size = (256, 256)
batch_size = 32


def predict(image_path):
    img = image.load_img(image_path, target_size=(224, 224, 3))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    class_labels = {'1': 0, '10': 1, '11': 2, '12': 3, '13': 4, '14': 5, '15': 6, '16': 7, '17': 8, '18': 9, '19': 10, '2': 11, '20': 12, '21': 13, '22': 14, '23': 15,
                    '24': 16, '25': 17, '26': 18, '27': 19, '28': 20, '29': 21, '3': 22, '30': 23, '31': 24, '32': 25, '33': 26,
                    '34': 27, '35': 28, '36': 29, '37': 30, '38': 31, '39': 32, '40': 33, '41': 34, '42': 35, '43': 36, '44': 37, '45': 38,
                    '46': 39, '5': 40, '6': 41, '7': 42, '8': 43, '9': 44}
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
   # class_labels = train_generator.class_indices
    predicted_class_label = list(class_labels.keys())[predicted_class_index]

    return predicted_class_label


st.title('Person Identification and validation using Iris image Recognition')


imge = st.file_uploader('Upload your file', type=['JPG', 'PNG', 'JPEG', 'TIFF', 'bmp'], accept_multiple_files=False, key=None, help=None,
                        on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")


if (imge != None):
    st.image(imge, caption='Uploaded Image')

if st.button('Predict'):

    predict = predict(imge)
    st.markdown(""" <style> .predict {
font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
</style> """, unsafe_allow_html=True)
    st.write("The predicted class label is:", predict)

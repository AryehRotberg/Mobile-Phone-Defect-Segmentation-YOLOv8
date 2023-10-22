import streamlit as st

from PIL import Image
from ultralytics import YOLO


model = YOLO('runs/segment/train/weights/best.pt')

st.title('Mobile Phone Defect Segmentation')
file = st.file_uploader('Upload image')

predict_btn = st.button('Predict')

if predict_btn and file:
    image = Image.open(file)
    results = model.predict(image, save=False, verbose=False)

    st.image(results[0].plot())

    num_scratches = results[0].boxes.cls.tolist().count(0)

    st.write(f'Detected {num_scratches} scratches.')

import streamlit as st

from PIL import Image
from ultralytics import YOLO


model = YOLO('runs/segment/train/weights/best.pt')

st.title('Mobile Phone Defect Segmentation')
file = st.file_uploader('Upload image')

predict_btn = st.button('Detect')

if predict_btn and file:
    image = Image.open(file)
    results = model.predict(image, save=False, verbose=False)

    num_scratches = results[0].boxes.cls.tolist().count(0)

    if num_scratches == 0:
        st.info("No scratches detected! You're good to go.")
    
    else:
        st.error(f'⚠️ Detected {num_scratches} scratches.')

    st.image(results[0].plot())

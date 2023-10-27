from io import BytesIO

from fastapi import FastAPI, UploadFile, File

import cv2
from PIL import Image

from ultralytics import YOLO
from starlette.responses import StreamingResponse


model = YOLO('runs/segment/train/weights/best.pt')

app = FastAPI(title='Mobile Phone Defect Segmentation Using YOLOv8')

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    input_image = Image.open(BytesIO(await file.read()))
    results = model.predict(input_image, save=False, verbose=False)
    
    num_scratches = results[0].boxes.cls.tolist().count(0)
    result_image = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)

    bytes_io = BytesIO()
    result_image = Image.fromarray(result_image)
    result_image.save(bytes_io, format='PNG')
    bytes_io.seek(0)

    return StreamingResponse(bytes_io, media_type='image/png', headers={'num_scratches': f'{num_scratches}'})

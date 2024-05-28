import math
import cv2

from ultralytics import YOLO


model = YOLO('models/production/best.pt')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, image = cap.read()
    results = model(image, save=False, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            # class_name = box.cls.to('cpu').item()

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            # cv2.putText(image, class_name, [x1 - 3, y1 - 10], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    cv2.imshow('Webcam', image)
    if cv2.waitKey(1) == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()

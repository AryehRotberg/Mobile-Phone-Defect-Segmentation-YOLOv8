import math
import cv2

from ultralytics import YOLO


class CameraUtils:
    def __init__(self,
                 camera_id: int,
                 model_path: str,
                 pred_dict: dict,
                 pred_display_color: tuple,
                 display_size: list) -> None:
        
        self.camera_id = camera_id
        self.pred_dict = pred_dict
        self.pred_display_color = pred_display_color
        self.display_size = display_size

        self.model = YOLO(model_path)

        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(3, display_size[0])
        self.cap.set(4, display_size[1])
    
    def start_capture(self):
        while True:
            success, image = self.cap.read()

            if success:
                results = self.model(image, save=False, verbose=False)

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        confidence = math.ceil((box.conf[0] * 100)) / 100
                        pred_class_name = self.pred_dict[box.cls.to('cpu').item()]
                        
                        cv2.rectangle(image, (x1, y1), (x2, y2), self.pred_display_color, 2)
                        cv2.putText(image, f'{pred_class_name}, {confidence}', [x1 - 3, y1 - 10], cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.pred_display_color, 2)

                cv2.imshow(f'Webcam_{self.camera_id}', image)
            
            else:
                break

            if cv2.waitKey(1) == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

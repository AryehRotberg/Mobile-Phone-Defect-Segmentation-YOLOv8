from ultralytics import YOLO


class ModelEvaluation:
    def __init__(self):
        self.model = YOLO('runs/segment/train/weights/best.pt')
    
    def get_parameters(self):
        return self.model.val()

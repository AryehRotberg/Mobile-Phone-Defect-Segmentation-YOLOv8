import os

import yaml
from ultralytics import YOLO


class ModelTrainer:
    def __init__(self):
        self.model = YOLO('models/yolov8s-seg.pt')

        with open('config.yaml') as file:
            self.config = yaml.safe_load(file)
            self.config = self.config['model_training']
    
    def train(self):
        self.model.train(data=os.path.join(self.config['data_directory'], 'data.yaml'),
                         epochs=100,
                         imgsz=640)

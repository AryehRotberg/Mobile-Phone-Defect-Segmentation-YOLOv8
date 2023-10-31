import yaml

from ultralytics import YOLO


class ModelEvaluation:
    def __init__(self):
        self.model = YOLO('runs/segment/train/weights/best.pt')

        with open('config.yaml') as file:
            self.config = yaml.safe_load(file)
            self.config = self.config['model_training']
    
    def get_parameters(self):
        self.model.val(data=self.config['data_directory'],
                       split='val')

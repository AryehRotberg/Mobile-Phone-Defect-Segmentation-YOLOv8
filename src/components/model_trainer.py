import yaml
from ultralytics import YOLO

import optuna

from src.components.utils import common

class ModelTrainer:
    def __init__(self):
        self.model = YOLO('models/yolov8s-seg.pt')

        with open('config.yaml') as file:
            self.config = yaml.safe_load(file)
            self.config = self.config['model_training']
            self.data_config_file = self.config['data_config_file']
    
    def objective(self, trial: optuna.trial.Trial):
        optimizer, patience = common().get_hyperparameters(trial)

        self.train(optimizer, patience)

        self.model.val(data=self.data_config_file,
                       split='val',
                       name=f'optimizer_{optimizer}_patience_{patience}_val')
        
        return self.model.metrics.fitness

    def train(self, optimizer, patience):
        self.model.train(data=self.data_config_file,
                         epochs=100,
                         imgsz=640,
                         optimizer=optimizer,
                         patience=patience,
                         name=f'optimizer_{optimizer}_patience_{patience}')
    
    def create_optuna_pipeline(self, n_trials):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

        return study
    
    def get_best_study_results(self, study):
        return {'best_params': study.best_params,
                'best_value': study.best_value}

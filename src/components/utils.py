import os
import yaml

import optuna


class common:
    def __init__(self):        
        with open('config.yaml') as file:
            self.config = yaml.safe_load(file)
    
    def update_model_data_directory(self):
        self.config['model_training']['data_config_file'] = os.path.join(os.getcwd(), 'data/data.yaml').replace('\\', '/')

        with open('config.yaml', 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)
    
    def get_hyperparameters(self, trial: optuna.trial.Trial):
        optimizer = trial.suggest_categorical('optimizer', ['auto', 'NAdam', 'AdamW', 'RMSProp'])
        patience = trial.suggest_categorical('patience', [10, 25, 50])

        return optimizer, patience

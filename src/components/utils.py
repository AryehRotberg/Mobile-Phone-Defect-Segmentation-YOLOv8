import os
import yaml


class common:
    def __init__(self):        
        with open('config.yaml') as file:
            self.config = yaml.safe_load(file)
    
    def update_model_data_directory(self):
        self.config['model_training']['data_directory'] = os.path.join(os.getcwd(), 'data').replace('\\', '/')

        with open('config.yaml', 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)

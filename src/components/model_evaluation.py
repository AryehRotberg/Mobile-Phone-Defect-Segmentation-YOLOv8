import yaml

class ModelEvaluation:
    def __init__(self, model):
        self.model = model

        with open('config.yaml') as file:
            self.config = yaml.safe_load(file)
            self.config = self.config['model_training']
            self.data_config_file = self.config['data_config_file']
    
    def get_fitness_score(self, split, file_name):
        self.model.val(data=self.data_config_file,
                       split=split,
                       name=file_name)
        
        return self.model.metrics.fitness

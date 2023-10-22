import logging

from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer


# Logging Configuration
logging.basicConfig(filename='logs/pipeline.log',
                    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# Data Ingestion
data_ingestion = DataIngestion()
data_ingestion.download_roboflow_project()
data_ingestion.extract_zip_file('Mobile Phone Defect Segmentation.v1i.yolov8.zip')

logging.info('Downloaded roboflow data and updated data directory in config.')

# Model Training
model_trainer = ModelTrainer()
results = model_trainer.train()

logging.info('Completed training model.')

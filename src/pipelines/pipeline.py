import logging

from src.components.data_ingestion import DataIngestion


# Logging Configuration
logging.basicConfig(filename='logs/pipeline.log',
                    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

data_ingestion = DataIngestion('data')
data_ingestion.download_roboflow_project()
data_ingestion.extract_zip_file('Mobile Phone Defect Segmentation.v1i.yolov8.zip')

import logging

from src.components.data_ingestion import DataIngestion
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


if __name__ == '__main__':
    # Logging Configuration
    logging.basicConfig(filename='logs/pipeline.log',
                        format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    # Data Ingestion
    # data_ingestion = DataIngestion()
    # data_ingestion.download_roboflow_project()
    # data_ingestion.extract_zip_file('Mobile Phone Defect Segmentation.v1i.yolov8.zip')

    # logging.info('Downloaded roboflow data and updated data directory in config.\n')

    # Model Training
    model_trainer = ModelTrainer()
    study = model_trainer.create_optuna_pipeline(n_trials=7)

    logging.info(f'Tuned YOLO model. Best Results -> {model_trainer.get_best_study_results(study)}')

    # model_evaluation = ModelEvaluation()
    # logging.info(f'Model Parameters -> \n{model_evaluation.get_parameters()}')

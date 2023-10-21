import yaml
import wget

import os
from zipfile import ZipFile


class DataIngestion:
    def __init__(self):
        self.data_directory = 'data'

        with open('config.yaml') as file:
            self.config = yaml.safe_load(file)
            self.config = self.config['data_ingestion']
    
    def download_roboflow_project(self):
        wget.download(self.config['data_download_url'], out=self.data_directory)
    
    def extract_zip_file(self, zip_file):
        with ZipFile(os.path.join(self.data_directory, zip_file), 'r') as file:
            file.extractall(self.data_directory)
        
        os.remove(os.path.join(self.data_directory, zip_file))

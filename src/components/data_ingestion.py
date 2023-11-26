import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        try:
            df = pd.read_csv('data\stud.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=None)
            logging.info(f'Raw data stored to {self.ingestion_config.raw_data_path}')

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=101)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=None)
            logging.info(f'Train data stored to {self.ingestion_config.train_data_path}')
            
            test_set.to_csv(self.ingestion_config.test_data_path, index=None)
            logging.info(f'Test data stored to {self.ingestion_config.test_data_path}')
            
            logging.info('Data ingestion completed')
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    x = DataIngestion()
    x.initiate_data_ingestion()
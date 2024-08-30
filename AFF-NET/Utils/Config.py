# Change here hyperparameters and directoties for training and testing the model

import os
from typing import List

# CONSTANT DIRECTORIES
DATA_PATH : str = os.path.abspath('D:\\Eye_tracking_dataset\\EVE')
TEST_NAMES: List[str] = ['train02', 'train10', 'train21', 'train27', 'train28',
                         'train32', 'train34', 'train35', 'train36', 'val03']
DIRECTORY : str = os.path.abspath('Z:\\PORTIONS_4\\EYE_TRACKING_ORD\\EVE')
JSONS_PATH : str = os.path.join(DIRECTORY, 'JSON_FILES', 'ALL')
MODELS_PATH : str = os.path.join(DIRECTORY, 'MODELS', 'distribucion_train_test_02')
RESULTS_PATH : str = os.path.join(DIRECTORY, 'RESULTS', 'distribucion_train_test_02')

# HYPERPARAMETERS - TRAINING 
EPOCH : int = 30
BATCH_SIZE : int = 8
# WEIGHTS_PATH : str = '' # set False if we want to train from scratch
PATH_TO_SAVE_WEIGHTS : str = '' # path to save network's weights

# HYPERPARAMETERS - TESTING 
PATH_TO_SAVE_RESULTS : str = '' # path to save excel results of the prediction


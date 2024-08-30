import os
from pathlib import Path
from typing import List

TEST_NAMES : List[str] = ['train02', 'train10', 'train21', 'train27', 'train28', 'train32', 'train34', 'train35', 'train36', 'val03']
DATA_PATH : Path = Path('D:\\Eye_tracking_dataset\\EVE')

MAIN_PATH : Path = Path('Z:\\PORTIONS_4\\EYE_TRACKING_ORD\\ORGANIZADO\\EVE')

# json files
JSON_PATH : Path = MAIN_PATH.joinpath('JSON_FILES')
TRAIN_JSON : Path = JSON_PATH.joinpath('train_data.json')
VAL_DATA : Path = JSON_PATH.joinpath('val_data.json')

# experiments - models
MODELS_PATH : Path = MAIN_PATH.joinpath('EXPERIMENTS')

# results
RESULTS_PATH : Path = MAIN_PATH.joinpath('RESULTS')

# PARAMETERS
LR : float = 0.0001
EPOCH : int = 20
BATCH_SIZE : int = 8

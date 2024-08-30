#######################################################
########### SCRIPT TO TRAIN ALL THE EXPERIMENTS #######
#######################################################
if __name__ == '__main__':
    import torch.multiprocessing as mp
    from pathlib import Path
    from typing import List
    import sys

    sys.path.append('.\\Utils')
    import EVE_config as c
    from Classes import Trainer
    from utils import createDirectory


    mp.set_start_method('spawn', force=True)
    ############ PARAMETERS ###############################
    data_path : Path = Path('D:\\ET_DATABASES\\EVE')
    PROJECT_DIRECTORY : Path = Path('D:\\TFM_ET\\CODE')


    lr = 0.0001 
    metric = 'MSE'
    exp_name = f'{metric}_LR_{lr}_LEFT'
    # createDirectory(PROJECT_DIRECTORY.joinpath('MODELS', 'BASE_MODEL', exp_name))
    train_json_path : Path = Path('D:\\TFM_ET\\DATA_AND_RESULTS\\JSON_FILES\\train_data.json')
    val_json_path : Path = Path('D:\\TFM_ET\\DATA_AND_RESULTS\\JSON_FILES\\val_data.json')

    # path_to_save : Path = PROJECT_DIRECTORY.joinpath('MODELS', 'BASE_MODEL', exp_name)
    # print(path_to_save)
    trainer = Trainer(data_path = data_path,
                                metric = metric,
                                train_json_path = train_json_path,
                                val_json_path = val_json_path, 
                                weights_path = False,
                                path_to_save = Path('models'),
                                layers_to_train = 'all',
                                batchsize = 8,
                                epoch_n = 100,
                                lr = lr)
    trainer.train()


#######################################################
########### SCRIPT TO TRAIN ALL THE EXPERIMENTS #######
#######################################################
if __name__ == '__main__':
    import torch.multiprocessing as mp
    from pathlib import Path
    from typing import List
    import sys

    sys.path.append('.\\Utils')
    from Classes import Trainer
    from utils import createDirectory


    mp.set_start_method('spawn', force=True)
    ############ PARAMETERS ###############################
    data_path : Path = Path('D:\\ET_DATABASES\\EVE')
    EVE_TEST_USERS : List[str] = ['train'+'%02d' % a for a in [2,10,21,27,28,32,34,35,36]] + ['val03']
    EVE_DIRECTORY : Path = Path('D:\\TFM_ET\\DATA_AND_RESULTS')

    for n in Path('D:\\TFM_ET\\DATA_AND_RESULTS\\JSON_FILES\\calibration_analysis').glob('*samples'):
        print(n.name)
        weights_path : Path = EVE_DIRECTORY.joinpath('MODELS', 'BASE_MODEL', f'BM_WMSE_LR_0.0001', 'model_60.pt')
        lr = 0.00001
        exp_name = f'AC_{n.name}'
        createDirectory(EVE_DIRECTORY.joinpath('MODELS', 'ANALYSIS_CALIBRATION', exp_name))
            

        for user in EVE_TEST_USERS:
                createDirectory(EVE_DIRECTORY.joinpath('MODELS', 'ANALYSIS_CALIBRATION', exp_name,user))
                train_json_path : Path = EVE_DIRECTORY.joinpath('JSON_FILES', 'calibration_analysis', n.name, user, 'calibration.json')
                val_json_path : Path = EVE_DIRECTORY.joinpath('JSON_FILES', 'CALIBRATION_TEST', user, 'test.json')
                print(train_json_path)
                path_to_save : Path = EVE_DIRECTORY.joinpath('MODELS', 'ANALYSIS_CALIBRATION', exp_name,user)

                trainer = Trainer(data_path = data_path,
                                metric = 'WMSE',
                                train_json_path = train_json_path,
                                val_json_path = val_json_path, 
                                weights_path = weights_path,
                                path_to_save = path_to_save,
                                layers_to_train = 'all',
                                batchsize = 8,
                                epoch_n = 40,
                                lr = lr)
                trainer.train()


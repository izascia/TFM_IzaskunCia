#######################################################
########### SCRIPT TO TEST ALL THE EXPERIMENTS ####### EVE !!
#######################################################
if __name__ == '__main__':
    import torch.multiprocessing as mp
    from pathlib import Path
    from typing import List
    import sys
    import pandas as pd

    sys.path.append('.\\Utils')
    from Classes import Tester
    from utils import createDirectory


    mp.set_start_method('spawn', force=True)
    ############ PARAMETERS ###############################
    data_path : Path = Path('D:\\ET_DATABASES\\EVE')
    EVE_TEST_USERS : List[str] = ['train'+'%02d' % a for a in [2,10,21,27,28,32,34,35,36]] + ['val03']
    PROJECT_DIRECTORY : Path = Path('D:\\TFM_ET')
    epoch = 20

    for metric in ['WMSE']:
        for lr in [0.0001]: 
            final_data = pd.DataFrame()
            exp_name = f'onlycal_{metric}_lr_{lr}'
                        

            for user in EVE_TEST_USERS:
                weights_path : Path = PROJECT_DIRECTORY.joinpath('DATA_AND_RESULTS','MODELS', 'CALIBRATION_MODEL', 'without_base_model', exp_name, user)
                test_json_path : Path = PROJECT_DIRECTORY.joinpath('DATA_AND_RESULTS','JSON_FILES', 'CALIBRATION_TEST', user, 'test.json')


                tester = Tester(data_path = data_path, dataset_name = 'EVE', test_json_path = test_json_path,
                        weights_path = weights_path.joinpath(f'model_{epoch}.pt'))
                print('Computing metrics...')
                results = tester.test()
                # print('Saving excel!')
                # results.to_excel(path_to_save)  
                final_data = pd.concat([final_data, results]) 
            path_to_save : Path = PROJECT_DIRECTORY.joinpath('DATA_AND_RESULTS','RESULTS','calibration_not_base_model', f'{exp_name}.xlsx')
            
            final_data.to_excel(path_to_save)
import torch.nn as nn
import torch
import sys
sys.path.append('.\\Network')
from Loader import load_data
from Model import Model
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import pandas as pd
sys.path.append('.\\Utils')
from utils import dist_angular, InfoEVE, InfoUPNA_multigaze
sys.path.append('.\\Utils')
from Losses import PonderatedLoss

class Trainer():
    def __init__(self, data_path : Path, train_json_path : Path, val_json_path : Path,  weights_path : str, metric : str, path_to_save : str , batchsize : int = 8, epoch_n : int = 60, lr : float = 0.0001,layers_to_train : str = 'all'):
        """
            data_path : path to images
            train_json_path : path to json file with train info
            val_json_path : path to json file with validation info
            weights_path : If none -> False, else the path to the weights to initialize the network
            metric : 'MSE' or 'WMSE' 
            path_to_save : path to save the checkpoints"""
        
        assert metric == 'MSE' or metric == 'WMSE'

        self.data_path =  data_path
        self.train_json_path : str = train_json_path
        self.val_json_path : str = val_json_path
        self.loss = nn.MSELoss() if metric == 'MSE' else PonderatedLoss(train_json_path)
        self.path_to_save = path_to_save
        self.layers_to_train = layers_to_train
        self.weights_path = weights_path
        self.batchsize = batchsize
        self.lr = lr
        self.epoch = epoch_n

    def train(self):
        device : str = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        train_data = load_data(self.data_path, self.train_json_path)
        train_loader = DataLoader(train_data, batch_size = self.batchsize, shuffle=True, pin_memory = True, num_workers = 5)
        val_data = load_data(self.data_path, self.val_json_path)
        val_loader = DataLoader(val_data, batch_size=self.batchsize, shuffle=False, pin_memory=True, num_workers=5)

        model = Model()
        model.to(device)

        loss_values : np.ndarray = np.empty(shape = (self.epoch, 2))

        # freeze some layers depending of layers_to_train param 
        if self.layers_to_train == 'FCN':
            for param in model.parameters():
                param.requires_grad = False

            # model.FCLayer.parameters()
            for param in model.FCLayer.parameters():
                param.requires_grad = True
            
        
        
        if self.layers_to_train == 'CNN':
            model.FCLayer.weight.requires_grad = False
            model.FCLayer.bias.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=self.lr, weight_decay=0.001)

        if self.weights_path:
            checkpoint = torch.load(self.weights_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
        model.train()

        for epoch in range(self.epoch):
            print(f'Epoch {epoch}')
            # train data
            train_loss = 0
            for i, data in enumerate(train_loader):
                loss_value = model.training_step(data, loss = self.loss, device=device)
                train_loss += loss_value.item()
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                print(f'Step {i}: {loss_value.item()}', end='\r')
            print('\n')
            loss_values[epoch, 0] = train_loss

            # val data
            # val_loss = 0
            # for i, data in enumerate(val_loader):
            #     loss_value = model.training_step(data, loss = self.loss, device = device)
            #     val_loss += loss_value.item()
            # print(f'Validation loss: {val_loss/(i+1)}\n')
            # loss_values[epoch,1] = val_loss
            
            torch.save(model.state_dict(), self.path_to_save.joinpath('model_%d.pt'%epoch))
        torch.save(model.state_dict(), self.path_to_save.joinpath('model_%d.pt'%self.epoch)) 

        loss_df = pd.DataFrame(loss_values, columns = ['train_loss', 'validation_loss'])

        with pd.ExcelWriter(self.path_to_save.joinpath('losses.xlsx')) as writer:  
            loss_df.to_excel(writer)

class Tester():
    def __init__(self, dataset_name : str,  data_path, test_json_path : str, weights_path : str, batchsize : int = 8):
        
        assert dataset_name == 'EVE' or dataset_name == 'UPNA_multigaze'
         
        self.test_json_path = test_json_path
        self.weights_path = weights_path
        self.batchsize = batchsize
        self.data_path = data_path
        self.dataset_name = dataset_name # EVE or UPNA_multigaze
    
    def test(self):
        device : str = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        test_data = load_data(self.data_path, self.test_json_path)
        test_loader = DataLoader(test_data, batch_size = self.batchsize, shuffle=False)

        model = Model()
        model.to(device)

        checkpoint = torch.load(self.weights_path)
        model.load_state_dict(checkpoint, strict=False)

        pred, names, session,  img_name, error = [], [], [], [], []

        for i, data in enumerate(test_loader,0):
            output, labels, resolution, name = model.validation_step(data, device)
            for k, gaze in enumerate(output):
                gaze = gaze.cpu().detach()
                xyGaze = [gaze[0]*(resolution[k][0]), gaze[1]*(resolution[k][1])]
                xyGaze = [xyGaze[0].item(), xyGaze[1].item()]
            
                xyTrue = [labels[k][0]*(resolution[k][0]), labels[k][1]*(resolution[k][1])]
                xyTrue = [xyTrue[0].item(), xyTrue[1].item()]

                xyTrue_cm = [x/38 for x in xyTrue]
                xyGaze_cm = [x/38 for x in xyGaze]

                er = dist_angular(xyTrue_cm, xyGaze_cm)
                xyGaze.append(xyTrue[0])
                xyGaze.append(xyTrue[1])
                pred.append(xyGaze)

                print(Path(name[k]), type(Path(name[k])))
                img_info = InfoEVE(Path(name[k])) if self.dataset_name == 'EVE' else InfoUPNA_multigaze(Path(name[k]))
        
                names.append(img_info.user)
                session.append(img_info.session)
                img_name.append(img_info.img_name)
              
                error.append(er)
            
            predictions = np.asarray(pred)
            predicts = pd.DataFrame(predictions, columns = ['pred_x', 'pred_y', 'gt_x', 'gt_y'])
            predicts['names'] = names
            predicts['session'] = session 
            predicts['img'] = img_name
            predicts['error'] = error
            
        return predicts



import json
import torch.nn as nn
import torch
from utils import get_grids_ponderations, loss_ponderation

class PonderatedLoss(nn.Module):
    def __init__(self, train_json_path):
        super(PonderatedLoss, self).__init__()
        with open(train_json_path, 'r') as f:
            train_json = json.load(f)
        self.ponderations = get_grids_ponderations(train_json)
        
    
    def forward(self, output, target):
        res =  target.cpu().numpy()
        
        w = loss_ponderation(res, self.ponderations)
        w = torch.tensor(w).cuda()
        
        return torch.sum(torch.mean((output-target)**2,dim=1)*w)/8
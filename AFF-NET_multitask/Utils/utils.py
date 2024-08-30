import math 
import numpy as np
import scipy 
from pathlib import Path
from typing import List

def dist_euclidea(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def dist_angular(p1,p2):
    d = dist_euclidea(p1,p2)
    tan_alpha = d/60
    return math.atan(tan_alpha)*180/math.pi

def loss_ponderation(point, ponderations):

    MARGIN = 100
    SCREEN_WIDTH =1920
    SCREEN_HEIGHT = 1080

    x_axis = np.linspace(MARGIN, SCREEN_WIDTH-MARGIN, 4)
    y_axis = np.linspace(MARGIN, SCREEN_HEIGHT-MARGIN, 4)
    meshgrid = np.array(np.meshgrid(x_axis, y_axis)).T.reshape(-1,2)
    arg_min = [np.argmin(scipy.spatial.distance.cdist(point, meshgrid, 'euclidean')[i,:]) for i in range(point.shape[0])]
    arg_min = np.array(arg_min)
    w = ponderations[arg_min]

    return w

def assign_cluster(points, grids = (4,4)):
    SCREEN_WIDTH =1920
    SCREEN_HEIGHT = 1080
    x_array = np.array(range(1,8,2))*SCREEN_WIDTH/8
    y_array = np.array(range(1,8,2))*SCREEN_HEIGHT/8
    meshgrid = np.array(np.meshgrid(x_array,y_array)).T.reshape(-1,2)
    distances = scipy.spatial.distance.cdist(points, meshgrid)

    closest_indices = np.argmin(distances, axis = 1)

    return closest_indices


def get_grids_ponderations(train_json):
    gt_list = [train_json[x]['gt'] for x in train_json]
    points_by_cluster = assign_cluster(gt_list)
    unique, counts = np.unique(points_by_cluster, return_counts=True)
    n = counts.max()
    
    
    return np.log(n/counts) + 1


def createDirectory(path : Path) -> None:
    if not path.exists():
        path.mkdir()


#### INFO CLASSES FOR EACH DATABASE
class InfoEVE():
    def __init__(self, path : Path):
        self.user, self.session, _, self.img_name = path.parts

class InfoUPNA_multigaze:
    def __init__(self, path : Path):
       _, _,self.user, self.session,_, _, self.img_name = path.parts
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 22:47:25 2025

@author: yihao
"""

# 

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import tqdm
import numpy as np
import pickle
import imageio
import argparse
import math
from torch.utils.data import Dataset
import soft_renderer as sr
import cv2 as cv
import math
from torch import sin, cos
from torch.autograd import Variable
from torch.utils.data import DataLoader
from LBS import LBS
from torch.nn import functional as F
from pathlib import Path
import json

if __name__=="__main__": 
    print("membrane point plot")
    point_index = 0
    obj_root = "D:/PhDProject_real_data/brunei_2023_bat_test_13_2/"
    folder = "stiffness_test"
    point_indexes = [1797,1632,299,569,849,274, 1910, 1681, 655,360]
    start_index = 1
    end_index = 100
    interval = 1
    xs = []
    ys = []
    zs = []
    for pose_index in range(start_index, end_index + interval, interval):
        if(pose_index < 10): 
            pose_index = f"000{pose_index}"
        elif(pose_index < 100): 
            pose_index = f"00{pose_index}"
        else: 
            pose_index = f"0{pose_index}"
        file_path = os.path.join(obj_root, folder, f"stiffness_{pose_index}.obj")
        # developing the use_previous to provide extra supervision
        mesh =  sr.Mesh.from_obj(file_path, load_texture=False, texture_res = 5, texture_type='surface')
       
        vertices = mesh.vertices.squeeze(0).cpu().numpy()[point_indexes[0]]
        print(vertices)
        xs.append(vertices[0])
        ys.append(vertices[1])
        zs.append(vertices[2])
        
    plt.plot(xs)
    plt.title("x translation")
    plt.show()
    plt.plot(ys)
    plt.title("y translation")
    plt.show()
    plt.plot(zs)
    plt.title("z translation")
    plt.show()
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

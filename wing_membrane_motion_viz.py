# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 13:57:27 2025

@author: yihao
"""
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
    root = "D:/PhDProject_real_data/brunei_2023_bat_test_13_2/"
    target_folders = ["tension_none","tension_test_0001", "tension_test_001", "tension_test_01","tension_test_1", "tension_test_10"]
    target_folders = [target_folders[0],target_folders[-1]]
    target_vertex_coord = 966
    obj_file_prefix = "bat_test_13_2"
    total_np_list = []
    for index, target_folder in enumerate(target_folders):
        target_list = []
        for pose_index in range(200): 
            if(pose_index < 10): 
                pose_index = f"000{pose_index}"
            elif(pose_index < 100): 
                pose_index = f"00{pose_index}"
            else: 
                pose_index = f"0{pose_index}"
            obj_file_path = os.path.join(root, target_folders[index], f"{obj_file_prefix}_{pose_index}.obj")
             
            template_mesh = sr.Mesh.from_obj(obj_file_path, load_texture=False, texture_res = 5, texture_type='surface')
            vertices = template_mesh.vertices.squeeze().cpu().numpy()
            target_list.append(vertices[target_vertex_coord]*20)
        points = np.array(target_list)
        total_np_list.append(points)
            
    
    
    for index, target_folder in enumerate(target_folders):
        plt.plot(total_np_list[index][:, 0])
    plt.legend(target_folders)
    plt.show()
    for index, target_folder in enumerate(target_folders):
        plt.plot(total_np_list[index][:, 1])
    plt.legend(target_folders)
    plt.show()
    for index, target_folder in enumerate(target_folders):
        plt.plot(total_np_list[index][:, 2])
    plt.legend(target_folders)
    plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
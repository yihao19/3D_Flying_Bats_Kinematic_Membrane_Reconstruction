# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 22:57:01 2025

@author: yihao
"""
import os
from general_utils import read_json_file
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
def list_subfolders(path):
    subfolders = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            subfolders.append(entry)
    return subfolders


if __name__=="__main__": 
    print("IOU loss summary for data in 2024")
    root_folder = "D:/PhDProject_real_data"
    subfolders = list_subfolders(root_folder)
    mean_list = []
    std_list = []
    indexes = []
    names = []
    index = 0
    counter = 1
    for subfolder in tqdm(subfolders[:], desc="folder processing"): 
        if subfolder in ["RHISED001_FlightTest3_2_4"]:
            continue
        path = os.path.join(root_folder, subfolder, "rearrange_pose")
        if not os.path.exists(path):
            print("Path doesn't exists, continue")
            continue
        pose_index_folders = list_subfolders(path)
        if(len(pose_index_folders) == 1): 
            print("not enough file, continue")
            continue
        loss_list = []
        for pose_index_folder in pose_index_folders: 
            json_file_path = os.path.join(path, pose_index_folder, "output.json")
            if not os.path.exists(json_file_path): 
                continue
            json_file = read_json_file(json_file_path)
            loss_list.append(json_file['IOU'])
        if(len(loss_list) <= 50): 
            continue
        indexes.append(index)
        index += 20
        names.append(counter)
        counter += 1
        mean = np.mean(loss_list)
        std = np.std(loss_list)
        if(mean is None or std is None):
            continue
        print(f"subfolder: {subfolder}  ", mean, "   ", std)
        mean_list.append(mean)
        std_list.append(std)
    
    plt.bar(indexes, mean_list,edgecolor='grey', color='grey',width=10)
    plt.tick_params(axis='x', labelsize=6)
    plt.xticks(indexes, names)
    plt.errorbar(indexes, mean_list, std_list,ls='none',color='black', elinewidth=1,capthick=1, capsize = 5)
    plt.savefig("all_iou.svg")
    print("Path does not exist")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
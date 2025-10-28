# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 16:14:37 2025

@author: yihao
"""

import os
import json
import matplotlib.pyplot as plt
def read_json_file(file_path): 
    with open(file_path) as f: 
        pose_dict = json.load(f)
        
    return pose_dict


if __name__=="__main__": 
    test_name = "brunei_2023_bat_test_13_2"
    root = f"D:/PhDProject_real_data/{test_name}/membrane_optimization_physical_attributes"
    epoch_number = 4
    
    start_index = 200
    end_index = 250
    attribute_index= 2
    
    physical_attributes_list = []
    
    for index in range(start_index, end_index): 
        file_path = os.path.join(root, f"bayesian_attrib_opt_{index}_{epoch_number}.json")
        attrib = read_json_file(file_path)
        physical_attributes_list.append(attrib['physical_attributes'][attribute_index])
    
    plt.plot(physical_attributes_list)
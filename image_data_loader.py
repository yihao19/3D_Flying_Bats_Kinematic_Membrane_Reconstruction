# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:56:28 2025

@author: yihao
"""


import os
import numpy as np
from torch.utils.data import Dataset
import cv2 as cv
from pathlib import Path
import json
from torch.utils.data import DataLoader

class Image_dataset(Dataset):
    def __init__(self, 
                 camera_meta_path:str, 
                 camera_list_path:str, 
                 silouette_image_path:str, 
                 current_pose:int, 
                 use_previous:bool):
        self.camera_meta_path = camera_meta_path
        self.camera_list_path = camera_list_path
        self.silouette_image_path = silouette_image_path
        self.use_previous = use_previous
        self.current_pose = current_pose
        # read the file
        
        camera_list_file = os.path.join(self.camera_list_path,str(self.current_pose), "camera.txt")
        
        
        camera_list_file = open(camera_list_file)
        
        camera_list_string = camera_list_file.read()
        camera_list_string = camera_list_string[1: len(camera_list_string)-1]
        camera_list = camera_list_string.split(', ')
       
        if(len(camera_list) <= 0):
            raise Exception("The number of silouettee image is: {}, not enough images!".format(len(camera_list)))

        entire_camera_matrix = np.loadtxt(os.path.join(self.camera_meta_path, 'camera_meta.txt'))
        self.image_list = []  # store the image path
        
        counter = 0
        glitch_camera_index = []#['2', '6', '11', '12', '13']  # glithes camera index that need to get rid of
        camera_list = set(camera_list).difference(set(glitch_camera_index))
        self.camera_number = len(camera_list)
        camera_matrix = np.zeros((self.camera_number, 12))
        for index in camera_list:
            index = int(index)
            image_name = "camera{}.png".format(index)
            self.image_list.append(image_name)
            camera_matrix[counter] = entire_camera_matrix[index-1]
            counter += 1
        camera_matrix = np.reshape(camera_matrix, (self.camera_number, 3, 4))
        self.camera_matrix = camera_matrix # camera matrix initialization
        
    def __len__(self):
        return self.camera_number
    def __getitem__(self, idx):
        # return the data sample indicated by the passed index
        
        index = idx % self.camera_number 

        camera_matrix = self.camera_matrix[index]
        
        image_path = os.path.join(self.silouette_image_path,str(self.current_pose), self.image_list[index])
        
        mask_image = cv.imread(image_path).astype('float32')[:, :, 0] / 255.
        mask_image = np.expand_dims(mask_image, -1)
        #cv.imwrite("test.png",255 * mask_image)
        mask_image = mask_image.transpose((2, 0, 1))
        
        
        if(self.use_previous == False):
            prev_pose = 0
            pre_local_adjust = 0
            try: 
                estimated_location_file = open(os.path.join(self.silouette_image_path, str(self.current_pose),'estimated_location.txt'))
                estimated_location_string = estimated_location_file.read()
                parts = estimated_location_string.split(' ')
                x_average = float(parts[0])
                y_average = float(parts[1])
                z_average = float(parts[2])
                estimated_location = np.array([x_average, y_average, z_average]).astype('float32') # randomly assign offset for
            except: 
                output_json_path = os.path.join(self.camera_list_path, str(self.current_pose), "output.json")
                file = open(output_json_path)
                current_pose = json.load(file)
                estimated_location = np.array(current_pose['template_displacement']).astype('float32')

        else:
            prev_pose_index = self.current_pose - 1
            current_path = Path(self.silouette_image_path)
            curr_root = current_path.parent
            prev_output_path = os.path.join(curr_root, str(prev_pose_index), "output.json")
            file = open(prev_output_path)
            prev_output = json.load(file) 
            prev_pose = np.array(prev_output['pose'])
            pre_local_adjust = np.array(prev_output['local_adjust'])
            estimated_location = np.array(prev_output['template_displacement']).astype('float32')
        sample = {'mask': mask_image, 
                  'camera_matrix':camera_matrix.astype('float32'), 
                  'prev_pose':prev_pose, 
                  'pre_local_adjust': pre_local_adjust,
                  'estimated_location':estimated_location}
        return sample
    


def image_dataloader(camera_meta_path:str, 
                     camera_list_path:str, 
                     silhouette_image_path:str, 
                     current_pose:int, 
                     use_previous:bool):
    dataset =  Image_dataset(camera_meta_path, camera_list_path, silhouette_image_path, current_pose, use_previous)
    
    batch_size =dataset.camera_number 
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dataloader, batch_size

if __name__=="__main__": 
    print("image data loader")















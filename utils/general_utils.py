# -*- coding: utf-8 -*-
"""
Created on Fri May  9 23:15:42 2025

@author: yihao
"""
import json

def read_json_file(file_path): 
    with open(file_path) as f: 
        pose_dict = json.load(f)
        
    return pose_dict


def save_json_file(input_dict, file_path): 
    with open(file_path, "w") as output:
        json.dump(input_dict, output)
    
def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union ).sum() / intersect.nelement()


def item_transform(item): 
    return item



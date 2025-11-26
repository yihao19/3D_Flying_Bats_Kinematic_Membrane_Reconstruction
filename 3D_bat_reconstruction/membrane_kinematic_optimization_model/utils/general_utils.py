# -*- coding: utf-8 -*-
"""
Created on Fri May  9 23:15:42 2025

@author: yihao
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
def read_pose_json(json_file, bone_index):
    output_json = 1
    f = open(json_file)
    output_json = json.load(f)
    
    pose_array = output_json['pose']
    output_json_x = []
    output_json_y = []
    output_json_z = []
    for bone in bone_index: 
        output_json_x.append(pose_array[bone][0])
        output_json_y.append(pose_array[bone][1])
        output_json_z.append(pose_array[bone][2])
    return output_json_x, output_json_y, output_json_z, output_json['template_displacement']

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


def rodrigues(pose):
    """
    utils function to convert the euler angles into quaternion

    Parameters
    ----------
    pose : TYPE
        DESCRIPTION.

    Returns
    -------
    quat : TYPE
        DESCRIPTION.

    """
    l1norm = np.linalg.norm(pose + [1e-8, 1e-8, 1e-8])
    angle = l1norm
    normalized = pose / l1norm
    angle = angle * 0.5
    v_cos = np.cos(angle)
    v_sin = np.sin(angle)
    
    quat = [v_cos, v_sin * normalized[0], v_sin * normalized[1], v_sin * normalized[2] ]
    
    return quat

def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/np.linalg.norm(norm_quat)
    w, x, y, z = norm_quat[0], norm_quat[1], norm_quat[2], norm_quat[3]

    w2, x2, y2, z2 = w**2, x**2, y**2, z**2
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = np.array([[w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz],
              [2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx],
              [2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2]])
    return (rotMat) 

def rotmat_to_euler(rot_matrix):

    """
    Convert a 3x3 rotation matrix to ZYX intrinsic Euler angles 
    (yaw-pitch-roll = psi, theta, phi).

    Returns:
        (psi, theta, phi)
    """
    # Protect against numerical issues
    r = R.from_matrix(rot_matrix)
    euler = r.as_euler('xyz', degrees=False)
    return euler

def kinematic_smoothing(kinematic_list:list, sigma:float = 1.0):
    """
    input shape: N * 40 * 3
    return shape N * 40 * 3
    """
    kinematic_array = np.array(kinematic_list)
    kinematic_array = np.swapaxes(kinematic_array, 0, 1)   # shape will be 40 * N * 3

    
    smoothed_kinematic = [gaussian_filter1d(kinematic_array[frame_index], sigma = sigma, axis=0) for frame_index in range(kinematic_array.shape[0])]
    smoothed_kinematic = np.array(smoothed_kinematic)
    smoothed_kinematic = np.swapaxes(smoothed_kinematic, 0, 1)
    return smoothed_kinematic 
def displacement_smoothing(displacement_list:list, sigma:float=1.0):
    """
    input shape: N * 3
    output shape: N * 3
    """
    displacement_array = np.array(displacement_list)
    smoothed_data = gaussian_filter1d(displacement_array, sigma=sigma, axis=0)

    return smoothed_data
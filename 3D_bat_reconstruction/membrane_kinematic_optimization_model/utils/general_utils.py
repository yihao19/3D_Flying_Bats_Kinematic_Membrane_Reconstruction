# -*- coding: utf-8 -*-
"""
Created on Fri May  9 23:15:42 2025

@author: yihao
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
import cv2
import os
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
        json.dump(input_dict, output, indent=4)
    
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
    smoothed_kinematic = np.array(smoothed_kinematic).astype('float32')
    smoothed_kinematic = np.swapaxes(smoothed_kinematic, 0, 1)
    return smoothed_kinematic 
def displacement_smoothing(displacement_list:list, sigma:float=1.0):
    """
    input shape: N * 3
    output shape: N * 3
    """
    displacement_array = np.array(displacement_list)
    smoothed_data = gaussian_filter1d(displacement_array, sigma=sigma, axis=0).astype('float32')

    return smoothed_data

def sample_sphere_surface(center, radius, n):
    """Uniform points ON sphere surface."""
    vec = np.random.normal(size=(n, 3))
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    return np.array(center) + radius * vec

def sample_sphere_volume(center, radius:float=0.03, n:int=300):
    """Uniform points INSIDE sphere volume."""
    dirs = np.random.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    u = np.random.random(size=n)
    rs = radius * (u ** (1/3.0))
    return np.array(center) + dirs * rs[:, None]

def sample_sphere_grid(center, radius, n_theta=30, n_phi=60):
    """Grid (lat-long) points ON sphere surface."""
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi)
    th, ph = np.meshgrid(theta, phi, indexing='xy')
    x = radius * np.sin(th) * np.cos(ph)
    y = radius * np.sin(th) * np.sin(ph)
    z = radius * np.cos(th)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1) + np.array(center)
    return pts

def projection_array(xyz, image, camera_matrix):
      proj = np.matmul(camera_matrix, xyz.transpose())
      proj = proj.transpose()

      proj[:, 0] = proj[:, 0] / (proj[:, 2] + 1e-5)
      proj[:, 1] = proj[:, 1] / (proj[:, 2] + 1e-5)
      proj = proj.astype(int)
      for index in range(proj.shape[0]):
          if(proj[index][0] < 0 or proj[index][0] >= 1280 or proj[index][1] < 0 or proj[index][1] >= 1024):
              continue
          x = proj[index][0]
          y = proj[index][1]
          image[y][x] = 255
      return image  

def if_keep_via_projection(xyz, image, camera_matrix):
    xyz = np.array([xyz[0],xyz[1],xyz[2],1])
    proj = np.matmul(camera_matrix, xyz.transpose())
    proj[0] = proj[0] / (proj[2] + 1e-10)
    proj[1] = proj[1] / (proj[2] + 1e-10)
    proj = proj.astype(int)
    keep = 0 
   
    if(proj[0] <0 or proj[0] >= 1280 or proj[1] < 0 or proj[1] >= 1024):
        return keep
    x = proj[0]
    y = proj[1]
    if(image[y][x][0] > 0):
        keep = 1
    return keep

def point_to_image(xyzs, image, camera_matrix):
    xyzs = np.array(xyzs)
    ones = np.ones(xyzs.shape[0])
    xyzs = np.stack([xyzs[:, 0], xyzs[:, 1],xyzs[:, 2],ones], axis=1)
    proj = np.matmul(camera_matrix, xyzs.transpose())
    proj = proj.transpose()

    proj[:, 0] = proj[:, 0] / (proj[:, 2] + 1e-5)
    proj[:, 1] = proj[:, 1] / (proj[:, 2] + 1e-5)
    proj = proj.astype(int)
    for index in range(proj.shape[0]):
        if(proj[index][0] < 0 or proj[index][0] >= 1280 or proj[index][1] < 0 or proj[index][1] >= 1024):
            continue
        x = proj[index][0]
        y = proj[index][1]
        cv2.circle(image, (x, y), 5, (255, 255, 0), -1)
    return image  

def list_subfolders(path) -> list:
    subfolders = []
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            subfolders.append(entry)
    return subfolders

def count_continuous_sublists(indices):
    if not indices:
        return 0

    count = 1
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            count += 1
    return count

def std_without_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = data[(data >= lower) & (data <= upper)]
    return np.std(filtered)  # sample standard deviation

def mean_without_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = data[(data >= lower) & (data <= upper)]
    return np.mean(filtered)  # sample standard deviation
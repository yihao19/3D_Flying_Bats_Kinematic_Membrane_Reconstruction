# -*- coding: utf-8 -*-
"""
Created on Sat May  3 20:10:13 2025

@author: yihao
"""

"""
Demo deform.
Deform template mesh based on input silhouettes and camera pose
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
from utils.general_utils import read_json_file
'''
trying to make the model learning euler angle and displacement
by using camera matrix

'''
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, './')

class Membrane_kinematic_model(nn.Module):
    def __init__(self, 
                 bone_skining_matrix_path:str, 
                 membrane_modified_obj_path:str,
                 pose_original_kinematic_path:str, 
                 template_scale_factor:float= 0.0035, 
                 device = "GPU"):
        super(Membrane_kinematic_model, self).__init__()
        """
    
        Parameters
        ----------
        membrane_modified_obj_path : TYPE
            membrane modified obj file using blender
        pose_original_kinematic_path : TYPE
            the original kinematic json containing all the info including 
        bone_skining_matrix_path : TYPE, optional
            DESCRIPTION. The default is './new_bat_params_version2_backward_membrane_24.pkl'.
        
        the LBS model need to be update using the membrane modified obj and original pose matrix
        Returns
        
        the new pose matrix will be export for next round of membrane optimization
        -------
        None.

        """
        
        """
        1. load membrane optimize mesh. 
        2. load template mesh. 
        """
        
        self.membrane_optimized_mesh = sr.Mesh.from_obj(membrane_modified_obj_path, load_texture=False, texture_res = 5, texture_type='surface')
        with open(os.path.join("/home/yihao19/3D_Flying_Bats_Kinematic_Membrane_Reconstruction/3D_bat_reconstruction/membrane_kinematic_optimization_model/model_template", bone_skining_matrix_path), 'rb') as f:
            data = pickle.load(f)
        self.template_mesh =  sr.Mesh.from_obj(membrane_modified_obj_path, load_texture=False, texture_res = 5, texture_type='surface')
        self.template_mesh.vertices = torch.tensor(data['v_template']).float().unsqueeze(0).cuda()
        self.template_mesh.faces = torch.tensor(data['faces']).unsqueeze(0).cuda()
        self.template_scale_factor = template_scale_factor
        self.template_skining = torch.tensor(data['weights']).unsqueeze(0).cuda()
        
        with open(pose_original_kinematic_path) as f: 
             pose_dict = json.load(f)
        
        if(len(pose_dict['pose']) == 34): 
            # this is the paper one design, append 6 more bones
            pose_dict['pose'].append([0,0,0])
            pose_dict['pose'].append([0,0,0])
            pose_dict['pose'].append([0,0,0])
            pose_dict['pose'].append([0,0,0])
            pose_dict['pose'].append([0,0,0])
            pose_dict['pose'].append([0,0,0])
            
        self.original_pose_np = np.array(pose_dict['pose'])
        root_pose = self.original_pose_np[0]
        ab_pose = self.original_pose_np[1]
        upper_body_pose =  self.original_pose_np[2]
        
        distance = np.array(pose_dict['template_displacement'])
        
        self.original_pose = torch.tensor(self.original_pose_np).unsqueeze(0).cuda()
        kintree_table = np.array([[ -1, 0, 1, 2, 2, 4, 5, 6, 7, 8, 9,  7,  11, 12, 7 , 14, 15, 2,  17, 18, 19, 20, 21, 22, 20, 24, 25, 20, 27, 28, 0,  30, 0,  32, 10, 13, 16, 23, 26, 29],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]])
      
        template_joints =  data['joints_matrix'][:3, :].transpose()
        template_joints_tail = data['joints_matrix'][3:, :].transpose()
        #joints = data['joints_matrix'][:3, :].transpose()
        #joints_tail = data['joints_matrix'][3:, :].transpose()
        
        #print(np.linalg.norm(joints_tail[7] - joints_tail[5]))

        self.joint_number = template_joints.shape[0]
        #kintree_table = data['kintree']  # numpy array that define the kinematic tree of the skeleton
        template_joints_tensor = torch.tensor(template_joints).unsqueeze(0).cuda()
        template_joints_tail_tensor = torch.tensor(template_joints_tail).unsqueeze(0).cuda()
        
        # define the kintree of the skeleton of the bat
        # define in the Blender 
      
        self.kintree_table = torch.tensor(kintree_table).cuda()
        self.register_buffer('template_joints',template_joints_tensor)
        self.register_buffer('template_joints_tail', template_joints_tail_tensor)
        self.register_buffer('vertices', self.template_mesh.vertices)
        self.register_buffer('faces', self.template_mesh.faces)
        self.parents = self.kintree_table[0].type(torch.LongTensor)
        
        # for each bone, then 
        

        template_LBS_model = LBS(self.template_joints, self.parents, self.template_skining)# define the LBS model using the initial template
        
        
        
        
        # based on the initial
        _vertices, new_joints = template_LBS_model(self.template_mesh.vertices,self.template_joints, self.original_pose, to_rotmats=True)
        _vertices, new_joints_tail = template_LBS_model(self.template_mesh.vertices,self.template_joints_tail, self.original_pose, to_rotmats=True)
        
        
        
        
        # new_joints is goint to be bind with the membrane optimized mesh 
        # the joints are still needed to be scaled and translated in order to bind with the membrane optimized mesh
        self.distance = torch.tensor(distance).repeat(1, new_joints.shape[1], 1).cuda()
        joints = self.template_scale_factor * new_joints + self.distance
        joints_tail =  self.template_scale_factor * new_joints_tail + self.distance
        
        
        
        
        
        # bind the joints with the optimized membrane template, right now, skeleton should be 
        # align with the membrane modified template
        new_joints_tail = torch.tensor(joints_tail).cuda()
        new_joints = torch.tensor(joints).cuda() # the joint 
        
        # reverse the original pose on the template optimized mesh
        #self.original_pose = torch.tensor(np.zeros((1, 34, 3))).cuda()
        
       
        
        
        membrane_LBS =LBS(new_joints, self.parents, self.template_skining)
        membrane_LBS_tail = LBS(new_joints_tail, self.parents, self.template_skining)
        
        self.displacement_range = 0.1
        
        
        #restore the reattached template orientation
        # step1: rotate the root with the inverse the the origianal pose
        for bone_index in range(40): 
            original_bone_pose = self.original_pose_np[bone_index]
            reverse_bone_pose = [-original_bone_pose[0], -original_bone_pose[1], -original_bone_pose[2]]
            
            reverse_pose = np.zeros((40, 3))
            reverse_pose[bone_index] = reverse_bone_pose
            reverse_pose = torch.tensor(reverse_pose).unsqueeze(0).cuda()
            
            #rotate the membrane optimized root, after this step the rotation should be align with the original template
            vertices, joints = membrane_LBS(self.membrane_optimized_mesh.vertices, new_joints, reverse_pose, to_rotmats=True)
            
            vertices, joints_tail = membrane_LBS(self.membrane_optimized_mesh.vertices, new_joints_tail, reverse_pose, to_rotmats=True)
            self.membrane_optimized_mesh.vertices = vertices
            new_joints = joints
            new_joints_tail = joints_tail
            membrane_LBS =LBS(new_joints, self.parents, self.template_skining)
            membrane_LBS_tail = LBS(new_joints_tail, self.parents, self.template_skining)
        # output the point cloud and the joint location to validate
        '''
        mesh = sr.Mesh(vertices.repeat(1, 1, 1),self.faces.repeat(1, 1, 1))
        mesh.save_obj("restore.obj")
        joints = joints[0].cpu().numpy()
        vertices = mesh.vertices[0].cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2])

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_xlim(-1, 1) # Same as above
        ax.set_ylim(-1, 1) # Same as above
        ax.set_zlim(-1, 1) # Same as above
        
        plt.show()
        return
        '''
        
        self.membrane_optimized_mesh.vertices = vertices
        self.new_joints = new_joints
        self.new_joints_tail = new_joints_tail
        self.membrane_LBS = LBS(self.new_joints, self.parents, self.template_skining)
        self.membrane_LBS_tail = LBS(self.new_joints_tail,self.parents, self.template_skining)
        
        
        self.vertices_number = self.template_mesh.num_vertices
        # optimize for displacement of the center of the mesh ball
        self.register_parameter('displacement', nn.Parameter(torch.zeros(1,1,3)))
        self.register_parameter('scale', nn.Parameter(torch.ones(1)))
        self.register_parameter('local_adjust', nn.Parameter(torch.zeros(1, self.vertices_number, 3))) # apply a small local adjustment on the template 
                                                                                                       # to adjust the template
        
        #self.register_parameter('pitch', nn.Parameter(torch.zeros(1)))
        #self.register_parameter('yaw', nn.Parameter(torch.zeros(1)))
        #self.register_parameter('roll', nn.Parameter(torch.zeros(1)))
        
        self.register_parameter('joint_0',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_1',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_2',nn.Parameter(torch.zeros(1, 3))) # 3 DOF
        self.register_parameter('joint_3',nn.Parameter(torch.zeros(1, 3)))    # 1 DOF Z
        self.register_parameter('joint_4',nn.Parameter(torch.zeros(1, 3))) # 3 DOF
        self.register_parameter('joint_5',nn.Parameter(torch.zeros(1, 3)))    # 1 DOF
        self.register_parameter('joint_6',nn.Parameter(torch.zeros(1, 3))) # 3 DOF
        self.register_parameter('joint_7',nn.Parameter(torch.zeros(1, 3))) # 3 DOF
        self.register_parameter('joint_8',nn.Parameter(torch.zeros(1, 3)))    # 1 DOF Z
        self.register_parameter('joint_9',nn.Parameter(torch.zeros(1, 3)))    # 1 DOF Z
        self.register_parameter('joint_10',nn.Parameter(torch.zeros(1, 3)))# 3 DOF
        self.register_parameter('joint_11',nn.Parameter(torch.zeros(1, 3)))   # 1 DOF Z
        self.register_parameter('joint_12',nn.Parameter(torch.zeros(1, 3)))# 3 DOF
        self.register_parameter('joint_13',nn.Parameter(torch.zeros(1, 3)))   # 1 DOF Z
        self.register_parameter('joint_14',nn.Parameter(torch.zeros(1, 3)))# 3 DOF
        self.register_parameter('joint_15',nn.Parameter(torch.zeros(1, 3)))   # 1 DOF Z
        self.register_parameter('joint_16',nn.Parameter(torch.zeros(1, 3)))# 3 DOF
        self.register_parameter('joint_17',nn.Parameter(torch.zeros(1, 3)))   # 1 DOF Z
        self.register_parameter('joint_18',nn.Parameter(torch.zeros(1, 3)))   # 1 DOF Z
        self.register_parameter('joint_19',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_20',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_21',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_22',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_23',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_24',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_25',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_26',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_27',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_28',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_29',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_30',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_31',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_32',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_33',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_34',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_35',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_36',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_37',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_38',nn.Parameter(torch.zeros(1, 3)))
        self.register_parameter('joint_39',nn.Parameter(torch.zeros(1, 3)))
        
        # make small displacement of the 
        #self.register_parameter("pose_tensor", nn.Parameter(torch.zeros(1,19,3)))
        # add the pose_tensor of the previous mesh model
        self.pose_tensor = torch.zeros((40, 3)).cuda()  #define how much adjustment we do on the original template
        
        self.laplacian_smoothing = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        
    def forward(self, batch_size):
        """
        Parameters
        ----------
        batch_size : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        self.pose_tensor[0][0] = self.original_pose_np[0][0] + math.pi / 6 * torch.tanh(self.joint_0[0][0]) #+0.2 * math.pi # for modifying template pose
        self.pose_tensor[0][1] = self.original_pose_np[0][1] + math.pi / 6 * torch.tanh(self.joint_0[0][1]) #
        self.pose_tensor[0][2] = self.original_pose_np[0][2] + math.pi / 6 * torch.tanh(self.joint_0[0][2]) #+0.2 * math.pi
            
        
        #self.pose_tensor[0][2] = math.pi / 2 * torch.tanh(self.joint_0[0][2]) +math.pi
        self.pose_tensor[1][0] = self.original_pose_np[1][0] + math.pi / 9 * torch.tanh(self.joint_1[0][0])
        self.pose_tensor[1][1] = self.original_pose_np[1][1] + math.pi / 9 * torch.tanh(self.joint_1[0][1])
        self.pose_tensor[1][2] = self.original_pose_np[1][2] + math.pi / 9 * torch.tanh(self.joint_1[0][2])
        
        
        self.pose_tensor[2][0] = self.original_pose_np[2][0] + math.pi / 9 * torch.tanh(self.joint_2[0][0])
        self.pose_tensor[2][1] = self.original_pose_np[2][1] + math.pi / 9 * torch.tanh(self.joint_2[0][1])
        self.pose_tensor[2][2] = self.original_pose_np[2][2] + math.pi / 9 * torch.tanh(self.joint_2[0][2])
        #self.pose_tensor[3][:] = math.pi / 18 * torch.tanh(self.joint_3)
        
        #self.pose_tensor[1][0] = self.original_pose_np[1][0] + math.pi / 9 * torch.tanh(self.joint_1[0][0])
        #self.pose_tensor[1][1] = self.original_pose_np[1][1] + math.pi / 9 * torch.tanh(self.joint_1[0][1])
        #self.pose_tensor[1][2] = self.original_pose_np[1][2] + math.pi / 9 * torch.tanh(self.joint_1[0][2])
        
        
        
        #self.pose_tensor[2][0] = self.original_pose_np[2][0] + math.pi / 9 * torch.tanh(self.joint_2[0][0])  # make the neck bone trainable(slightly)
        #self.pose_tensor[2][1] = self.original_pose_np[2][1] + math.pi / 9 * torch.tanh(self.joint_2[0][1])
        #self.pose_tensor[2][2] = self.original_pose_np[2][2] + math.pi / 9 * torch.tanh(self.joint_2[0][2])
        #for the shoulder using the symmetric deformation

        self.pose_tensor[5][0] = self.original_pose_np[5][0] +math.pi / 9 * torch.tanh(self.joint_5[0][0])
        self.pose_tensor[5][1] = self.original_pose_np[5][1] +math.pi / 9 * torch.tanh(self.joint_5[0][1])
        self.pose_tensor[5][2] = self.original_pose_np[5][2] +math.pi / 9 * torch.tanh(self.joint_5[0][2])
        
        
        self.pose_tensor[18][0] = self.original_pose_np[18][0] + math.pi / 9 * torch.tanh(self.joint_18[0][0])
        self.pose_tensor[18][1] = self.original_pose_np[18][1] + math.pi / 9 * torch.tanh(self.joint_18[0][1])
        self.pose_tensor[18][2] = self.original_pose_np[18][2] + math.pi / 9 * torch.tanh(self.joint_18[0][2])
        
        #self.pose_tensor[4][0] = ( math.pi / 3 * torch.tanh(self.joint_4[0][0]))
        #self.pose_tensor[4][1] = (  math.pi / 3 * torch.tanh(self.joint_4[0][1]))
        self.pose_tensor[4][2] =  self.original_pose_np[4][2] + math.pi / 9 * torch.tanh(self.joint_4[0][2])
        
        
        
        #self.pose_tensor[17][0] = (  math.pi / 3 * torch.tanh(self.joint_4[0][0]))
        #self.pose_tensor[17][1] = (  -math.pi / 3 * torch.tanh(self.joint_4[0][1]))
        self.pose_tensor[17][2] =  self.original_pose_np[17][2] - math.pi / 9 * torch.tanh(self.joint_4[0][2])
        

        self.pose_tensor[6][0]  =   self.original_pose_np[6][0] + math.pi / 9 * torch.tanh(self.joint_6[0][0])
        self.pose_tensor[6][1]  =   self.original_pose_np[6][1] + math.pi / 9 * torch.tanh(self.joint_6[0][1])
        self.pose_tensor[6][2]  =   self.original_pose_np[6][2] + math.pi / 9 * torch.tanh(self.joint_6[0][2])
        
        self.pose_tensor[19][0] =   self.original_pose_np[19][0] + math.pi / 9 * torch.tanh(self.joint_19[0][0])
        self.pose_tensor[19][1] =   self.original_pose_np[19][1] + math.pi / 9 * torch.tanh(self.joint_19[0][1])
        self.pose_tensor[19][2] =   self.original_pose_np[19][2] + math.pi / 9 * torch.tanh(self.joint_19[0][2])
        
        
        
        self.pose_tensor[7][0] = self.original_pose_np[7][0] + math.pi / 9 * torch.tanh(self.joint_7[0][0])
        self.pose_tensor[7][1] = self.original_pose_np[7][1] + math.pi / 9 * torch.tanh(self.joint_7[0][1])
        self.pose_tensor[7][2] = self.original_pose_np[7][2] + math.pi / 9 * torch.tanh(self.joint_7[0][2])
        
        self.pose_tensor[20][0] = self.original_pose_np[20][0] +  math.pi / 9 * torch.tanh(self.joint_20[0][0])
        self.pose_tensor[20][1] = self.original_pose_np[20][1] +  math.pi / 9 * torch.tanh(self.joint_20[0][1])
        self.pose_tensor[20][2] = self.original_pose_np[20][2] +  math.pi / 9 * torch.tanh(self.joint_20[0][2])
 
        # For the third, forth, fifth bone, based on the template design, only make it able to rotate around the y axis
        # for opening and closing the wing
        # if the third finger moves 1 rad, the forth finger will move 0.5 rad and and fifth will move 0.3 rad
        
        
        #self.pose_tensor[8][:] = math.pi / 6 * torch.tanh(self.joint_8)
        self.pose_tensor[8][1] =  self.original_pose_np[8][1] + math.pi / 9 * torch.tanh(self.joint_8[0][1])
        #self.pose_tensor[8][:] = math.pi / 6 * torch.tanh(self.joint_8)
        #self.pose_tensor[9][:] = math.pi / 18 * torch.tanh(self.joint_9)
        #self.pose_tensor[10][:] = math.pi / 18 * torch.tanh(self.joint_10)
        
        #self.pose_tensor[11][:] = math.pi /  6 * torch.tanh(self.joint_11)
        self.pose_tensor[11][1] =  self.original_pose_np[11][1] + 0.5 * math.pi /  9 * torch.tanh(self.joint_8[0][1])
        #self.pose_tensor[11][:] = math.pi /  6 * torch.tanh(self.joint_11)
        #self.pose_tensor[12][:] = math.pi / 18 * torch.tanh(self.joint_12)
        #self.pose_tensor[13][:] = math.pi / 18 * torch.tanh(self.joint_13)
        
        
        #self.pose_tensor[14][:] = math.pi /  6 * torch.tanh(self.joint_14)
        self.pose_tensor[14][1] =  self.original_pose_np[14][1] +  0.3 * math.pi /  9 * torch.tanh(self.joint_8[0][1])
        #self.pose_tensor[14][:] = math.pi /  6 * torch.tanh(self.joint_14)
        #self.pose_tensor[15][:] = math.pi / 18 * torch.tanh(self.joint_15)
        #self.pose_tensor[16][:] = math.pi / 18 * torch.tanh(self.joint_16)
        
        #self.pose_tensor[21][:] = math.pi /  6 * torch.tanh(self.joint_21)
        self.pose_tensor[21][1] =  self.original_pose_np[21][1] + math.pi /  9 * torch.tanh(self.joint_21[0][1])
        #self.pose_tensor[21][:] = math.pi /  6 * torch.tanh(self.joint_21)
        #self.pose_tensor[22][:] = math.pi / 18 * torch.tanh(self.joint_22)
        #self.pose_tensor[23][:] = math.pi / 18 * torch.tanh(self.joint_23)
        
        #self.pose_tensor[24][:] = math.pi /  6 * torch.tanh(self.joint_24)
        self.pose_tensor[24][1] =  self.original_pose_np[24][1] + 0.5 * math.pi /  9 * torch.tanh(self.joint_21[0][1])
        #self.pose_tensor[24][:] = math.pi /  6 * torch.tanh(self.joint_24)
        #self.pose_tensor[25][:] = math.pi / 18 * torch.tanh(self.joint_25)
        #self.pose_tensor[26][:] = math.pi / 18 * torch.tanh(self.joint_26)
        
        #self.pose_tensor[27][:] = math.pi /  6 * torch.tanh(self.joint_27)
        self.pose_tensor[27][1] =  self.original_pose_np[27][1] + 0.3 * math.pi /  9 * torch.tanh(self.joint_21[0][1])
        #self.pose_tensor[27][:] = math.pi /  6 * torch.tanh(self.joint_27)
        #self.pose_tensor[28][:] = math.pi / 18 * torch.tanh(self.joint_28)
        #self.pose_tensor[29][:] = math.pi / 18 * torch.tanh(self.joint_29)
        
        self.pose_tensor[30][0] =  self.original_pose_np[30][0] + math.pi /  9 * torch.tanh(self.joint_30[0][0])
        self.pose_tensor[30][1] =  self.original_pose_np[30][1] + math.pi /  9 * torch.tanh(self.joint_30[0][1])
        self.pose_tensor[30][2] =  self.original_pose_np[30][2] + math.pi /  9 * torch.tanh(self.joint_30[0][2])
        #self.pose_tensor[31][:] = math.pi / 18 * torch.tanh(self.joint_31)
        
        self.pose_tensor[32][0] =  self.original_pose_np[32][0] + math.pi /  9 * torch.tanh(self.joint_32[0][0])
        self.pose_tensor[32][1] =  self.original_pose_np[32][1] + math.pi /  9 * torch.tanh(self.joint_32[0][1])
        self.pose_tensor[32][2] =  self.original_pose_np[32][2] + math.pi /  9 * torch.tanh(self.joint_32[0][2])
        #self.pose_tensor[33][:] = math.pi / 18 * torch.tanh(self.joint_33)
       
            
        #self.pose_tensor[1] = self.joint_0

        #self.pose_tensor = self.pose_tensor.unsqueeze(0)
        # model will deform the mesh and then add the predetermined offset and learned displacement
        # apply the small adjustment on template first
        
        
        vertices, joints = self.membrane_LBS(self.membrane_optimized_mesh.vertices,self.new_joints, self.pose_tensor, to_rotmats=True)
        
        
        _, joints_tail = self.membrane_LBS(self.membrane_optimized_mesh.vertices,self.new_joints_tail, self.pose_tensor, to_rotmats=True)
        #self.pose_tensor = self.pose_tensor.squeeze()
        
        vertices = vertices + self.displacement_range * torch.tanh(self.displacement.repeat(1, self.vertices_number, 1)).cuda() 
        joints =   joints +   self.displacement_range * torch.tanh(self.displacement.repeat(1, self.joint_number, 1)).cuda() 
        # define a new regulaization term to prevent the wing folder crazy
        # make sure the distance between 23 26 and 26 29 are the same
        # make sure the distance between 10, 13 and 13, 16 are the same
        distance_1 = joints[0][23] - joints[0][26]
        distance_2 = joints[0][29] - joints[0][26]
        reg_1 = torch.abs(torch.norm(distance_1) - torch.norm(distance_2))
        distance_1 = joints[0][10] - joints[0][13]
        distance_2 = joints[0][16] - joints[0][13]
        
        reg_2 = torch.abs(torch.norm(distance_1) -  torch.norm(distance_2))
        
        reg = reg_1 + reg_2 
            
        # assign weight for bone angle that for manuever that the bone with more muscles should move more
        # right now, classify the bones into three categories based on their muscle groups
        # weights: class 1: 0.1
        #          class 2: 0.3
        #          class 3: 0.5
        # measured with l2 norm
        # with loss function like this, the model will encourage bone that close to body deform the most
        
        # level 1 bone 4, 17, 5, 18
        # level 2 bone 6, 19 , 30, 32
        # level 3 bone 8, 11, 14, 21, 24, 27, 31, 33
        
        
        bone_prior_1 = 0.01 * (torch.norm(self.pose_tensor[4][2]) +torch.norm(self.pose_tensor[17][2]) + torch.norm(self.pose_tensor[5])+torch.norm(self.pose_tensor[18]))
        
                                                                                                           
        bone_prior_2 = 0.2 * (torch.norm(self.pose_tensor[6]) + torch.norm(self.pose_tensor[19]) + 
                              torch.norm(self.pose_tensor[7])  + torch.norm(self.pose_tensor[20]))
                     
        bone_prior_3 = 0.3 * (
                              +torch.norm(self.pose_tensor[8]) 
                              +torch.norm(self.pose_tensor[11])
                              +torch.norm(self.pose_tensor[14])
                              +torch.norm(self.pose_tensor[21])
                              +torch.norm(self.pose_tensor[24])
                              +torch.norm(self.pose_tensor[27])
                              +torch.norm(self.pose_tensor[30])
                              +torch.norm(self.pose_tensor[32]))                                                                                                   
       
        # regulate some bone  
        # define a symmetric loss for each level of bones
        # the x axis rotation in the same direction while the y and z rotate in the opposite direction
        bone_prior = bone_prior_1 + bone_prior_2 + bone_prior_3
        
        bone_symmetric_1 = torch.norm(self.pose_tensor[4][2] + self.pose_tensor[17][2]) + \
                           torch.norm(self.pose_tensor[5][0] - self.pose_tensor[18][0]) + \
                           torch.norm(self.pose_tensor[5][1] + self.pose_tensor[18][1]) + \
                           torch.norm(self.pose_tensor[5][2] + self.pose_tensor[18][2]) 
                           
        bone_symmetric_2 = torch.norm(self.pose_tensor[6][0] - self.pose_tensor[19][0]) + \
                           torch.norm(self.pose_tensor[6][1] + self.pose_tensor[19][1]) + \
                           torch.norm(self.pose_tensor[6][2] + self.pose_tensor[19][2]) + \
                           torch.norm(self.pose_tensor[7][0] - self.pose_tensor[20][0]) + \
                           torch.norm(self.pose_tensor[7][1] + self.pose_tensor[20][1]) + \
                           torch.norm(self.pose_tensor[7][2] + self.pose_tensor[20][2])
                               
        bone_symmetric_3 = torch.norm(self.pose_tensor[8][0] - self.pose_tensor[21][0]) + \
                           torch.norm(self.pose_tensor[8][1] + self.pose_tensor[21][1]) + \
                           torch.norm(self.pose_tensor[8][2] + self.pose_tensor[21][2]) + \
                           torch.norm(self.pose_tensor[11][0] - self.pose_tensor[24][0]) + \
                           torch.norm(self.pose_tensor[11][1] + self.pose_tensor[24][1]) + \
                           torch.norm(self.pose_tensor[11][2] + self.pose_tensor[24][2]) + \
                           torch.norm(self.pose_tensor[14][0] - self.pose_tensor[27][0]) + \
                           torch.norm(self.pose_tensor[14][1] + self.pose_tensor[27][1]) + \
                           torch.norm(self.pose_tensor[14][2] + self.pose_tensor[27][2]) + \
                           torch.norm(self.pose_tensor[30][0] - self.pose_tensor[32][0]) + \
                           torch.norm(self.pose_tensor[30][1] + self.pose_tensor[32][1]) + \
                           torch.norm(self.pose_tensor[30][2] + self.pose_tensor[32][2])
                        
        bone_symmetric = 0.1 * bone_symmetric_1 + 0.2* bone_symmetric_2 + 0.5 * bone_symmetric_3
        
        laplacian_loss = self.laplacian_smoothing(vertices).mean()
        #pose_tensor = self.pose_tensor.unsqueeze(0)
        
        
        return sr.Mesh(vertices.repeat(batch_size, 1, 1),self.faces.repeat(batch_size, 1, 1)), \
                       laplacian_loss, \
                       reg, \
                       bone_prior, \
                       bone_symmetric, \
                       self.pose_tensor.unsqueeze(0), \
                       self.scale,       \
                       self.distance + self.displacement_range * torch.tanh(self.displacement), \
                       torch.tensor(0), \
                       vertices,            \
                       joints,              \
                       joints_tail,         \
                       torch.tensor(self.distance),        \
                       torch.tensor(0)
                       

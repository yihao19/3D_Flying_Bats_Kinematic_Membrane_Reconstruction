# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:52:28 2025

@author: yihao
"""
import os
from image_data_loader import image_dataloader
from membrane_kinematic_model import Membrane_kinematic_model
from kinematic_model import Kinematic_model
from utils.general_utils import (neg_iou_loss, 
                                item_transform, 
                                read_json_file,
                                save_json_file,
                                read_pose_json, 
                                quat_to_rotmat, 
                                rotmat_to_euler,
                                rodrigues, 
                                kinematic_smoothing, 
                                displacement_smoothing,
                                sample_sphere_volume,
                                if_keep_via_projection,
                                point_to_image)
from utils.blender_utils import update_simulations, get_target_objs, y_forward_z_up
import torch
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt import Optimizer
from torch.utils import data
import torch.multiprocessing as mp
import soft_renderer as sr
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from consecutive_frame_search_space import searching_space_angular, searching_space_linear
from matplotlib.ticker import MaxNLocator
from skopt.space import Real, Integer
from scipy.ndimage import gaussian_filter1d
import subprocess
from num2words import num2words
import imageio
import cv2




PROJECT_ROOT = "/home/yihao19"
class TqdmCallBack:
    def __init__(self, total_iterations, description):
        self.pbar = tqdm(total = total_iterations, desc= description)
    def __call__(self, res): 
        self.pbar.update(1)
    def close(self):
        self.pbar.close()
        
class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class Optimize_Driver(): 
    def __init__(self, 
                 project_root_path:str,
                 project_name:str, 
                 test_name:str, 
                 start_pose:int, 
                 end_pose:int, 
                 current_pose_index:int,
                 half_window_size:int, 
                 membrane_optimized_frame:int,
                 kinematic_opt_epoch:int, 
                 membrane_opt_epoch:int, 
                 membrane_kinematic_opt_epoch:int,
                 whole_opt_epoch:int, 
                 model_template:str, 
                 if_use_previous_attr:bool,
                 if_use_previous_kinamatics:bool,
                 opposite_direction:bool,
                 template_flip:bool=False, 
                 glitched_camera_indexes:list = []):
        self.project_root_path = project_root_path
        self.project_name = project_name
        self.test_name = test_name
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.current_pose_index = current_pose_index
        self.current_epoch = 0
        self.half_window_size = half_window_size
        self.membrane_optimized_frame = membrane_optimized_frame
        assert self.membrane_optimized_frame <= self.half_window_size
        self.kinematic_opt_epoch = kinematic_opt_epoch
        self.membrane_opt_epoch = membrane_opt_epoch
        self.whole_opt_epoch = whole_opt_epoch
        self.model_template = model_template
        self.membrane_kinematic_opt_epoch = membrane_kinematic_opt_epoch
        
        
        self.camera_meta_path_root = os.path.join(self.project_root_path,
                                            self.project_name, 
                                            self.test_name, 
                                            "rearrange_pose")
        self.camera_list_path_root = os.path.join(self.project_root_path,
                                            self.project_name, 
                                            self.test_name, 
                                            "rearrange_pose")
        
        self.silhouette_image_path_root = os.path.join(self.project_root_path, 
                                                       self.project_name, 
                                                       self.test_name, 
                                                       "rearrange_pose")
        
        self.kinematic_save_path_root = os.path.join(self.project_root_path,
                                            self.project_name, 
                                            self.test_name, 
                                            "rearrange_pose")
        
        self.raw_mesh_save_path_root  = os.path.join(self.project_root_path,
                                            self.project_name, 
                                            self.test_name, 
                                            "reconstruction")
        if not os.path.exists(self.raw_mesh_save_path_root):
            os.makedirs(self.raw_mesh_save_path_root)
        self.original_mesh_save_path_root = os.path.join(self.project_root_path, 
                                                self.project_name, 
                                                self.test_name, 
                                                "original_reconstruction")
        
        if not os.path.exists(self.original_mesh_save_path_root):
            os.makedirs(self.original_mesh_save_path_root)
        
        folder_index = num2words(self.membrane_optimized_frame).lower()
        self.membrane_optimized_frame_str = folder_index
        self.membrane_optimize_mesh_save_path_root = os.path.join(self.project_root_path, 
                                                self.project_name, 
                                                self.test_name, 
                                                f"membrane_optimized_mesh")
        
        if not os.path.exists(self.membrane_optimize_mesh_save_path_root):
            os.makedirs(self.membrane_optimize_mesh_save_path_root)
            
        self.membrane_kinematic_optimized_mesh_save_root =  os.path.join(self.project_root_path, 
                                                self.project_name, 
                                                self.test_name, 
                                                "membrane_kinematic_optimized_mesh_final")
        
        if not os.path.exists(self.membrane_kinematic_optimized_mesh_save_root):
            os.makedirs(self.membrane_kinematic_optimized_mesh_save_root)
        
        
        self.membrane_optimize_attributes_save_path_root =  os.path.join(self.project_root_path, 
                                                self.project_name, 
                                                self.test_name, 
                                                "membrane_optimization_physical_attributes",
                                                f"{folder_index}_average")
        
        if not os.path.exists(self.membrane_optimize_attributes_save_path_root):
            os.makedirs(self.membrane_optimize_attributes_save_path_root)
            
        self.blender_render_save_path = os.path.join(self.project_root_path,
                                                     self.project_name, 
                                                     self.test_name, 
                                                     "blender_render")
        self.original_reconstruction_smooth_kinematic_path = os.path.join(self.project_root_path,
                                                     self.project_name, 
                                                     self.test_name, 
                                                     "original_kinematic_smooth")
                                                     
        if not os.path.exists(self.original_reconstruction_smooth_kinematic_path):
             os.makedirs(self.original_reconstruction_smooth_kinematic_path)

        self.calibration_check_path = os.path.join(self.project_root_path, 
                                                   self.project_name, 
                                                   self.test_name, 
                                                   "calibration_check")
        if not os.path.exists(self.calibration_check_path):
            os.makedirs(self.calibration_check_path)
        self.result_path_root = './result'
        self.use_previous_attr = if_use_previous_attr
        self.use_previous = if_use_previous_kinamatics
        if(self.start_pose > self.end_pose):
            self.reverse = True
        else:
            self.reverse = False
        self.opposite_direction = opposite_direction  # back and force
        self.template_flip = template_flip   # to accomodate calibration difference between 2023 and 2024  2023:False, 2024:True
        self.image_size = (1024,1280)
        self.glitched_camera_indexes = glitched_camera_indexes
    def kinematic_optimize(self, pose_index, use_previous = False, reverse = False) -> None: 
        """
        Parameters
        ----------
        pose_index : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
       
        kinematic_model = Kinematic_model(bone_skining_matrix_name=self.model_template, 
                                          use_previous=use_previous,
                                          opposite_direction=self.opposite_direction,
                                          template_flip=self.template_flip).cuda()
        optimizer = torch.optim.Adam(kinematic_model.parameters(), 0.005,betas=(0.5, 0.99))
        train_dataloader, batch_size = image_dataloader(
            self.camera_meta_path_root, 
            self.camera_list_path_root, 
            self.silhouette_image_path_root, 
            pose_index, 
            self.use_previous, 
            self.reverse, 
            self.glitched_camera_indexes
            )
        epoch = tqdm(list(range(0,self.kinematic_opt_epoch)))
        for i in epoch:
            
            for training_sample in train_dataloader:
          
                images_gt = training_sample['mask'].cuda()
                camera_matrix = training_sample['camera_matrix'].cuda()
                prev_pose = training_sample['prev_pose'].cuda()
                prev_local_adjust = training_sample['pre_local_adjust'].cuda()
                estimated_location = training_sample['estimated_location'].cuda()
                mesh, laplacian_loss,wing_tip_reg, bone_prior, bone_symmetric, current_pose, scale, displacement, local_adjust, vertices, joints, joints_tail, displacement,weight_tensor = kinematic_model(batch_size, estimated_location, self.use_previous, prev_pose[0])
                renderer = sr.SoftRenderer(image_height=self.image_size[0], image_width=self.image_size[1],sigma_val=1e-6,
                                           camera_mode='projection', P = camera_matrix ,orig_height=self.image_size[0], orig_width=self.image_size[1], 
                                           near=0, far=100)
                         
                images_pred = renderer.render_mesh(mesh)
                
                IOU_loss = neg_iou_loss(images_pred[:, -1], images_gt[:, 0])
                loss = IOU_loss #+ 1 * l2_norm 
              
                pose_loss = torch.tensor(0)
                l2_adjust = torch.tensor(0)
                if(self.use_previous == True):
                    pose_loss = 0.005 * torch.norm(current_pose[:][:] - prev_pose[:][:]) #+ 0.1 * torch.norm(current_pose[1:][:] - prev_pose[1:][:])
                    #l2_adjust = 0.02 * torch.norm(local_adjust - prev_local_adjust)
                image_number = len(images_gt)
                bone_symmetric_coeff = 0.005
                bone_prior_coeff =0.0
                if(image_number >= 10):
                    bone_symmetric_coeff = 0.0
                elif(image_number >= 5 and image_number < 10):
                    bone_symmetric_coeff = 0.01
                elif(image_number >= 3 and image_number < 5):
                    bone_symmetric_coeff = 0.5
                elif(image_number < 3):
                    bone_symmetric_coeff = 1
                loss = IOU_loss + pose_loss  + bone_prior_coeff * bone_prior + bone_symmetric_coeff * bone_symmetric 
                epoch.set_description(f'Project name: {self.test_name} Image_num: {len(images_gt)} pose: {pose_index} IOU loss: {IOU_loss.item():.4f}')
                #epoch.set_description('IOU Loss: %.4f   Pose Loss: %.4f  Wingtip_reg: %.4f  Bone prior: %.4f  Bone symmetry: %.4f  L2 adjust: %.4f' % (IOU_loss.item(),pose_loss.item(), wing_tip_reg.item(), 0.1 * bone_prior.item(), bone_symmetric.item(), l2_adjust.item()))
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        output_mesh,laplacian_loss, wing_tip_reg,bone_prior, bone_symmetric, current_pose, current_scale, current_displacement, local_adjust, vertices, joints, joints_tail,  displacement,weight_tensor = kinematic_model(1, estimated_location,self.use_previous, prev_pose[0])
      
        current_pose = current_pose.cpu().detach().numpy().tolist()
        current_scale = current_scale.cpu().detach().item()
        current_displacement = current_displacement.cpu().detach().numpy().tolist()
        local_adjust = local_adjust.cpu().detach().numpy().tolist()
        vertices = vertices.cpu().detach().numpy()
        vertices_mean = np.mean(vertices[0], axis=0)
        joints = joints.cpu().detach().numpy()
        joints_tail = joints_tail.cpu().detach().numpy()
        template_displacement = displacement[0].cpu().detach().numpy().tolist()[0]
        output_skining_weight =weight_tensor[0].cpu().detach().numpy().tolist()
        output_mesh.save_obj(os.path.join(self.raw_mesh_save_path_root, '{}_bat_{}.obj'.format(self.test_name, pose_index)), save_texture=False)

        save_dict = {"pose":current_pose,
                     "joints": joints.tolist(), 
                     "joints_tail":joints_tail.tolist(), 
                     "scale":current_scale, 
                     "displacement":current_displacement, 
                     "local_adjust":local_adjust, 
                     "pose_loss":pose_loss.cpu().detach().item(), 
                     "IOU":IOU_loss.cpu().detach().item(),
                     "vertices_mean": vertices_mean.tolist(),
                     "template_displacement":template_displacement,
                     "skining_tensor": output_skining_weight}

        save_file_path = os.path.join(self.kinematic_save_path_root,str(pose_index) ,"output.json")
        save_json_file(save_dict, save_file_path)
    
    def original_reconstruction(self, pose_index):
        """
        Parameters
        ----------
        pose_index : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
       
        kinematic_model = Kinematic_model(bone_skining_matrix_name=self.model_template,
                                          opposite_direction=self.opposite_direction).cuda()
        output_json_path = os.path.join(self.camera_list_path_root, str(pose_index), "output.json")
        file = open(output_json_path)
        current_pose = json.load(file)
        estimated_location = np.array([current_pose['template_displacement']]).astype('float32')
        pose = current_pose['pose']
        if(len(pose) == 34):
            # acamodate the new template with 40 bone
            pose.append([0,0,0])
            pose.append([0,0,0])
            pose.append([0,0,0])
            pose.append([0,0,0])
            pose.append([0,0,0])
            pose.append([0,0,0])

        pose = np.array(pose).astype('float32')
       
        estimated_location = torch.tensor(estimated_location).cuda()
        pose = torch.tensor(pose).cuda()
        output_mesh = kinematic_model.render_original(estimated_location, pose)
        output_mesh.save_obj(os.path.join(self.original_mesh_save_path_root, f'{pose_index}.obj'), save_texture=False)
        return
    
    def membrane_optimization_loss_bayes_mode(self, membrane_physical_attribues): 
        """
        this will call the blender to render the membrane using the passed physical attributes (deprecated)
        
        Returns
        -------
        None.
        """
        name_parts = self.test_name.split('_')
        blender_test_name = f"{name_parts[-4]}_{name_parts[-3]}_{name_parts[-2]}_{name_parts[-1]}"
        launch_args = dict(
            scene=Path(__file__).parent / "membrane_blender" /f"{blender_test_name}"/f"{blender_test_name}.blend",
            script=Path(__file__).parent / "membrane_blender" /f"{blender_test_name}"/f"{blender_test_name}.blend.py",
            num_instances=self.SIM_INSTANCES,
            named_sockets=["DATA", "CTRL"],
        )
        with btt.BlenderLauncher(**launch_args) as bl:
            # Create remote dataset
            addr = bl.launch_info.addresses["DATA"]
            sim_ds = btt.RemoteIterableDataset(addr, item_transform=item_transform)
            sim_dl = data.DataLoader(sim_ds, batch_size=1, num_workers=0, shuffle=False)
            # Create a control channel to each Blender instance. We use this channel to
            # communicate new shape parameters to be rendered.
            addr = bl.launch_info.addresses["CTRL"]
            remotes = [btt.DuplexChannel(a) for a in addr]
            # sample the mass of the cloth modifier to modify it 
            
            # modify the current mesh renderer
            update_simulations(remotes, [membrane_physical_attribues])
            
            # fetch the objs that you want to optimize the 
            rendered_frame = half_window_size * 1 + 1 + 1 # total number of frame + buffer size
            _target_objs_list = get_target_objs(
                sim_dl, remotes, n= rendered_frame
            )
        # the rendered obj will be stored in the temp folder
        total_iou_loss_list  = []
        total_iou_loss = 0
        
        for pose_index in range(self.current_pose_index-self.membrane_optimized_frame + 1, self.current_pose_index + 1): 
        #for pose_index in range(self.current_pose_index, self.current_pose_index + 1): 
            train_dataloader, batch_size = image_dataloader(
                self.camera_meta_path_root, 
                self.camera_list_path_root, 
                self.silhouette_image_path_root, 
                pose_index, 
                self.use_previous
                )
            for training_sample in train_dataloader:
                images_gt = training_sample['mask'].cuda()
                camera_matrix = training_sample['camera_matrix'].cuda()
            
            renderer = sr.SoftRenderer(image_height=self.image_size[0], image_width=self.image_size[1],sigma_val=1e-6,
                                        camera_mode='projection', P = camera_matrix ,orig_height=self.image_size[0], orig_width=self.image_size[1], 
                                        near=0, far=100)
            
            if not os.path.exists(self.blender_render_save_path):
                os.makedirs(self.blender_render_save_path)
                
            #cloth_obj_path = 'D:/PhDProject_real_data/cloth_simulation/{}/{}.obj'.format(test_name, pose_index)
            cloth_obj_path = os.path.join(self.blender_render_save_path, f"{pose_index}.obj")
            
            mesh = sr.Mesh.from_obj(cloth_obj_path, load_texture=False, texture_res = 1, texture_type='surface')
            vertices = mesh.vertices
            faces = mesh.faces
            vertices = y_forward_z_up(vertices) # manually change the orientation of the exported obj
            mesh = sr.Mesh(vertices.repeat(batch_size, 1, 1),faces.repeat(batch_size, 1, 1))
            
            # save the orientation correct obj for future use
            if not os.path.exists(self.membrane_optimize_mesh_save_path_root):
                os.makedirs(self.membrane_optimize_mesh_save_path_root)
                
            mesh.save_obj(os.path.join(self.membrane_optimize_mesh_save_path_root, '{}.obj'.format(pose_index)), save_texture=False)
            #if(pose_index == self.current_pose_index):
            images_pred = renderer.render_mesh(mesh)
            with torch.no_grad():
                iou_loss = neg_iou_loss(images_pred[:, -1], images_gt[:, 0])      
                total_iou_loss_list.append(iou_loss.item())
                total_iou_loss += iou_loss.item()
            print(f"IOU loss: {iou_loss.item()}") 
            #np.save(f"{self.result_path_root}/{self.test_name}_membrane_optimize_{pose_index}_{self.current_epoch}.npy", np.array(iou_loss.item()))
        frame_number =  self.half_window_size + 1
        average_iou_loss = total_iou_loss / self.membrane_optimized_frame
        max_loss = max(total_iou_loss_list)
        output_attribute_list = {"physical_attributes":membrane_physical_attribues} # only save the attributes that are stiffness related
        output_path= os.path.join(self.membrane_optimize_attributes_save_path_root, f"bayesian_attrib_opt_linear_{self.current_pose_index}_{self.current_epoch}_{self.membrane_opt_counter}.json")
        save_json_file(output_attribute_list, output_path)
        self.membrane_opt_counter += 1
        # check if the previous stiffness exist, if so, read as reference
        
        ref_coef = 40
        use_previous = False
        pre_coef = 60
        if(not use_previous):
            print("max_loss: ", max_loss, "   reg_term: ", ref_coef * membrane_physical_attribues[0]**2 )
            print("mean_loss: ", average_iou_loss, "   reg_term: ", ref_coef * membrane_physical_attribues[0]**2)
            return average_iou_loss + ref_coef * membrane_physical_attribues[0]**2  # add it as regularization
        else: 
            # if using the previous read the tension attributes of the previous frame
            prev_attribute_list = []
            for counter in range(self.membrane_opt_epoch - 2, self.membrane_opt_epoch):
                prev_attributes_path =  os.path.join(self.membrane_optimize_attributes_save_path_root, f"bayesian_attrib_opt_linear_{self.current_pose_index-1}_{self.current_epoch}_{counter}.json")
                attrib = read_json_file(prev_attributes_path)
                prev_attribute_list.append(attrib['physical_attributes'][0])
            prev_tension = np.min(prev_attribute_list)
            print("max_loss: ", max_loss, "   reg_term: ", ref_coef * (membrane_physical_attribues[0] - prev_tension)**2 )
            print("mean_loss: ", average_iou_loss, "   reg_term: ", ref_coef * (membrane_physical_attribues[0] - prev_tension)**2 )
            print("prev pose: ", self.current_pose_index-1 )
            return average_iou_loss + ref_coef * (membrane_physical_attribues[0] - prev_tension)**2 +  pre_coef * (membrane_physical_attribues[0])**2 # add it as regularization

    def iou_loss_cal(self, pose_index:int, reconstruction_type:str = "original") -> None:
        """Calculate the iou loss using the corresponding obj and images. 
        """
        train_dataloader, batch_size = image_dataloader(
            self.camera_meta_path_root, 
            self.camera_list_path_root, 
            self.silhouette_image_path_root, 
            pose_index, 
            False, 
            reverse = False
            )
        for training_sample in train_dataloader:
            images_gt = training_sample['mask'].cuda()
            camera_matrix = training_sample['camera_matrix'].cuda()
        
        renderer = sr.SoftRenderer(image_height=self.image_size[0], image_width=self.image_size[1],sigma_val=1e-6,
                                    camera_mode='projection', P = camera_matrix ,orig_height=self.image_size[0], orig_width=self.image_size[1], 
                                    near=0, far=100)
        
        if not os.path.exists(self.blender_render_save_path):
            os.makedirs(self.blender_render_save_path)
        
        if reconstruction_type == "original": 
            obj_path = os.path.join(self.original_mesh_save_path_root, f"{pose_index}.obj")
        elif reconstruction_type == "membrane_opt":
            obj_path = os.path.join(self.membrane_optimize_mesh_save_path_root, f"{pose_index}.obj")
        elif reconstruction_type == "kinematic_smooth":
            obj_path = os.path.join(self.original_reconstruction_smooth_kinematic_path, f"{pose_index}.obj")
        elif reconstruction_type == "kinematic_membrane_opt":
            obj_path = os.path.join(self.membrane_kinematic_optimized_mesh_save_root, f"{pose_index}.obj")
        #cloth_obj_path = 'D:/PhDProject_real_data/cloth_simulation/{}/{}.obj'.format(test_name, pose_index)

        mesh = sr.Mesh.from_obj(obj_path, load_texture=False, texture_res = 1, texture_type='surface')
     
        mesh = sr.Mesh(mesh.vertices.repeat(batch_size, 1, 1),mesh.faces.repeat(batch_size, 1, 1))
        
        # save the orientation correct obj for future use
        if not os.path.exists(self.membrane_optimize_mesh_save_path_root):
            os.makedirs(self.membrane_optimize_mesh_save_path_root)
        #if(pose_index == self.current_pose_index):
        images_pred = renderer.render_mesh(mesh)
        with torch.no_grad():
            iou_loss = neg_iou_loss(images_pred[:, -1], images_gt[:, 0])
        return iou_loss.item()
        
    def membrane_optimization_loss_bayes_mode_v2(self, membrane_physical_attribues): 
        """
        this is the new version of the origianl function "membrane optimization loss bayes mode that support background render"

        Parameters
        ----------
        membrane_physical_attributes : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        name_parts = self.test_name.split('_')
        blender_test_name = self.test_name.replace("Brunei_2023_", "")
        #blender_test_name = f"{name_parts[-4]}_{name_parts[-3]}_{name_parts[-2]}_{name_parts[-1]}"
        #1. render the mesh optimized mesh
        cmd = []
        cmd.append("./blender")
        cmd.append(f"{PROJECT_ROOT}/3D_Flying_Bats_Kinematic_Membrane_Reconstruction/3D_bat_reconstruction/membrane_kinematic_optimization_model/membrane_blender/{blender_test_name}/{blender_test_name}.blend")
        cmd.append("-b")  # run in backgroud
        cmd.append("--python-use-system-env")
        cmd.append("--python")
        cmd.append(f"{PROJECT_ROOT}/3D_Flying_Bats_Kinematic_Membrane_Reconstruction/3D_bat_reconstruction/membrane_kinematic_optimization_model/blender_script_template.blend.py")
        cmd.append("--")

        cmd.append(f"{self.project_root_path}{self.project_name}")
        cmd.append(f"{blender_test_name}")
        cmd.append(f"{self.current_pose_index-self.half_window_size}")
        cmd.append(f"{self.current_pose_index}")
        cmd.append(f"{membrane_physical_attribues[0]}")
        cmd.append(f"{self.current_epoch}")
        cmd.append("True")
        _result = subprocess.run(cmd, capture_output=True,cwd="/home/yihao19/blender-4.5.4-linux-x64")
        #2. calculate loss
        total_iou_loss = 0
        total_iou_loss_list = []
        # implement multi-processing
        for pose_index in range(self.current_pose_index-self.membrane_optimized_frame+1, self.current_pose_index + 1): 
        #for pose_index in range(self.current_pose_index, self.current_pose_index + 1):
            train_dataloader, batch_size = image_dataloader(
                self.camera_meta_path_root, 
                self.camera_list_path_root, 
                self.silhouette_image_path_root, 
                pose_index, 
                self.use_previous, 
                reverse=False
                )
            for training_sample in train_dataloader:
                images_gt = training_sample['mask'].cuda()
                camera_matrix = training_sample['camera_matrix'].cuda()
            
            renderer = sr.SoftRenderer(image_height=self.image_size[0], image_width=self.image_size[1],sigma_val=1e-6,
                                        camera_mode='projection', P = camera_matrix ,orig_height=self.image_size[0], orig_width=self.image_size[1], 
                                        near=0, far=100)
            
            if not os.path.exists(self.blender_render_save_path):
                os.makedirs(self.blender_render_save_path)
                
            #cloth_obj_path = 'D:/PhDProject_real_data/cloth_simulation/{}/{}.obj'.format(test_name, pose_index)
            cloth_obj_path = os.path.join(self.blender_render_save_path, f"{pose_index}.obj")
            
            mesh = sr.Mesh.from_obj(cloth_obj_path, load_texture=False, texture_res = 1, texture_type='surface')
            vertices = mesh.vertices
            faces = mesh.faces
            vertices = y_forward_z_up(vertices) # manually change the orientation of the exported obj
            mesh = sr.Mesh(vertices.repeat(batch_size, 1, 1),faces.repeat(batch_size, 1, 1))
            
            # save the orientation correct obj for future use
            if not os.path.exists(self.membrane_optimize_mesh_save_path_root):
                os.makedirs(self.membrane_optimize_mesh_save_path_root)
                
            mesh.save_obj(os.path.join(self.membrane_optimize_mesh_save_path_root, '{}.obj'.format(pose_index)), save_texture=False)
            #if(pose_index == self.current_pose_index):
            images_pred = renderer.render_mesh(mesh)
            with torch.no_grad():
                iou_loss = neg_iou_loss(images_pred[:, -1], images_gt[:, 0])      
                total_iou_loss_list.append(iou_loss.item())
                total_iou_loss += iou_loss.item()
            #print(f"IOU loss: {iou_loss.item()}") 
            

        average_iou_loss = sum(total_iou_loss_list) / self.membrane_optimized_frame
        max_loss = max(total_iou_loss_list)
        output_attribute_list = {"physical_attributes":membrane_physical_attribues} # only save the attributes that are stiffness related
        output_path= os.path.join(self.membrane_optimize_attributes_save_path_root, f"bayesian_attrib_opt_linear_{self.current_pose_index}_{self.current_epoch}_{self.membrane_opt_counter}.json")
        save_json_file(output_attribute_list, output_path)
        self.membrane_opt_counter += 1
        # check if the previous stiffness exist, if so, read as reference
        
        ref_coef = 40
        pre_coef = 200
        IOU_coef = 1
        if(not self.use_previous_attr):
            #print("max_loss: ", max_loss, "   reg_term: ", ref_coef * membrane_physical_attribues[0]**2 )
            #print("mean_loss: ", average_iou_loss, "   reg_term: ", ref_coef * membrane_physical_attribues[0]**2)
            return IOU_coef * average_iou_loss + ref_coef * membrane_physical_attribues[0]**2  # add it as regularization
        else: 
            # if using the previous read the tension attributes of the previous frame
            prev_attribute_list = []
            for counter in range(self.membrane_opt_epoch - 5, self.membrane_opt_epoch):
                prev_attributes_path =  os.path.join(self.membrane_optimize_attributes_save_path_root, f"bayesian_attrib_opt_linear_{self.current_pose_index-1}_{self.current_epoch}_{counter}.json")
                attrib = read_json_file(prev_attributes_path)
                prev_attribute_list.append(attrib['physical_attributes'][0])
            prev_tension = np.min(prev_attribute_list)
            #print("max_loss: ", max_loss, "   reg_term: ", ref_coef * (membrane_physical_attribues[0] - prev_tension)**2 )
            #print("mean_loss: ", average_iou_loss, "   reg_term: ", ref_coef * (membrane_physical_attribues[0] - prev_tension)**2 )
            #print("prev pose: ", self.current_pose_index-1 )
            return IOU_coef * average_iou_loss + pre_coef * (membrane_physical_attribues[0] - prev_tension)**2 +  ref_coef * (membrane_physical_attribues[0])**2 # add it as regularization
        
    def membrane_optimize_bayesian(self, epoch_number): 
        """
        this function will implement the bayesian network for optimizing the membrane parameters

        Returns
        -------
        None.

        """
        self.membrane_opt_counter = 0
        prev_pose_index = self.current_pose_index - 1
        prev_attribute_path = os.path.join(self.membrane_optimize_attributes_save_path_root,f"bayesian_attrib_opt_linear_{prev_pose_index}_{self.whole_opt_epoch-1}_{self.membrane_opt_epoch-1}.json" )
        searching_space = []
        '''
        try:
            attribute_file = open(prev_attribute_path)
            attribute = json.load(attribute_file)
            prev_tension = attribute['physical_attributes'][0]
        
            searching_space.append(Real(max(0, prev_tension - 0.01), min(0.1, prev_tension + 0.01), name='tension_stiffness'))
            opt_iteration = 20
            print(f"read attributes from previous pose: {prev_pose_index}: {prev_attribute_path}")
        except:
            print(f"attributes from previous pose: {prev_pose_index} doesn't exist:  {prev_attribute_path}")
        '''
        searching_space = searching_space_angular
        
        self.current_epoch = epoch_number
        
        # check if the previous pkl file exist, if not optimize from scratch
       
        gp = GaussianProcessRegressor(kernel=Matern(length_scale=10), 
                              alpha=1e-6, normalize_y=True, random_state=0)
        tqdm_callback = TqdmCallBack(total_iterations=self.membrane_opt_epoch, description=f"{self.test_name}_membrane optimizing pose: {self.current_pose_index}")
        result = gp_minimize(self.membrane_optimization_loss_bayes_mode_v2,     # The function to minimize
                             searching_space,                   # The search space
                             n_calls= self.membrane_opt_epoch,              # The number of evaluations
                             random_state=42,
                             initial_point_generator='lhs',
                             n_initial_points=int(0.8 * self.membrane_opt_epoch),
                             acq_optimizer="lbfgs",
                             kappa = 0.19,
                             acq_func = "LCB",
                             noise=1e-6,
                             base_estimator = gp,
                             callback=[tqdm_callback])
        tqdm_callback.close()         # R
        return result
        
        
    def membrane_kinematic_optimize(self,pose_index, pipeline_epoch=0):
        """Coupled the membrane optimized mesh and initial kinematics and optimize the kinematics
        """
        membrane_modified_obj_path = os.path.join(self.membrane_optimize_mesh_save_path_root, f"{pose_index}.obj")
        pose_original_kinematic_path = os.path.join(self.kinematic_save_path_root, f"{pose_index}", f"membrane_output_{pipeline_epoch}.json")
        membrane_kinematic_model = Membrane_kinematic_model(bone_skining_matrix_path=self.model_template,
                                                            membrane_modified_obj_path=membrane_modified_obj_path, 
                                                            pose_original_kinematic_path=pose_original_kinematic_path).cuda()
        optimizer = torch.optim.Adam(membrane_kinematic_model.parameters(), 0.005 ,betas=(0.5, 0.99))
        train_dataloader, batch_size = image_dataloader(
            self.camera_meta_path_root, 
            self.camera_list_path_root, 
            self.silhouette_image_path_root, 
            pose_index, 
            self.use_previous
            )
        epoch = tqdm(list(range(0,self.membrane_kinematic_opt_epoch)))
        for i in epoch:
            
            for training_sample in train_dataloader:
          
                images_gt = training_sample['mask'].cuda()
                camera_matrix = training_sample['camera_matrix'].cuda()
                prev_pose = training_sample['prev_pose'].cuda()
                prev_local_adjust = training_sample['pre_local_adjust'].cuda()
                estimated_location = training_sample['estimated_location'].cuda()
                mesh, laplacian_loss,wing_tip_reg, bone_prior, bone_symmetric, current_pose, scale, displacement, local_adjust, vertices, joints, joints_tail, displacement,weight_tensor = membrane_kinematic_model(batch_size)
                renderer = sr.SoftRenderer(image_height=self.image_size[0], image_width=self.image_size[1],sigma_val=1e-6,
                                           camera_mode='projection', P = camera_matrix ,orig_height=self.image_size[0], orig_width=self.image_size[1], 
                                           near=0, far=100)
                         
                images_pred = renderer.render_mesh(mesh)
                
                IOU_loss = neg_iou_loss(images_pred[:, -1], images_gt[:, 0])
                loss = IOU_loss #+ 1 * l2_norm 
              
                pose_loss = torch.tensor(0)
                l2_adjust = torch.tensor(0)
                if(self.use_previous == True):
                    pose_loss = 0.005 * torch.norm(current_pose[:][:] - prev_pose[:][:]) #+ 0.1 * torch.norm(current_pose[1:][:] - prev_pose[1:][:])
                    l2_adjust = 0.02 * torch.norm(local_adjust - prev_local_adjust)
            
                loss = IOU_loss + pose_loss  + 0.005 * bone_prior + 0.005 * bone_symmetric
                epoch.set_description(f'Project name: {self.test_name}   pose: {pose_index}    IOU loss: {IOU_loss.item():.4f}')
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        output_mesh,laplacian_loss, wing_tip_reg,bone_prior, bone_symmetric, current_pose, current_scale, current_displacement, local_adjust, vertices, joints, joints_tail,  displacement,weight_tensor = membrane_kinematic_model(1)
      
        current_pose = current_pose.cpu().detach().numpy().tolist()[0]
        current_scale = current_scale.cpu().detach().item()
        current_displacement = current_displacement.cpu().detach().numpy().tolist()
        local_adjust = local_adjust.cpu().detach().numpy().tolist()
        vertices = vertices.cpu().detach().numpy()
        vertices_mean = np.mean(vertices[0], axis=0)
        joints = joints.cpu().detach().numpy()
        joints_tail = joints_tail.cpu().detach().numpy()
        template_displacement = displacement[0].cpu().detach().numpy().tolist()[0]
        #output_skining_weight =weight_tensor[0].cpu().detach().numpy().tolist()
        save_dict = {"pose":current_pose,
                     "joints": joints.tolist(), 
                     "joints_tail":joints_tail.tolist(), 
                     "scale":current_scale, 
                     "displacement":current_displacement, 
                     "local_adjust":local_adjust, 
                     "pose_loss":pose_loss.cpu().detach().item(), 
                     "IOU":IOU_loss.cpu().detach().item(),
                     "vertices_mean": vertices_mean.tolist(),
                     "template_displacement":template_displacement}
        if(pose_index == self.current_pose_index):
            output_mesh.save_obj(os.path.join(self.membrane_kinematic_optimized_mesh_save_root, f"{pose_index}.obj"), save_texture=False)
            save_file_path = os.path.join(self.kinematic_save_path_root,str(pose_index), f"membrane_output_{pipeline_epoch}.json")
            save_json_file(save_dict, save_file_path)
        return
    
    def sequence_membrane_kinematic_optimize(self, pipeline_epoch= 0):
        """
        entire sequence kinematic optmization (membrane fixed)

        Returns
        -------
        None.

        """
        for pose_index in range(self.current_pose_index-self.half_window_size, self.current_pose_index +1):
            self.membrane_kinematic_optimize(pose_index, pipeline_epoch)
    
    def output_json_IOU_loss(self, json_name): 
        iou_loss = []
            
        for index in range(self.start_pose, self.end_pose): 
            json_file_path = os.path.join(self.camera_list_path_root, str(index), f"{json_name}.json")
            output_dict = read_json_file(json_file_path)
            iou_loss.append(output_dict['IOU'])
        
        plt.plot(iou_loss)
        plt.xlabel('frame index')
        plt.ylabel('IOU loss')
        plt.title('membrane kinematic optimized iou loss')
        plt.show()
        return iou_loss
        
    def result_plot(self, json_name = "original"): 
        original_iou_loss = np.load(os.path.join(self.result_path_root, f"{self.test_name}_{self.start_pose}_{self.end_pose}_{json_name}.npy"))
        plt.plot(original_iou_loss)
        plt.xlabel('frame index')
        plt.ylabel('IOU loss')
        plt.title('orginal IOU loss')
        plt.show()
        return original_iou_loss
    
    def kinematic_plot_comparison(self): 
        bone_index = 6
        axis = 2
        axis_map = {0:"x", 1:"y",2:"z"}
        original_kinematic = []
        membrane_opt_kinematic = []
        for index in range(self.start_pose, self.end_pose): 
            original_output_path =os.path.join(self.camera_list_path_root,str(index), f"output.json")
            membrane_output_path = os.path.join(self.camera_list_path_root, str(index), f"membrane_output.json")
            original_output = read_json_file(original_output_path)
            membrane_output = read_json_file(membrane_output_path)
            original_kinematic.append(original_output['pose'][bone_index][axis])
            membrane_opt_kinematic.append(membrane_output['pose'][bone_index][axis])
        plt.plot(original_kinematic)
        plt.plot(membrane_opt_kinematic)
        plt.xlabel('frame index')
        plt.ylabel(f'ruler of bone: {bone_index}, axis: {axis_map[axis]}')
        plt.title('kinematic optimization')
        plt.legend(["original kinematic", "after optimization"])
        plt.show()
        
    def initialize_membrane_kinematic_training(self, epoch_number = 0): 
        """
        Purpose of this function is to initialize the membrane_output.json with output.json (raw) for each pose

        Returns
        -------
        None.

        """
        if(epoch_number == 0):
            
            for index in range(self.current_pose_index -self.half_window_size-1, self.current_pose_index+1): 
                source_json_path = os.path.join(self.kinematic_save_path_root, str(index), "output_smoothed.json")
                target_json_path = os.path.join(self.kinematic_save_path_root, str(index), f"membrane_output_{epoch_number}.json")
                source_json = read_json_file(source_json_path)
                save_json_file(source_json, target_json_path)
        else:
            for index in range(self.current_pose_index -self.half_window_size-1, self.current_pose_index+1): 
                source_json_path = os.path.join(self.kinematic_save_path_root, str(index), f"membrane_output_{epoch_number-1}.json")
                target_json_path = os.path.join(self.kinematic_save_path_root, str(index), f"membrane_output_{epoch_number}.json")
                source_json = read_json_file(source_json_path)
                save_json_file(source_json, target_json_path)
        print(f"membrane kinematic initialization done!!")
        return
    

    def original_kinematic_smooth_rendering(self, start_frame, end_frame): 
        """function rendering the reconstruction using original template and kinematic with default gaussian smoothing
            save the reconstruction in self.original_reconstruction_smooth_kinematic_path
        """
        name_parts = self.test_name.split('_')
        blender_test_name = self.test_name.replace("Brunei_2023_", "")
        #blender_test_name = f"{name_parts[-4]}_{name_parts[-3]}_{name_parts[-2]}_{name_parts[-1]}"
        #1. render the mesh optimized mesh
        cmd = []
        cmd.append("./blender")
        cmd.append(f"{PROJECT_ROOT}/PhD_research/3D_bat_reconstruction/SoftRas/models/membrane_kinematic_optimization_model/membrane_blender/{blender_test_name}/{blender_test_name}.blend")
        cmd.append("-b")  # run in backgroud
        cmd.append("--python-use-system-env")
        cmd.append("--python")
        cmd.append(f"{PROJECT_ROOT}/PhD_research/3D_bat_reconstruction/SoftRas/models/membrane_kinematic_optimization_model/blender_script_template.blend.py")
        cmd.append("--")
        cmd.append(f"{self.project_root_path}{self.project_name}")
        cmd.append(f"{blender_test_name}")
        cmd.append(f"{start_frame}")
        cmd.append(f"{end_frame}")
        cmd.append(f"{0.001}")  # random value, won't be used 
        cmd.append(f"{self.current_epoch}")
        cmd.append(f"False")
        print(cmd)
        _result = subprocess.run(cmd, capture_output=True,cwd="/home/yihao19/blender-4.5.4-linux-x64")
        return None
    
    def run_original_kinematic_smooth_rendering(self):
        for pose in range(self.start_pose, self.end_pose):
            self.original_kinematic_smooth_rendering(pose - self.half_window_size, pose)
        return
    
    def iou_loss_membrane_compare(self) -> None:
        """iou loss before and after membrane optimization
        """
        iou_loss_original_list = []
        iou_loss_membrane_opt_list = []
        for pose_index in tqdm(range(self.start_pose, self.end_pose, 1), desc="iou_loss cal..."):
            iou_loss = self.iou_loss_cal(pose_index,reconstruction_type = "kinematic_smooth")
            iou_loss_original_list.append(iou_loss)
            iou_loss = self.iou_loss_cal(pose_index, reconstruction_type="membrane_opt")
            iou_loss_membrane_opt_list.append(iou_loss)

        plt.plot(iou_loss_membrane_opt_list, color='black')
        plt.plot(iou_loss_original_list, color='black', linestyle=':')
        x = np.array([index for index in range(len(iou_loss_original_list))])
        iou_loss_original_list = np.array(iou_loss_original_list)
        iou_loss_membrane_opt_list = np.array(iou_loss_membrane_opt_list)
        plt.fill_between(x, iou_loss_original_list,iou_loss_membrane_opt_list, 
                 where=(iou_loss_original_list >= iou_loss_membrane_opt_list),
                 facecolor='green',
                 alpha=0.2,
                 interpolate=True)

        plt.fill_between(x, iou_loss_original_list, iou_loss_membrane_opt_list, 
                         where=(iou_loss_original_list <iou_loss_membrane_opt_list),
                         facecolor='red',
                         alpha=0.2,
                         interpolate=True
                         )
        plt.xlabel("Frame index")
        plt.ylabel("IOU loss")
        #plt.legend(["original","membrane optimized",])
        if not os.path.exists(f"./result_plot/{self.test_name}/"):
            os.makedirs(f"./result_plot/{self.test_name}/")
        plt.savefig(f"./result_plot/{self.test_name}/{self.test_name}_IOU_loss_original_VS_membrane_opt.svg",format="svg")
        plt.legend(["cloth-based membrane", "LBS"])
        plt.savefig(f"./result_plot/{self.test_name}/{self.test_name}_IOU_loss_original_VS_membrane_opt.svg",format="svg")
        plt.close()

        return
    
    def stiffness_visualization(self):
        iteration = 0
        attrib_list = []
        baseline_list = []
        for pose in range(self.start_pose, self.end_pose):
            temp_list = []
            baseline_temp_list=  []
            for counter in range(97,100):
                try:
                    attrib = read_json_file(f"{PROJECT_ROOT}/PhDProject_real_data/{self.test_name}/membrane_optimization_physical_attributes/{self.membrane_optimized_frame_str}_average/bayesian_attrib_opt_linear_{pose}_{iteration}_{counter}.json")
                    temp_list.append(attrib['physical_attributes'][0])
                except:
                    #temp_list.append(0)
                    break
            if(len(temp_list) == 0):
                break
            attrib_list.append(np.min(temp_list))
         
        
        plt.plot(attrib_list[:], color = 'black')
        plt.xlabel("Frame index")
        plt.ylabel("Tension stiffness")
        plt.ylim(0.0, 0.015)
        #plt.plot(no_cloth_baseline['loss_list'], color = 'black',linestyle = '--')
        #plt.plot(overall_iou_loss, color = 'black')
        if not os.path.exists(f"./result_plot/{self.test_name}/"):
            os.makedirs(f"./result_plot/{self.test_name}/")
        plt.savefig(f"./result_plot/{self.test_name}/{self.test_name}_{self.membrane_optimized_frame_str}_average_physical_attrib.svg",format="svg")
        plt.close()
        return
    
    def iou_loss_original(self, suffix:str=""):
        original_iou_loss_list = []
        original_raw_iou_loss_list = []
        for pose_index in tqdm(range(self.start_pose, self.end_pose, 1), desc="iou_loss cal..."):
            iou_loss = self.iou_loss_cal(pose_index, reconstruction_type="original")
            original_iou_loss_list.append(iou_loss)
            json_file = os.path.join(self.kinematic_save_path_root, str(pose_index), "output.json")
            file = open(json_file)
            data = json.load(file)
            original_raw_iou_loss_list.append(data['IOU'])

        plt.plot(original_iou_loss_list, color='black')
        plt.plot(original_raw_iou_loss_list, color='grey', linestyle=':')
        plt.xlabel("Frame index")
        plt.ylabel("IOU loss")
        plt.savefig(f"./result_plot/{self.test_name}{suffix}/{self.test_name}_IOU_original_wo_legend.svg",format="svg")
        plt.legend(["fixed template","size&shape varying template",])
        if not os.path.exists(f"./result_plot/{self.test_name}{suffix}/"):
            os.makedirs(f"./result_plot/{self.test_name}{suffix}/")
        plt.savefig(f"./result_plot/{self.test_name}{suffix}/{self.test_name}_IOU_original_w_legend.svg",format="svg")
        plt.close()
        return None
    def iou_loss_initial_vs_final(self): 
        original_iou_loss_list = []
        final_optimized_loss_list = []
        for pose_index in tqdm(range(self.start_pose, self.end_pose, 1), desc="iou_loss cal..."):
            iou_loss = self.iou_loss_cal(pose_index, reconstruction_type="original")
            original_iou_loss_list.append(iou_loss)
            iou_loss = self.iou_loss_cal(pose_index, reconstruction_type="kinematic_membrane_opt")
            final_optimized_loss_list.append(iou_loss)
        plt.plot(original_iou_loss_list, color='black',linestyle = ':')
        plt.plot(final_optimized_loss_list,color = 'black')
        x = np.array([index for index in range(len(original_iou_loss_list))])
        original_iou_loss_list = np.array(original_iou_loss_list)
        final_optimized_loss_list = np.array(final_optimized_loss_list)
        plt.fill_between(x, original_iou_loss_list,final_optimized_loss_list, 
                 where=(original_iou_loss_list >= final_optimized_loss_list),
                 facecolor='green',
                 alpha=0.2)

        plt.fill_between(x, np.array(original_iou_loss_list), np.array(final_optimized_loss_list), 
                         where=(original_iou_loss_list < final_optimized_loss_list),
                         facecolor='red',
                         alpha=0.2
                         )
        plt.xlabel("Frame index")
        plt.ylabel("IOU loss")
        #plt.legend(["original","membrane optimized",])
        if not os.path.exists(f"./result_plot/{self.test_name}/"):
            os.makedirs(f"./result_plot/{self.test_name}/")
        plt.savefig(f"./result_plot/{self.test_name}/{self.test_name}_IOU_initial_vs_final_wo_legend.svg",format="svg")
        plt.legend(["initial kinematic + LBS","updated kinematic + cloth-based membrane"])
        plt.savefig(f"./result_plot/{self.test_name}/{self.test_name}_IOU_initial_vs_final_w_legend.svg",format="svg")
        plt.close()

    def iou_loss_compare(self):
        original_iou_loss_list = []
        membrane_optimized_loss_list = []
        for pose_index in tqdm(range(self.start_pose, self.end_pose, 1), desc="iou_loss cal..."):
            iou_loss = self.iou_loss_cal(pose_index, reconstruction_type="original")
            original_iou_loss_list.append(iou_loss)
            iou_loss = self.iou_loss_cal(pose_index, reconstruction_type="membrane_opt")
            membrane_optimized_loss_list.append(iou_loss)
        plt.plot(original_iou_loss_list, color='black',linestyle = ':')
        plt.plot(membrane_optimized_loss_list,color = 'black')
        x = np.array([index for index in range(len(original_iou_loss_list))])
        original_iou_loss_list = np.array(original_iou_loss_list)
        membrane_optimized_loss_list = np.array(membrane_optimized_loss_list)
        plt.fill_between(x, original_iou_loss_list,membrane_optimized_loss_list, 
                 where=(original_iou_loss_list >= membrane_optimized_loss_list),
                 facecolor='green',
                 alpha=0.2)

        plt.fill_between(x, np.array(original_iou_loss_list), np.array(membrane_optimized_loss_list), 
                         where=(original_iou_loss_list < membrane_optimized_loss_list),
                         facecolor='red',
                         alpha=0.2
                         )
        plt.xlabel("Frame index")
        plt.ylabel("IOU loss")
        #plt.legend(["original","membrane optimized",])
        if not os.path.exists(f"./result_plot/{self.test_name}/"):
            os.makedirs(f"./result_plot/{self.test_name}/")
        plt.savefig(f"./result_plot/{self.test_name}/{self.test_name}_IOU_compare_wo_legend.svg",format="svg")
        plt.legend(["initial kinematic + LBS","initial kinematic + cloth-based membrane"])
        plt.savefig(f"./result_plot/{self.test_name}/{self.test_name}_IOU_compare_w_legend.svg",format="svg")
        plt.close()
        '''
        plt.plot(original_iou_loss_list, color='black',linestyle = ':')
        plt.plot(membrane_optimized_loss_list,color = 'black')
        plt.xlabel("Frame index")
        plt.ylabel("IOU loss")
        plt.legend(["original","membrane optimized",])
        plt.savefig(f"./result_plot/{self.test_name}_IOU_compare.svg",format="svg")
        '''
    def scale_parameter_plot(self)-> None:
        scale_parameter_list = []
        for pose_index in tqdm(range(self.start_pose, self.end_pose, 1), desc="iou_loss cal..."):
            json_file = os.path.join(self.kinematic_save_path_root, str(pose_index), "output.json")
            file = open(json_file)
            data = json.load(file)
            scale_parameter_list.append(data['scale'])
        average = np.mean(np.array(scale_parameter_list))
        plt.xlabel("Frame index")
        plt.ylabel("Template scale factor")
        plt.plot(scale_parameter_list, color='black')
        plt.plot([average for _counter in range(len(scale_parameter_list))], color='black', linestyle=":")
        plt.savefig(f"./result_plot/{self.test_name}/{self.test_name}_scale_parameter_wo_legend.svg",format="svg")
        plt.legend(["template scale parameter", "average"])
        plt.savefig(f"./result_plot/{self.test_name}/{self.test_name}_scale_parameter_w_legend.svg",format="svg")
        plt.close()
        return
    
    def iou_loss_w_wo_membrane(self):
        iou_loss_list_prev = []
        iou_loss_list_wo_prev = []
        iou_loss_list_original = []
        for pose_index in tqdm(range(self.start_pose, self.end_pose, 1), desc="iou_loss cal..."):
            iou_loss = self.iou_loss_cal(pose_index,original=False,use_prev=True)
            iou_loss_list_prev.append(iou_loss)
            iou_loss = self.iou_loss_cal(pose_index, original=False,use_prev=False)
            iou_loss_list_wo_prev.append(iou_loss)
            iou_loss = self.iou_loss_cal(pose_index, original=True)
            iou_loss_list_original.append(iou_loss)
        plt.plot(iou_loss_list_prev, color='black')
        plt.plot(iou_loss_list_wo_prev, color='black', linestyle=':')
        plt.plot(iou_loss_list_original, color='grey')
        plt.xlabel("Frame index")
        plt.ylabel("IOU loss")
        #plt.legend(["original","membrane optimized",])
        if not os.path.exists(f"./result_plot/{self.test_name}/"):
            os.makedirs(f"./result_plot/{self.test_name}/")
        plt.savefig(f"./result_plot/{self.test_name}/{self.test_name}_IOU_loss_membrane_opt_w_prev.svg",format="svg")
        plt.legend(["with prev","without prev","no membrane opt"])
        plt.savefig(f"./result_plot/{self.test_name}/{self.test_name}_IOU_loss_membrane_opt_w_prev.svg",format="svg")
        plt.close()

    def plot_initial_kinematic(self, bone_index = [0,6,19], kinematic_smoothed:bool=False, suffix:str=""):
        output_path = os.path.join(self.camera_list_path_root)
        if not os.path.exists(f"./result_plot/{self.test_name}{suffix}/"):
            os.makedirs(f"./result_plot/{self.test_name}{suffix}/")
        output_jsons_x = []
        output_jsons_y = []
        output_jsons_z = []
        displacement_list = []

        for counter in range(self.start_pose, self.end_pose): 
            if(kinematic_smoothed == True):
                json_path = os.path.join(output_path, str(counter), "output_smoothed.json")
            else: 
                json_path = os.path.join(output_path, str(counter), "output.json")
            output_json_x,output_json_y, output_json_z, displacement = read_pose_json(json_path, bone_index)
            output_jsons_x.append(output_json_x)
            output_jsons_y.append(output_json_y)
            output_jsons_z.append(output_json_z)
            displacement_list.append(displacement)
            

        output_json_x_array = np.array(output_jsons_x)
        output_json_y_array = np.array(output_jsons_y)
        output_json_z_array = np.array(output_jsons_z)
        displacement_array = np.array(displacement_list)
        if(kinematic_smoothed == False):
            prefix = "Initial"
        else: 
            prefix = "Smoothed_initial"
        fig = plt.figure()    
        plt.title("X axis")
        plt.xlabel("Frame index")
        plt.ylabel("Rotation in radiant")
        plt.plot(output_json_x_array[:, 1], color = "black", linestyle=':')
        plt.plot(-output_json_x_array[:, 2], color = "black")
        plt.legend(["Bone: {}".format(bone_index[1]), "Bone: {}".format(bone_index[2])])
        plt.axhline(y = 0, color = 'black', linestyle = '--') 
        plt.ylim(-1, 1)

        plt.savefig(f"./result_plot/{self.test_name}{suffix}/{prefix}_Kinematic_X.svg")
        plt.close()
        fig = plt.figure()
        plt.title("Y axis")
        #plt.plot(output_json_array[:, 0])
        plt.xlabel("Frame index")
        plt.ylabel("Rotation in radiant")

        plt.plot(output_json_y_array[:, 1], color = "black", linestyle=':')
        plt.plot(output_json_y_array[:, 2], color = "black")
        plt.legend(["Bone: {}".format(bone_index[1]), "Bone: {}".format(bone_index[2])])
        plt.axhline(y = 0, color = 'black', linestyle = '--') 
        plt.ylim(-1, 1)

        plt.savefig(f"./result_plot/{self.test_name}{suffix}/{prefix}_Kinematic_Y.svg")
        plt.close()

        fig = plt.figure()
        plt.title("Z axis")
        #plt.plot(output_json_array[:, 0])
        plt.xlabel("Frame index")
        plt.ylabel("Rotation in radiant")

        plt.plot(output_json_z_array[:, 1], color = "black", linestyle=':')
        plt.plot(output_json_z_array[:, 2], color = "black")
        plt.legend(["Bone: {}".format(bone_index[1]), "Bone: {}".format(bone_index[2])])
        plt.axhline(y = 0, color = 'black', linestyle = '--')
        plt.ylim(-1, 1)
        plt.savefig(f"./result_plot/{self.test_name}{suffix}/{prefix}_Kinematic_Z.svg")
        plt.close()

        fig = plt.figure()
        plt.title("Displacement X axis")
        plt.plot(displacement_array[:, 0], color = "black")
        plt.savefig(f"./result_plot/{self.test_name}{suffix}/{prefix}_displacement_X.svg")
        plt.close()

        fig = plt.figure()
        plt.title("Displacement Y axis")
        plt.plot(displacement_array[:, 1], color = "black")
        plt.savefig(f"./result_plot/{self.test_name}{suffix}/{prefix}_displacement_Y.svg")
        plt.close()

        fig = plt.figure()
        plt.title("Displacement Z axis")
        plt.plot(displacement_array[:, 2], color = "black")
        plt.savefig(f"./result_plot/{self.test_name}{suffix}/{prefix}_displacement_Z.svg")
        plt.close()
        return
    
    def calibration_validation(self, pose_index:int, threshold:int = 5) -> None: 
        """
        function to generate a small cube around the pose to validate the projection and select the "most" calibrated camera list. 
        Function will return the projection of all the calibration
        :param self: Description
        :param pose_index: Description
        :type pose_index: int
        """
        pose_json_path = os.path.join(self.kinematic_save_path_root, str(pose_index), "output.json")
        file = open(pose_json_path)
        pose_json = json.load(file)
        center = pose_json['template_displacement']
        sample_point = sample_sphere_volume(center=center)
        # turn it into homogeneous
        camera_matrix = np.loadtxt(os.path.join(self.camera_meta_path_root, "camera_meta.txt"))
        camera_list_txt_path = os.path.join(self.camera_list_path_root, str(pose_index), "camera.txt")
        with open(camera_list_txt_path) as f:
            text = f.read()
            data = json.loads(text)
        camera_list = str(data)[1:-1].split(", ")
        camera_number = camera_matrix.shape[0]
        camera_matrix = np.reshape(camera_matrix, (camera_number, 3, 4))
        candidate = []
        for point in tqdm(sample_point, desc="projecting points to images..."): 
            keeps = 0
            for _index, camera_index in enumerate(camera_list):
                image_path = os.path.join(self.camera_list_path_root, str(pose_index),f"camera{camera_index}.png")
                image =cv2.imread(image_path, cv2.COLOR_RGB2GRAY)
                matrix = camera_matrix[int(camera_index) - 1]
                keep = if_keep_via_projection(point, image, matrix)
                if(keep == 1): 
                    keeps +=1
            if(keeps >= threshold):
                candidate.append(point)
        # projects all the points onto all the images
        for _index, camera_index in enumerate(camera_list):
            image_path = os.path.join(self.camera_list_path_root, str(pose_index),f"camera{camera_index}.png")
            image =cv2.imread(image_path, cv2.COLOR_RGB2GRAY)
            matrix = camera_matrix[int(camera_index) - 1]
            image = point_to_image(candidate, image, matrix)
            # write it into calibration check folder
            save_path_root = os.path.join(self.calibration_check_path, str(pose_index))
            if not os.path.exists(save_path_root):
                os.makedirs(save_path_root)
            save_path = os.path.join(save_path_root, f"camera{camera_index}.png")
            cv2.imwrite(save_path, image)
        return
    
    def run_original_reconstruction(self):
        for pose_index in tqdm(range(self.start_pose, self.end_pose, 1), desc=f"original reconstruction: {self.test_name}"): 
            self.current_pose_index = pose_index
            self.original_reconstruction(self.current_pose_index)
        return None
    
    def generate_flying_trajectory_gif(self, if_smoothed:bool=False, suffix:str=""):
        """
        generate the gif that contains the flying trajectory of the point cloud in bev view
        """
        if not os.path.exists(f"./result_plot/{self.test_name}{suffix}/"):
            os.makedirs(f"./result_plot/{self.test_name}{suffix}/")
        if(if_smoothed == True):
            prefix = "smoothed"
            kinematic_file_name = "output_smoothed.json"
        else: 
            prefix = "initial"
            kinematic_file_name = "output.json"
        gif_output_path = os.path.join(f"./result_plot/{self.test_name}{suffix}/{self.test_name}_{prefix}_trajectory.gif")
        gif_writter = imageio.get_writer(gif_output_path, loop=0, fps=40)
        # get the first reconstruction pose and function as reference
        first_pose_json_path =  os.path.join(self.kinematic_save_path_root, str(self.start_pose), kinematic_file_name)
        file = open(first_pose_json_path)
        first_pose_json = json.load(file)
        
        root_rotation_euler = first_pose_json['pose'][0]
        root_displacement = first_pose_json['template_displacement']
        root_rotation_matrix = quat_to_rotmat(rodrigues(root_rotation_euler))
        root_inv_rotation_matrix = np.linalg.inv(root_rotation_matrix)
        #root_inv_rotation_matrix = np.eye(3)
        
        for pose_index in tqdm(range(self.start_pose, self.end_pose),desc="composing viz video..."):
            pose_json_path = os.path.join(self.kinematic_save_path_root, str(pose_index),kinematic_file_name)
            file = open(pose_json_path)
            pose_json = json.load(file)
            rotation_euler = pose_json['pose'][0]
            displacement = pose_json['template_displacement']
            rotation_matrix = quat_to_rotmat(rodrigues(rotation_euler))
            rect_rotation_matrix =  root_inv_rotation_matrix @ rotation_matrix
            rect_rotation_euler = rotmat_to_euler(rect_rotation_matrix)
            rect_displacement = (np.array(displacement) - np.array(root_displacement)).tolist()
            pose_json['template_displacement'] = [0,0,0]
            pose_json['pose'][0] = rect_rotation_euler

            kinematic_model = Kinematic_model(bone_skining_matrix_name=self.model_template,
                                          opposite_direction=self.opposite_direction).cuda()

            estimated_location = np.array([pose_json['template_displacement']]).astype('float32')
            pose = pose_json['pose']
            if(len(pose) == 34):
                # acamodate the new template with 40 bone
                pose.append([0,0,0])
                pose.append([0,0,0])
                pose.append([0,0,0])
                pose.append([0,0,0])
                pose.append([0,0,0])
                pose.append([0,0,0])

            pose = np.array(pose).astype('float32')
    
            estimated_location = torch.tensor(estimated_location).cuda()
            pose = torch.tensor(pose).cuda()
            output_mesh = kinematic_model.render_original(estimated_location, pose)
            vertices = output_mesh.vertices.cpu().numpy()[0]
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d') # or use plt.axes(projection='3d')

            # 3. Create the 3D plot
           
            x = vertices[:, 0]
            y = vertices[:, 1]
            z = vertices[:, 2]
            ax.scatter(x, y, z, c='b', marker='o',s=0.01) # 'c' for color, 'marker' for style
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim([-0.05, 0.05])
            ax.set_ylim([-0.05, 0.05])
            ax.set_zlim([-0.05, 0.05])
            ax.set_title(f"{self.test_name}: {pose_index}")
            ax.view_init(elev=80, azim=-60)
            fig.savefig(f"{self.test_name}_{pose_index}.png")
            plt.close()
            plot_img = cv2.imread(f"{self.test_name}_{pose_index}.png")
            gif_writter.append_data(plot_img)
            os.remove(f"{self.test_name}_{pose_index}.png")
           
        gif_writter.close()
        return
    
    def run_kinematic_smoothing(self, sigma:float=2):
        

        rotation_matrix_list = []
        displacement_vector_list = []
        for pose_index in tqdm(range(self.start_pose, self.end_pose), desc="reading initial kinematics..."):
            pose_json_path = os.path.join(self.kinematic_save_path_root, str(pose_index), "output.json")
            file = open(pose_json_path)
            pose_json = json.load(file)
            rotation_matrix_list.append(np.array(pose_json['pose']))
            displacement_vector_list.append(np.array(pose_json['template_displacement']))
        smoothed_rotation_euler_list = kinematic_smoothing(rotation_matrix_list, sigma=sigma)
        smoothed_vector_list = displacement_smoothing(displacement_vector_list, sigma=sigma)

        for pose_index in tqdm(range(self.start_pose, self.end_pose), desc="writting smoothed kinematic back..."):
            
            pose_json_path = os.path.join(self.kinematic_save_path_root, str(pose_index), "output.json")
            smooth_json_path = os.path.join(self.kinematic_save_path_root, str(pose_index), "output_smoothed.json")
            index = pose_index - self.start_pose
            file = open(pose_json_path)
            pose_json = json.load(file)
            pose_json["pose"] = smoothed_rotation_euler_list[index].tolist()
            pose_json["template_displacement"] = smoothed_vector_list[index].tolist()
            with open(smooth_json_path, 'w') as file: 
                json.dump(pose_json,file, indent=4)
        return
    #three major pipelines

    def run_raw_kinematic_optimize_pipeline(self):
        reverse = False
        step = 1
        if(self.end_pose <= self.start_pose):
            # reconstruct from last frame to start_frame
            reverse = True
            step = -1
            
        for pose_index in range(self.start_pose, self.end_pose, step): 
            self.current_pose_index = pose_index
            self.kinematic_optimize(pose_index, reverse = reverse)
        return None
    
    def run_membrane_optimize_pipeline(self, epoch_index): 
        # initialize the membrane output.json with the original output.json
        for pose_index in range(self.start_pose, self.end_pose, 1): 
            self.current_pose_index = pose_index
            self.current_epoch = epoch_index
            self.initialize_membrane_kinematic_training(epoch_number = self.current_epoch)
            # optimizing the membrane with kinematic fixed
            result = self.membrane_optimize_bayesian(epoch_number = self.current_epoch)
        
        return None
    
    def run_membrane_kinematic_update_pipeline(self, epoch_index):
        for pose_index in range(self.start_pose, self.end_pose, 1): 
            self.current_pose_index = pose_index
            self.current_epoch = epoch_index
            self.membrane_kinematic_optimize(pose_index, pipeline_epoch=self.current_epoch)
        
        return
    
  
    
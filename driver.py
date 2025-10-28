# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:52:28 2025

@author: yihao
"""
import os
from image_data_loader import image_dataloader
from membrane_kinematic_model import Membrane_kinematic_model
from kinematic_model import Kinematic_model
from utils.general_utils import neg_iou_loss, item_transform, read_json_file,save_json_file
from utils.blender_utils import update_simulations, get_target_objs, y_forward_z_up
from genetic_algorithm import initialize_population, mutate,crossover
import torch
from skopt import gp_minimize, dump,load
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from skopt import Optimizer
from torch.utils import data
import threading
import torch.multiprocessing as mp
import soft_renderer as sr
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from blendtorch import btt
import random
import matplotlib.pyplot as plt
from consecutive_frame_search_space import searching_space_angular, searching_space_linear
from matplotlib.ticker import MaxNLocator
from skopt.space import Real, Integer
from scipy.ndimage import gaussian_filter1d
import subprocess
from num2words import num2words
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
                 if_use_previous:bool, 
                 opposite_direction:bool):
        self.project_root_path = project_root_path
        self.project_name = project_name
        self.test_name = test_name
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.current_pose_index = current_pose_index
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
        
        self.original_mesh_save_path_root = os.path.join(self.project_root_path, 
                                                self.project_name, 
                                                self.test_name, 
                                                "original_reconstruction")
        
        if not os.path.exists(self.original_mesh_save_path_root):
            os.makedirs(self.original_mesh_save_path_root)
            
        self.membrane_optimize_mesh_save_path_root = os.path.join(self.project_root_path, 
                                                self.project_name, 
                                                self.test_name, 
                                                "membrane_optimized_mesh")
        
        if not os.path.exists(self.membrane_optimize_mesh_save_path_root):
            os.makedirs(self.membrane_optimize_mesh_save_path_root)
            
        self.membrane_kinematic_optimized_mesh_save_root =  os.path.join(self.project_root_path, 
                                                self.project_name, 
                                                self.test_name, 
                                                "membrane_kinematic_optimized_mesh_final")
        
        if not os.path.exists(self.membrane_kinematic_optimized_mesh_save_root):
            os.makedirs(self.membrane_kinematic_optimized_mesh_save_root)
        
        folder_index = num2words(self.membrane_optimized_frame).lower()
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
        self.result_path_root = './result'
        self.use_previous = if_use_previous
        self.opposite_direction = opposite_direction
        self.image_size = (1024,1280)
        self.SIM_INSTANCES = 1
        self.membrane_parameter = {
            
            }
    def kinematic_optimize(self, pose_index, use_previous = False): 
        """
        Parameters
        ----------
        pose_index : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        """
       
        kinematic_model = Kinematic_model(bone_skining_matrix_name=self.model_template, opposite_direction=self.opposite_direction).cuda()
        optimizer = torch.optim.Adam(kinematic_model.parameters(), 0.005,betas=(0.5, 0.99))
        train_dataloader, batch_size = image_dataloader(
            self.camera_meta_path_root, 
            self.camera_list_path_root, 
            self.silhouette_image_path_root, 
            pose_index, 
            self.use_previous
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
                    l2_adjust = 0.02 * torch.norm(local_adjust - prev_local_adjust)
            
                loss = IOU_loss + pose_loss  + 0 * laplacian_loss + 0.00 * wing_tip_reg  + 0.005 * bone_prior + 0.005 * bone_symmetric + 0*l2_adjust
                epoch.set_description('IOU Loss: %.4f   Pose Loss: %.4f  Wingtip_reg: %.4f  Bone prior: %.4f  Bone symmetry: %.4f  L2 adjust: %.4f' % (IOU_loss.item(),pose_loss.item(), wing_tip_reg.item(), 0.1 * bone_prior.item(), bone_symmetric.item(), l2_adjust.item()))
                
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
        output_mesh.save_obj(os.path.join(self.membrane_optimize_mesh_save_path_root, '{}_bat_{}.obj'.format(test_name, pose_index)), save_texture=False)

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
    
    def iou_loss_cal(self, pose_index:int, total_iou_loss_list) -> None:
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
        print(f"IOU loss: {iou_loss.item()}") 
        total_iou_loss_list.append(iou_loss.item())
        return None
        
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
        blender_test_name = f"{name_parts[-4]}_{name_parts[-3]}_{name_parts[-2]}_{name_parts[-1]}"
        #1. render the mesh optimized mesh
        cmd = []
        cmd.append("blender")
        cmd.append(f"./membrane_blender/{blender_test_name}/{blender_test_name}.blend")
        cmd.append("-b")  # run in backgroud
        cmd.append("--python-use-system-env")
        cmd.append("--python")
        cmd.append("blender_script_template.blend.py")
        cmd.append("--")
        #
        cmd.append(f"{self.project_root_path}{self.project_name}")
        cmd.append(f"{blender_test_name}")
        cmd.append(f"{self.current_pose_index-self.half_window_size}")
        cmd.append(f"{self.current_pose_index}")
        cmd.append(f"{membrane_physical_attribues[0]}")
        result = subprocess.run(cmd, capture_output=True)
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
        if(not self.use_previous):
            #print("max_loss: ", max_loss, "   reg_term: ", ref_coef * membrane_physical_attribues[0]**2 )
            #print("mean_loss: ", average_iou_loss, "   reg_term: ", ref_coef * membrane_physical_attribues[0]**2)
            return average_iou_loss + ref_coef * membrane_physical_attribues[0]**2  # add it as regularization
        else: 
            # if using the previous read the tension attributes of the previous frame
            prev_attribute_list = []
            for counter in range(self.membrane_opt_epoch - 2, self.membrane_opt_epoch):
                prev_attributes_path =  os.path.join(self.membrane_optimize_attributes_save_path_root, f"bayesian_attrib_opt_linear_{self.current_pose_index-1}_{self.current_epoch}_{counter}.json")
                attrib = read_json_file(prev_attributes_path)
                prev_attribute_list.append(attrib['physical_attributes'][0])
            prev_tension = np.min(prev_attribute_list)
            #print("max_loss: ", max_loss, "   reg_term: ", ref_coef * (membrane_physical_attribues[0] - prev_tension)**2 )
            #print("mean_loss: ", average_iou_loss, "   reg_term: ", ref_coef * (membrane_physical_attribues[0] - prev_tension)**2 )
            #print("prev pose: ", self.current_pose_index-1 )
            return average_iou_loss + ref_coef * (membrane_physical_attribues[0] - prev_tension)**2 +  pre_coef * (membrane_physical_attribues[0])**2 # add it as regularization
        
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
       
        gp = GaussianProcessRegressor(kernel=Matern(length_scale=500), 
                              alpha=1e-6, normalize_y=True, random_state=0)
        tqdm_callback = TqdmCallBack(total_iterations=self.membrane_opt_epoch, description=f"membrane optimizing pose: {self.current_pose_index}")
        result = gp_minimize(self.membrane_optimization_loss_bayes_mode_v2,     # The function to minimize
                             searching_space,                   # The search space
                             n_calls= self.membrane_opt_epoch,              # The number of evaluations
                             random_state=42,
                             initial_point_generator='lhs',
                             n_initial_points=int(0.5 * self.membrane_opt_epoch),
                             acq_optimizer="lbfgs",
                             kappa = 0.19,
                             acq_func = "LCB",
                             noise=1e-6,
                             base_estimator = gp,
                             callback=[tqdm_callback])
        tqdm_callback.close()         # R
        return result
        
        
    def membrane_kinematic_optimize(self,pose_index, pipeline_epoch=0): 
        membrane_modified_obj_path = os.path.join(self.membrane_optimize_mesh_save_path_root, f"{pose_index}.obj")
        pose_original_kinematic_path = os.path.join(self.kinematic_save_path_root, f"{pose_index}", "membrane_output.json")
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
            
                loss = IOU_loss + pose_loss  + 0 * laplacian_loss + 0.00 * wing_tip_reg  + 0.005 * bone_prior + 0.005 * bone_symmetric + 0*l2_adjust
                epoch.set_description('IOU Loss: %.4f   Pose Loss: %.4f  Wingtip_reg: %.4f  Bone prior: %.4f  Bone symmetry: %.4f  L2 adjust: %.4f' % (IOU_loss.item(),pose_loss.item(), wing_tip_reg.item(), 0.1 * bone_prior.item(), bone_symmetric.item(), l2_adjust.item()))
                
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
        save_file_path = os.path.join(self.kinematic_save_path_root,str(pose_index) ,"membrane_output.json")
        save_json_file(save_dict, save_file_path)
        # save another copy for tracking
        if(pose_index == self.current_pose_index):
            output_mesh.save_obj(os.path.join(self.membrane_kinematic_optimized_mesh_save_root, '{}_bat_{}.obj'.format(test_name, pose_index)), save_texture=False)
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

    def run_membrane_kinematic_optimize_pipeline(self): 
        # initialize the membrane output.json with the original output.json
        self.initialize_membrane_kinematic_training()
        for pipeline_epoch in range(self.whole_opt_epoch): 
            # optimizing the membrane with kinematic fixed
            result = self.membrane_optimize_bayesian(pipeline_epoch)
            
            # save the result
            #dump(result,"./gp_minimize.pkl")
            
            #self.current_epoch = pipeline_epoch
            #self.membrane_optimization_loss_bayes_mode([0,0,0,0,0,0])
            # save the iou loss for performance tracking
            
            # optimizing the kinematic with membrane fixed
            self.sequence_membrane_kinematic_optimize(pipeline_epoch)
            # calculate the IOU loss
        return
    
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
    def initialize_membrane_kinematic_training(self): 
        """
        Purpose of this function is to initialize the membrane_output.json with output.json (raw) for each pose

        Returns
        -------
        None.

        """
        for index in range(self.current_pose_index -self.half_window_size, self.current_pose_index+1): 
            source_json_path = os.path.join(self.kinematic_save_path_root, str(index), "output.json")
            target_json_path = os.path.join(self.kinematic_save_path_root, str(index), "membrane_output.json")
            source_json = read_json_file(source_json_path)
            save_json_file(source_json, target_json_path)
        print(f"membrane kinematic initialization done!!")
        return
if __name__=="__main__":
    
    print("membrane kinematic optimizing driver function")
    project_root_path = "D:/"
    project_name = "PhDProject_real_data"
    test_name = "brunei_2023_bat_test_13_2"
    membrane_simulation_mode = "ANGULAR"
    start_pose = 10
    end_pose =100
    current_pose_index = start_pose
    half_window_size = 8  # animation rendering window size
    membrane_optimized_frame = 1# frame number that will be optimized
    kinematic_opt_epoch = 300
    membrane_opt_epoch =100
    membrane_kinematic_opt_epoch = 1
    whole_opt_epoch =1
    if_use_previous = False
    opposite_direction = False # bat flying direction
    model_template_name = "new_bat_params_version2_backward_membrane_24.pkl"
    driver = Optimize_Driver(project_root_path, 
                             project_name, 
                             test_name, 
                             start_pose, 
                             end_pose, 
                             current_pose_index, 
                             half_window_size,
                             membrane_optimized_frame,
                             kinematic_opt_epoch, 
                             membrane_opt_epoch, 
                             membrane_kinematic_opt_epoch,
                             whole_opt_epoch, 
                             model_template_name,
                             if_use_previous, 
                             opposite_direction)
  
    for pose_index in range(start_pose, end_pose, 1): 
        driver.current_pose_index = pose_index
        driver.run_membrane_kinematic_optimize_pipeline()
    
    

    #driver.run_membrane_kinematic_optimize_pipeline()

    final_iou_loss = []
    overall_iou_loss = []
    original_iou_loss = []
    attrib_list = []
    '''
    for pose in range(650, 750): 
        json_output = read_json_file(f"D:/PhDProject_real_data/brunei_2023_bat_test_13_2/rearrange_pose/{pose}/membrane_output.json")
        overall_iou_loss.append(json_output['IOU'])
    '''
    iteration = 0
    mean_list = []
    std_list = []
    indexes = [index + 1 for index in range(3)]
    index = 201
    for pose in range(0, 100):
        '''
        if(pose < 250): 
            final_iou_loss.append(None)
            continue
        '''
        index = 15
        try:
            attrib = read_json_file(f"D:/PhDProject_real_data/brunei_2023_bat_test_13_2/membrane_optimization_physical_attributes/three_average/bayesian_attrib_opt_linear_{index}_{iteration}_{pose}.json")
            attrib_list.append(attrib['physical_attributes'][0])
        except:
            break
        #json_output = read_json_file(f"D:/PhDProject_real_data/brunei_2023_bat_test_13_2/rearrange_pose/{pose}/membrane_output_{iteration}.json")
        #final_iou_loss.append(json_output['IOU'])
        
        #json_output = read_json_file(f"D:/PhDProject_real_data/brunei_2023_bat_test_13_2/rearrange_pose/{pose}/membrane_output_no_membrane0.json")
        #original_iou_loss.append(json_output['IOU'])
    #mean_list.append(np.mean(final_iou_loss))
    #std_list.append(np.std(final_iou_loss))
    plt.plot(attrib_list[:], color = 'black')
    #plt.plot(no_cloth_baseline['loss_list'], color = 'black',linestyle = '--')
    #plt.plot(overall_iou_loss, color = 'black')
    plt.savefig("physical_attrib.svg",format="svg")
    plt.show()
    '''
    #no_cloth_baseline_json = f"D:/PhD_research/3D_bat_reconstruction/SoftRas/models/membrane_kinematic_optimization_model/{test_name}_no_cloth_baseline_{start_pose - half_window_size}_{start_pose + half_window_size}.json"
    #no_cloth_baseline = read_json_file(no_cloth_baseline_json)
    plt.plot(final_iou_loss[:], color = 'black')
    plt.plot(original_iou_loss[:], color = 'black',linestyle = '--')
    #plt.legend(["membrane+kinematic optimized", "Original"])
    #plt.title(f"IOU loss after iteration: {iteration}")
    #plt.ylabel("IOU loss")
    #plt.xlabel("frame index")
    plt.ylim(0.18, 0.3)
    #plt.plot(no_cloth_baseline['loss_list'], color = 'black',linestyle = '--')
    #plt.plot(overall_iou_loss, color = 'black')
    plt.savefig(f"iteration_{iteration}.svg",format="svg")
    plt.show()
    '''
    attrib_list = []
    baseline_list = []
    for pose in range(10, 100):
        '''
        if(pose < 250): 
            final_iou_loss.append(None)
            continue
        '''
        temp_list = []
        baseline_temp_list=  []
        for counter in range(95,100):
            try:
                attrib = read_json_file(f"D:/PhDProject_real_data/brunei_2023_bat_test_13_2/membrane_optimization_physical_attributes/five_average/bayesian_attrib_opt_linear_{pose}_{iteration}_{counter}.json")
                temp_list.append(attrib['physical_attributes'][0])
                attrib = read_json_file(f"D:/PhDProject_real_data/brunei_2023_bat_test_13_2/membrane_optimization_physical_attributes/three_average_lc_10_5050/bayesian_attrib_opt_linear_{pose}_{iteration}_{counter}.json")
                baseline_temp_list.append(attrib['physical_attributes'][0])
            except:
                #temp_list.append(0)
                break
        
        
        if(len(temp_list) == 0): 
            pass
        else:
            attrib_list.append(np.min(temp_list))
        if(len(baseline_temp_list) == 0): 
            pass
        else:
            baseline_list.append(np.min(baseline_temp_list))
            
        
        
        #json_output = read_json_file(f"D:/PhDProject_real_data/brunei_2023_bat_test_13_2/rearrange_pose/{pose}/membrane_output_{iteration}.json")
        #final_iou_loss.append(json_output['IOU'])
        
        #json_output = read_json_file(f"D:/PhDProject_real_data/brunei_2023_bat_test_13_2/rearrange_pose/{pose}/membrane_output_no_membrane0.json")
        #original_iou_loss.append(json_output['IOU'])
    #mean_list.append(np.mean(final_iou_loss))
    #std_list.append(np.std(final_iou_loss))
    #attrib_list = gaussian_filter1d(attrib_list[:], sigma=1)
    '''
    while(len(attrib_list) < 175):
        attrib_list.append(0)
    
    while(len(baseline_list) < 200):
        baseline_list.append(0)
    '''
    plt.plot(attrib_list[:], color = 'black')
    plt.plot(baseline_list[:], color = 'grey',linestyle='--')
    plt.xlabel("Frame index")
    plt.ylabel("Tension stiffness")
    plt.ylim(0.0, 0.01)
    #plt.plot(no_cloth_baseline['loss_list'], color = 'black',linestyle = '--')
    #plt.plot(overall_iou_loss, color = 'black')
    plt.savefig("physical_attrib.svg",format="svg")
    plt.show()
   
    
  
    
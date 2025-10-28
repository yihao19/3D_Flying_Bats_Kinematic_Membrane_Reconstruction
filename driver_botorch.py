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
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
from torch.utils import data
import soft_renderer as sr
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
from blendtorch import btt
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from consecutive_frame_search_space import searching_space, searching_space_bnn
import math
import time
import GPyOpt

class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
PASSED_POSE_INDEX = 200
global interation
interation = 0
class Optimize_Driver(): 
    def __init__(self, 
                 project_root_path:str,
                 project_name:str, 
                 test_name:str, 
                 start_pose:int, 
                 end_pose:int, 
                 render_start_pose:int, 
                 render_end_pose:int,
                 kinematic_opt_epoch:int, 
                 membrane_opt_epoch:int, 
                 membrane_kinematic_opt_epoch:int,
                 whole_opt_epoch:int, 
                 render_buffer:int, 
                 model_template:str, 
                 if_use_previous:bool, 
                 opposite_direction:bool):
        self.project_root_path = project_root_path
        self.project_name = project_name
        self.test_name = test_name
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.render_start_pose = render_start_pose
        self.render_end_pose = render_end_pose
        self.kinematic_opt_epoch = kinematic_opt_epoch
        self.membrane_opt_epoch = membrane_opt_epoch
        self.whole_opt_epoch = whole_opt_epoch
        self.render_buffer = render_buffer
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
        self.membrane_optimize_mesh_save_path_root = os.path.join(self.project_root_path, 
                                                self.project_name, 
                                                self.test_name, 
                                                "membrane_optimized_mesh")
        self.membrane_kinematic_optimized_mesh_save_root =  os.path.join(self.project_root_path, 
                                                self.project_name, 
                                                self.test_name, 
                                                "membrane_kinematic_optimized_mesh")
        self.membrane_optimize_attributes_save_path_root =  os.path.join(self.project_root_path, 
                                                self.project_name, 
                                                self.test_name, 
                                                "membrane_optimization_physical_attributes")
        self.result_path_root = './result'
        self.use_previous = if_use_previous
        self.opposite_direction = opposite_direction
        self.image_size = (1024,1280)
        self.SIM_INSTANCES = 1
        self.membrane_parameter = {
            
            }
        self.current_pose = 0
        self.iteration = 0
    def kinematic_update(self, pose_index, membrane_attributes): 
        """
        this function will make modified the existing kinematics for pose_index and pose_index + 1

        Parameters
        ----------
        pose_index : TYPE
            DESCRIPTION.
        membrane_attributes : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        
        """
        def check_all_zeros(lst):
            return all(x == 0 for x in lst)
        first_json_path = os.path.join(self.kinematic_save_path_root, str(pose_index), "membrane_kinematic_bayes_output_original.json")
        #second_json_path = os.path.join(self.kinematic_save_path_root, str(pose_index + 1), "membrane_kinematic_bayes_output_original.json")
        first_json_target_path = os.path.join(self.kinematic_save_path_root, str(pose_index), "membrane_kinematic_bayes_output_optimizing.json")
        #second_json_target_path = os.path.join(self.kinematic_save_path_root, str(pose_index+1), "membrane_kinematic_bayes_output_optimizing.json")
        first_kinematic = read_json_file(first_json_path)
        #second_kinematic = read_json_file(second_json_path)
        
        # modify the first and second json based on the membrane_attributes
 
        first_pose = first_kinematic['pose']
        for joint_index, rotation in enumerate(first_pose): 
            if(check_all_zeros(rotation) == True): 
                continue
            else: 
                if(rotation[0] != 0): 
                    rotation[0] = rotation[0] + membrane_attributes[3*joint_index]
                if(rotation[1] != 0): 
                    rotation[1] = rotation[1] + membrane_attributes[3*joint_index + 1]
                if(rotation[2] != 0):
                    rotation[2] = rotation[2] + membrane_attributes[3*joint_index + 2]
                
        
        # add the displacement adjustmetn
        first_kinematic['template_displacement'][0] = first_kinematic['template_displacement'][0] + membrane_attributes[108]
        first_kinematic['template_displacement'][1] = first_kinematic['template_displacement'][1] + membrane_attributes[109]
        first_kinematic['template_displacement'][2] = first_kinematic['template_displacement'][2] + membrane_attributes[110]
        
        
        save_json_file(first_kinematic, first_json_target_path)
        # define the original regularization        
        return
        
        
    def membrane_optimization_loss(self, passed_pose_index, membrane_physical_attributes): 
        """
        this will call the blender to render the membrane using the passed physical attributes
        
        Returns
        -------
        None.
        """
        self.kinematic_update(passed_pose_index, membrane_physical_attributes) #update the kinematic 
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
            update_simulations(remotes, [membrane_physical_attributes])
          
            # fetch the objs that you want to optimize the 
            rendered_frame = self.end_pose - self.start_pose + 1 + 2 * self.render_buffer #security buffer
            target_objs_list = get_target_objs(
                sim_dl, remotes, n= rendered_frame
            )
        # the rendered obj will be stored in the temp folder
        total_iou_loss_list  = []
        total_iou_loss = 0
        for pose_index in range(passed_pose_index, passed_pose_index+1):   # only render two objs
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
            
            cloth_obj_path = 'D:/PhDProject_real_data/cloth_simulation/{}/{}_only_stiffness.obj'.format(test_name, pose_index)
            mesh = sr.Mesh.from_obj(cloth_obj_path, load_texture=False, texture_res = 1, texture_type='surface')
            vertices = mesh.vertices
            faces = mesh.faces
            vertices = y_forward_z_up(vertices) # manually change the orientation of the exported obj
            mesh = sr.Mesh(vertices.repeat(batch_size, 1, 1),faces.repeat(batch_size, 1, 1))
            
            # save the orientation correct obj for future use
            mesh.save_obj(os.path.join(self.membrane_optimize_mesh_save_path_root, '{}.obj'.format(pose_index)), save_texture=False)
            images_pred = renderer.render_mesh(mesh)
            with torch.no_grad():
                iou_loss = neg_iou_loss(images_pred[:, -1], images_gt[:, 0])      
                total_iou_loss_list.append(iou_loss.item())
                total_iou_loss += iou_loss.item()
            
            frame_number = 1
            average_iou_loss = total_iou_loss / frame_number
        
        #np.save(f"{self.result_path_root}/{self.test_name}_{self.start_pose}_{self.end_pose}_bayes_optimized_only_stiffness_{interation}.npy", np.array(total_iou_loss_list))
        #ITERATION += 1
        return average_iou_loss, total_iou_loss_list
        
    
    def membrane_optimization_loss_bayes_mode(self, membrane_physical_attributes): 
            """
            this will call the blender to render the membrane using the passed physical attributes
            
            Returns
            -------
            None.
            """
            self.kinematic_update(self.current_pose, membrane_physical_attributes) #update the kinematic 
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
                update_simulations(remotes, [membrane_physical_attributes])
              
                # fetch the objs that you want to optimize the 
                rendered_frame = 5
                target_objs_list = get_target_objs(
                    sim_dl, remotes, n= rendered_frame
                )
            # the rendered obj will be stored in the temp folder
            total_iou_loss_list  = []
            total_iou_loss = 0
            for pose_index in range(self.current_pose, self.current_pose+1): 
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
                
                cloth_obj_path = 'D:/PhDProject_real_data/cloth_simulation/{}/{}.obj'.format(test_name, pose_index)
                mesh = sr.Mesh.from_obj(cloth_obj_path, load_texture=False, texture_res = 1, texture_type='surface')
                vertices = mesh.vertices
                faces = mesh.faces
                vertices = y_forward_z_up(vertices) # manually change the orientation of the exported obj
                mesh = sr.Mesh(vertices.repeat(batch_size, 1, 1),faces.repeat(batch_size, 1, 1))
                
                # save the orientation correct obj for future use
                mesh.save_obj(os.path.join(self.membrane_kinematic_optimized_mesh_save_root, '{}_stiffness_damping.obj'.format(pose_index)), save_texture=False)
                images_pred = renderer.render_mesh(mesh)
                with torch.no_grad():
                    iou_loss = neg_iou_loss(images_pred[:, -1], images_gt[:, 0])      
                    total_iou_loss_list.append(iou_loss.item())
                    total_iou_loss += iou_loss.item()
                
            frame_number = 1
            average_iou_loss = total_iou_loss / frame_number
            
            np.save(f"{self.result_path_root}/LBFGS_LCB_low/{self.test_name}_{self.current_pose}_bayes_optimized_kinematic_stiffness_damping_10_degree_{self.iteration}.npy", np.array(total_iou_loss_list))
            self.iteration += 1
            return average_iou_loss
        
    def kinematic_membrane_optimize_bayesian_GP(self, epoch_number): 
        """
        this function will implement the bayesian network for optimizing the membrane parameters

        Returns
        -------
        None.
        """
        rng = np.random.RandomState(123)
        ITERATION = 0

        result = gp_minimize(self.membrane_optimization_loss_bayes_mode,     # The function to minimize
                             searching_space,                   # The search space
                             n_calls=self.membrane_opt_epoch, 
                             acq_func="LCB",
                             kappa = 0.1,
                             acq_optimizer='lbfgs',
                             n_jobs = -1,
                             random_state=rng)         # R
        output_attribute_list = {"physical_attributes":[result.x[0],result.x[1],result.x[2],result.x[3],result.x[4],result.x[5]]} # only save the attributes that are stiffness related
        output_path= os.path.join(self.membrane_optimize_attributes_save_path_root, f"bayesian_attrib_opt_{self.current_pose}.json")
        save_json_file(output_attribute_list, output_path)

        return
    def kinematic_membrane_optimize_bayesian_BNN(self,epoch_number): 
        """
        

        Parameters
        ----------
        epoch_number : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        optimizer = GPyOpt.methods.BayesianOptimization(
                                    f=self.membrane_optimization_loss_bayes_mode,
                                    domain=searching_space_bnn,
                                    acquisition_type='EI',  # Expected Improvement
                                    normalize_Y=True,
                                    initial_design_numdata=5,
                                    evaluator_type='sequential'
                                )
        optimizer.run_optimization(self.membrane_opt_epoch)
        return
    def run_membrane_kinematic_optimize_bayesian_pipeline(self): 
        # initialize the membrane output.json with the original output.json
        self.initialize_membrane_kinematic_training()
        
        
        for pipeline_epoch in range(self.whole_opt_epoch): 
            # optimizing the membrane with kinematic fixed
            self.membrane_optimize_bayesian(pipeline_epoch)
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
        axis_map = {0:"x", 1:"y", 2:"z"}
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
        for index in range(self.start_pose - self.render_buffer, self.end_pose + self.render_buffer + 1): 
            source_json_path = os.path.join(self.kinematic_save_path_root, str(index), "output.json")
            target_json_path = os.path.join(self.kinematic_save_path_root, str(index), "membrane_kinematic_bayes_output_original.json")
            source_json = read_json_file(source_json_path)
            save_json_file(source_json, target_json_path)
            target_json_path = os.path.join(self.kinematic_save_path_root, str(index), "membrane_kinematic_bayes_output_optimizing.json")
            source_json = read_json_file(source_json_path)
            save_json_file(source_json, target_json_path)
        print(f"membrane kinematic initialization done!!")
        return
    
    def plot_sequence_kinematic(self, pose_index, bone_index = 0):
        
        x_euler = []
        y_euler = []
        z_euler = []
        x_euler_original = []
        y_euler_original = []
        z_euler_original = []
        for index in range(pose_index - 2, pose_index + 2 + 1): 
            kinematic_dict = read_json_file(os.path.join(self.kinematic_save_path_root, str(index), "membrane_output.json"))
            bone_dict = kinematic_dict['pose'][bone_index]
            x_euler.append(bone_dict[0])
            y_euler.append(bone_dict[1])
            z_euler.append(bone_dict[2])
            original_dict = read_json_file(os.path.join(self.kinematic_save_path_root, str(index), "output.json"))
            bone_dict =  original_dict['pose'][bone_index]
            x_euler_original.append(bone_dict[0])
            y_euler_original.append(bone_dict[1])
            z_euler_original.append(bone_dict[2])
        plt.subplot(1, 3, 1)  # (rows, columns, index)
        
        plt.title(f"Euler angle for bone: {bone_index}")
        plt.plot(x_euler)
        plt.plot(x_euler_original)
        plt.xlabel("frame_index")
        plt.title('x_euler')
        plt.ylim(min(x_euler)-0.1, max(x_euler)+0.1)
        plt.subplot(1, 3, 2)
        plt.plot(y_euler)
        plt.plot(y_euler_original)
        plt.xlabel("frame_index")
        plt.title('y_euler')
        plt.ylim(min(y_euler)-0.1, max(y_euler)+0.1)
        plt.subplot(1, 3, 3)
        plt.plot(z_euler)
        plt.plot(z_euler_original)
        plt.xlabel("frame_index")
        plt.title('z_euler')
        plt.ylim(min(z_euler)-0.1,max(z_euler)+0.1 )
        plt.legend(["after optimization", "original"])
        plt.locator_params(axis='x', nbins=5)
        
        #plt.tight_layout()  # Adjust subplot parameters for a tight layout
        plt.show()

        return
        
if __name__=="__main__": 
    print("membrane kinematic optimizing driver function")
    project_root_path = "D:/"
    project_name = "PhDProject_real_data"
    test_name = "brunei_2023_bat_test_13_2"

    start_pose =230
    end_pose = 300
    render_start_pose = 195, 
    render_end_pose = 305, 
    kinematic_opt_epoch = 300
    membrane_opt_epoch = 2000
    membrane_kinematic_opt_epoch = 1
    whole_opt_epoch = 10
    interval = 1
    render_buffer = 5  #make sure the start - render_buffer and end_pose + render_buffer is valid
    if_use_previous = False
    opposite_direction = False
    model_template_name = "new_bat_params_version2_backward_membrane_24.pkl"
    driver = Optimize_Driver(project_root_path, 
                             project_name, 
                             test_name, 
                             start_pose, 
                             end_pose, 
                             render_start_pose, 
                             render_end_pose, 
                             kinematic_opt_epoch, 
                             membrane_opt_epoch, 
                             membrane_kinematic_opt_epoch,
                             whole_opt_epoch, 
                             render_buffer, 
                             model_template_name, 
                             if_use_previous, 
                             opposite_direction)
    #driver.kinematic_membrane_optimize_bayesian(0)

    #driver.initialize_membrane_kinematic_training()
    
    #driver.plot_sequence_kinematic(230, bone_index = 7
    for pose_index in range(start_pose, start_pose + 1): 
        driver.current_pose = pose_index
        os.environ['start_range'] = str(pose_index -2)
        os.environ['end_range'] = str(pose_index + 2)
        #driver.kinematic_membrane_optimize_bayesian_GP(0)

    #driver.result_plot(json_name = "membrane_optimize_bayes")
    #_, total_iou_loss_list = driver.membrane_optimization_loss(1)
    #print(total_iou_loss_list)

    loss = []
    first_time = os.path.getmtime(f"./result/LBFGS_LCB_low/brunei_2023_bat_test_13_2_230_bayes_optimized_kinematic_stiffness_damping_10_degree_0.npy")
    time = []
    for iteration in range(176): 
        loss.append(np.load(f"./result//LBFGS_LCB_low/brunei_2023_bat_test_13_2_230_bayes_optimized_kinematic_stiffness_damping_10_degree_{iteration}.npy"))
        first_timestamp = os.path.getmtime(f"./result//LBFGS_LCB_low/brunei_2023_bat_test_13_2_230_bayes_optimized_kinematic_stiffness_damping_10_degree_{iteration}.npy")
        second_timestamp = os.path.getmtime(f"./result//LBFGS_LCB_low/brunei_2023_bat_test_13_2_230_bayes_optimized_kinematic_stiffness_damping_10_degree_{iteration+1}.npy")
        time.append(second_timestamp - first_timestamp)

    plt.plot(loss)
    plt.title("IOU loss for pose 230")
    plt.xlabel("Iteration")
    plt.ylabel("IOU loss")
    plt.show()
    plt.plot(time)
    plt.title("Optimizing time")
    plt.xlabel("iteration")
    plt.ylabel("second")
    plt.show()
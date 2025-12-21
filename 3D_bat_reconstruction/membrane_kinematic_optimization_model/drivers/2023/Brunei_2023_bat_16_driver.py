# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 21:59:25 2025

@author: yihao
"""
import sys
import os
project_root_path = "/home/yihao19/"
sys.path.append(os.path.join(project_root_path,"3D_Flying_Bats_Kinematic_Membrane_Reconstruction/3D_bat_reconstruction/membrane_kinematic_optimization_model"))
from driver import Optimize_Driver
import kinematic_reconstruction_config
import membrane_opt_config
import kinematic_update_config



if __name__ == "__main__":
    project_root_path = "/home/yihao19/"
    project_name = "PhDProject_real_data"
    test_name = "Brunei_2023_bat_16"
    membrane_simulation_mode = "ANGULAR"
    #[580, 1300]
    start_pose = 580
    end_pose =1300
    current_pose_index = start_pose
    membrane_simulation_mode = membrane_opt_config.membrane_simulation_mode
    half_window_size = membrane_opt_config.half_window_size  # animation rendering window size
    membrane_optimized_frame = membrane_opt_config.membrane_optimized_frame# frame number that will be optimized
    kinematic_opt_epoch = kinematic_reconstruction_config.kinematic_opt_epoch
    membrane_opt_epoch = membrane_opt_config.membrane_opt_epoch
    membrane_kinematic_opt_epoch = kinematic_update_config.membrane_kinematic_opt_epoch
    whole_opt_epoch =kinematic_update_config.whole_opt_epoch
    if_use_previous_attr = membrane_opt_config.if_use_previous_attr
    if_use_previous_kinematics =kinematic_reconstruction_config.if_use_previous_kinematics
    model_template_name = kinematic_reconstruction_config.model_template_name
    
    opposite_direction = True # bat flying direction
    template_flip = False
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
                             if_use_previous_attr,
                             if_use_previous_kinematics,
                             opposite_direction)
    
    #driver.run_raw_kinematic_optimize_pipeline()
    #exit(0)
    #driver.run_kinematic_smoothing()
    driver.run_membrane_optimize_pipeline(epoch_index = 0)
    #driver.up_down_stroke_stiffness()
    #exit(0)
    #driver.plot_camera_number()
    #driver.run_membrane_kinematic_update_pipeline(epoch_index=0)
    #driver.run_original_reconstruction()
    #driver.iou_loss_initial_vs_final()
    #driver.generate_flight_speed()
    #driver.stiffness_visualization()
    #driver.stiffness_visualization_frame(pose_index=1000)
    #driver.iou_loss_compare()
    #driver.iou_loss_compare()
    #driver.stiffness_visualization()
    #driver.plot_initial_kinematic(kinematic_smoothed=False)
    #driver.plot_initial_kinematic(kinematic_smoothed=True)
   
    #driver.iou_loss_original()
    #driver.run_original_kinematic_smooth_rendering()
    #driver.iou_loss_membrane_compare()
    #driver.scale_parameter_plot()
    #driver.generate_flying_trajectory_gif(if_smoothed=True)
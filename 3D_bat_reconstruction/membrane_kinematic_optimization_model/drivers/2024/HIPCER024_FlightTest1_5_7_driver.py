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




if __name__ == "__main__":
    project_name = "PhDProject_real_data"
    test_name = "Brunei_2024_HIPCER024_FlightTest1_5_7"
    membrane_simulation_mode = "ANGULAR"
    #[2666,2850 ,3255]
    start_pose = 2700#2849
    end_pose =3200#
    current_pose_index = start_pose
    half_window_size = 8  # animation rendering window size
    membrane_optimized_frame = 1# frame number that will be optimized
    kinematic_opt_epoch = 50
    membrane_opt_epoch =100
    membrane_kinematic_opt_epoch = 10
    whole_opt_epoch =1
    if_use_previous_attr = False
    if_use_previous_kinematics = True
    opposite_direction = True # bat flying direction
    template_flip = True
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
                             opposite_direction,
                             template_flip=template_flip
                             )
    
    #driver.run_raw_kinematic_optimize_pipeline()
    #exit(0)
    #driver.run_kinematic_smoothing()
    driver.run_membrane_optimize_pipeline(epoch_index = 0)
    driver.generate_flight_speed()
    driver.stiffness_visualization()
    exit(0)    
    #driver.run_membrane_kinematic_update_pipeline(epoch_index=0)
    driver.run_original_reconstruction()
    #driver.stiffness_visualization()
    driver.plot_initial_kinematic(kinematic_smoothed=False)
    driver.plot_initial_kinematic(kinematic_smoothed=True)
    #driver.iou_loss_compare()
    driver.iou_loss_original()
    #driver.run_original_kinematic_smooth_rendering()
    #driver.iou_loss_membrane_compare()
    #driver.scale_parameter_plot()
    driver.generate_flying_trajectory_gif(if_smoothed=True)
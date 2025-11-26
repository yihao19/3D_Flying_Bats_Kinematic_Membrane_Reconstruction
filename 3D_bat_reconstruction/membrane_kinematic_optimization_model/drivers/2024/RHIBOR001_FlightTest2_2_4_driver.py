# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 21:59:25 2025

@author: yihao
"""
import sys
import os
project_root_path = "/home/yihao19/"
sys.path.append(os.path.join(project_root_path,"PhD_research/3D_bat_reconstruction/SoftRas/models/membrane_kinematic_optimization_model"))
from driver import Optimize_Driver




if __name__ == "__main__":
    project_name = "PhDProject_real_data"
    test_name = "Brunei_2024_RHIBOR001_FlightTest2_2_4"
    membrane_simulation_mode = "ANGULAR"
    start_pose = 340
    end_pose = 341
    current_pose_index = start_pose
    half_window_size = 8  # animation rendering window size
    membrane_optimized_frame = 1# frame number that will be optimized
    kinematic_opt_epoch = 300
    membrane_opt_epoch =100
    membrane_kinematic_opt_epoch = 10
    whole_opt_epoch =1
    if_use_previous_attr = False
    if_use_previous_kinematics =False
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
    
    driver.run_raw_kinematic_optimize_pipeline()
    #driver.run_membrane_optimize_pipeline(epoch_index = 0)
    #driver.run_membrane_kinematic_update_pipeline(epoch_index=0)
    #driver.stiffness_visualization()
    #driver.iou_loss_stage()
    #driver.run_original_reconstruction()
    #driver.iou_loss_compare()
    #driver.run_kinematic_smooth_only(epoch_index=0)
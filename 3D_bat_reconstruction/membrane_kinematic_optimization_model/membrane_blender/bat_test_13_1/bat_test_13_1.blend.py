# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:43:06 2024

@author: yihao
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 00:06:55 2024

@author: yihao
"""

import bpy
import math
import numpy as np
from bpy_extras.io_utils import axis_conversion
import blendtorch.btb as btb
import mathutils
from bpy import context
from mathutils import Vector, Matrix
import json
import time
import os

def rodrigues(pose): 
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

def read_json_file(file_path): 
    with open(file_path) as f: 
        pose_dict = json.load(f)
        
    return pose_dict


def kinematic_smoothing(): 
    # Find the Graph Editor area
    
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    graph_area = next((area for area in bpy.context.screen.areas if area.type == 'GRAPH_EDITOR'), None)
    if graph_area:
        # Override the context to the Graph Editor
        with bpy.context.temp_override(area=graph_area):
            
            bpy.ops.graph.gaussian_smooth(factor=1, sigma=0.33, filter_width=6)
    else:
        print("Graph Editor area not found.")
    bpy.ops.object.mode_set(mode='OBJECT') # don't forget this step for normal rendering
    
    return
    
def membrane_config(start, end, attrib_dict):
   
    mesh_obj_name = "Icosphere.001"
    obj =  bpy.data.objects.get(mesh_obj_name)

    cloth_modifier = obj.modifiers.new(name="Cloth", type='CLOTH')
       
    point_cache = cloth_modifier.point_cache
    point_cache.frame_start = start  # Set the start frame
    point_cache.frame_end = end  # Set the end frame
    #cloth_modifier = obj.modifiers["Cloth"]
    settings = cloth_modifier.settings
    
    settings.bending_model = 'LINEAR'
    settings.gravity = Vector((0.0, 0.0, 0.0))
    
    
    settings.quality = 5  # Simulation quality
    settings.mass = 0.0000001  # Cloth mass
    settings.air_damping = 1  # Air resistance
    settings.tension_stiffness =attrib_dict[0][0] # Tension stiffness
    settings.shear_stiffness = attrib_dict[0][1] # Shear stiffness
    settings.bending_stiffness = attrib_dict[0][2]  # Bending stiffness
    
    settings.tension_damping = attrib_dict[0][3] # Tension stiffness
    settings.shear_damping = attrib_dict[0][4] # Shear stiffness
    settings.bending_damping = attrib_dict[0][5]  # Bending stiffness
    
    

    vertex_group = obj.vertex_groups.get("pin_group")
    if vertex_group:
        settings.vertex_group_mass = "pin_group"
    else:
        print("Pinned vertex group not found.")
    


def load_kinematics(start_frame, end_frame): 
    #obj = bpy.context.selected_objects[0]
    armature_name = "Armature"  # Change this to match your armature's name
    armature = bpy.data.objects.get(armature_name)
    
    if armature is None or armature.type != 'ARMATURE':
        raise Exception(f"Object '{armature_name}' not found or is not an armature.")
    
    # Make sure you're in object mode
    if bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Clear all keyframes from the armature
    armature.animation_data_clear()
    
    # Enter pose mode to manipulate bone transforms
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')
    
    # Reset all bones to rest pose
    for bone in armature.pose.bones:
        bone.location = (0, 0, 0)
        bone.rotation_quaternion = (1, 0, 0, 0)
        bone.scale = (1, 1, 1)
    
                                 
    bpy.ops.object.mode_set(mode='OBJECT')           
        
    for file_index in range(start_frame,end_frame+1): 
            #print(file_index)
        #if file_index == 1:
        json_file_path = f"D:/PhDProject_real_data/brunei_2023_bat_test_13_1/rearrange_pose/{file_index}/membrane_output.json"
        pose_dict = read_json_file(json_file_path)
        #template_displacement = [math.floor(pose_dict['template_displacement'][0]*1e4)/1e4, math.ceil(pose_dict['template_displacement'][1]*1e4)/1e4,math.ceil(pose_dict['template_displacement'][2]*1e4)/1e4]
        
        
        bone_rest_rotation = read_json_file('D:/PhD_research/3D_bat_reconstruction/SoftRas/models/membrane_kinematic_optimization_model/model_template/template_bone_rest_rotation.json')
        bone_rest_rotation = bone_rest_rotation['bone_rest_rotation']
      
        template_displacement = Vector(pose_dict['template_displacement']) 
        scale = 0.005
    
        pose_array = pose_dict['pose'] 
    
        # Get the active bone. If you want to operator on a specific bone, use it's name instead. arm.pose.bones['Bone']
        #bones = context.selected_objects[0].pose.bones
       
        for bone_index, bone in enumerate(armature.pose.bones):

            quat = rodrigues(pose_array[bone_index])
            mat = quat_to_rotmat(quat) 
            eu = Matrix(mat)
            bone_rest = bone_rest_rotation[bone_index]
            bone_rest = Matrix(np.array(bone_rest))
            
    
            final_rotation = eu @ bone_rest 
            
    
            #print(l+Vector(template_displacement)*285)
            joints = np.array(pose_dict['joints'][0]) 
           
            
            joint = Vector(joints[bone_index])
            mat = mathutils.Matrix.LocRotScale(None, final_rotation, None)
            
            bone.matrix  = mat
            
            #bone.keyframe_insert(data_path="location", frame=file_index)     
            bone.keyframe_insert(data_path="rotation_quaternion", frame=file_index)
            #bones = context.selected_objects[0].pose.bones
        bone = armature.pose.bones[0]
        quat = rodrigues(pose_array[0])
        rot_matrix = quat_to_rotmat(quat)
        eu = Matrix(rot_matrix)
        bone_rest = bone_rest_rotation[0]
        bone_rest = Matrix(np.array(bone_rest))
    
                 
        final_rotation =  eu @ bone_rest
            
        joints = np.array(pose_dict['joints'][0])
            
        joint = Vector(joints[0])
        mat = mathutils.Matrix.LocRotScale((joint)*1/scale,final_rotation, None)
            
        bone.matrix  = mat
      
        bone.keyframe_insert(data_path="location", frame=file_index)     
        bone.keyframe_insert(data_path="rotation_quaternion", frame=file_index)

    # Find the Graph Editor area
   
    return


class Blender_animator(): 
    def __init__(self, start_frame, end_frame, test_name="brunei_2023_bat_test_13_1" , obj_save_root="D:\\PhDProject_real_data\\"): 
        btargs, remainder = btb.parse_blendtorch_args()
        self.btargs = btargs
        self.remainder = remainder
        self.pub = btb.DataPublisher(self.btargs.btsockets["DATA"], self.btargs.btid)
        self.duplex = btb.DuplexChannel(self.btargs.btsockets["CTRL"], self.btargs.btid)
        self.spawn_function = btb.ObjSpawner(name= "obj_spawner", test_name=test_name, obj_save_root = obj_save_root)
        #off.set_render_style(shading="RENDERED", overlays=False)
        self.cam = None
        # Setup the animation and run endlessly
        self.anim = btb.AnimationController()
        self.start_frame = start_frame
        self.end_frame = end_frame
        #msgs = self.duplex.recv(timeoutms=0)
        #print(msgs)
        #attrib_dict = dict(msgs)
        
        load_kinematics(start_frame, end_frame)
        kinematic_smoothing()
    def pre_frame(self): 
        """
        will be executed before rendering each frame

        Returns
        -------
        None.

        """
        msgs = self.duplex.recv(timeoutms=10)
        if(msgs != None):
            attrib_dict = msgs['shape_params']
            membrane_config(self.start_frame, self.end_frame,attrib_dict)
        else: 
            return

        return
    def post_frame(self):
        # Called every after Blender finished processing a frame.
        # here is the place to export the kinematics
        # Will be sent to one of the remote dataset listener connected.
        self.pub.publish(
            obj=self.spawn_function.spawn(self.anim.frameid), xy=0, frameid=self.anim.frameid
        )
        # try to read the rest_pose and convert it to matrix
    
    def flying_job_creator(self): 
        
        return
    def animate(self): 
        self.anim.pre_frame.add(self.pre_frame)
        self.anim.post_frame.add(self.post_frame)
        self.anim.play(frame_range=(self.start_frame, self.end_frame), num_episodes=-1)


    
if __name__=="__main__": 

    start_pose = os.environ.get('start_range')
    end_pose = os.environ.get('end_range')
    animator = Blender_animator(start_frame = int(start_pose)-1, end_frame = int(end_pose)+1)
    animator.animate()

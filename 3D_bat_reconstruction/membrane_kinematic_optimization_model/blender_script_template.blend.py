"""
Blender driver function for loading the initial kinematic, smooth, render membrane 
"""
import bpy
import math
import numpy as np
from bpy_extras.io_utils import axis_conversion

import mathutils
from bpy import context
from mathutils import Vector, Matrix
import json
import time
import sys
import os

import bpy

def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = cycles_preferences.devices

    if not devices:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []
    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)
            print('activated gpu', device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus


enable_gpus("CUDA")


def y_forward_z_up(vertices, scale_factor:float=1.0): 
    x = np.expand_dims(vertices[:,  0] * scale_factor, 1) 
    y = np.expand_dims(-vertices[:, 2] * scale_factor, 1)
    z = np.expand_dims(vertices[:,  1] * scale_factor, 1)
    vertices =np.concatenate([x, y, z], axis=1)
    return vertices


def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """
    Save a mesh to OBJ format.
    
    :param vertices: (N,3) float array of vertex coordinates.
    :param faces:    (M,K) int array of face indices (0-based).
    :param filepath: Path to output .obj file.
    :param comment:  Optional string to insert as a comment at top.
    """
    # Validate shapes
    assert vertices.ndim == 2 and vertices.shape[1] == 3, "vertices must be shape (N,3)"
    assert faces.ndim == 2, "faces must be shape (M,K)"
    
    with open(filepath, 'w') as f:
        # optional comment
        # write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # write faces (convert to 1-based)
        for face in faces:
            # Convert each index to 1-based
            face_indices = face + 1
            # Join them into space-separated
            idx_str = " ".join(str(idx) for idx in face_indices)
            f.write(f"f {idx_str}\n")

    return None

def load_obj(filename):
    """
    Load a Wavefront OBJ file and return vertices and faces.
    
    :param filename: Path to .obj file.
    :return: (vertices, faces)
      - vertices is an (N,3) float array
      - faces is a (M,K) int array (K = number of vertices per face; often 3)
    """
    verts = []
    faces = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v':
                # Vertex position
                x, y, z = map(float, parts[1:4])
                verts.append((x, y, z))
            elif parts[0] == 'f':
                # Face: can be "f v1 v2 v3", or "f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3"
                face = []
                for v in parts[1:]:
                    # take vertex index part before any '/'
                    v_idx = v.split('/')[0]
                    # OBJ is 1-based indexing, so subtract 1
                    face.append(int(v_idx) - 1)
                faces.append(tuple(face))
    
    vertices = np.array(verts, dtype=float)
    # make faces into an array: if all faces have same length K, this works
    # else this will create an array of object dtype
    faces = np.array(faces, dtype=int)
    
    return vertices, faces

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

def read_json_file(file_path):
    """
    util function to read the json file and return the content

    Parameters
    ----------
    file_path : TYPE
        DESCRIPTION.

    Returns
    -------
    pose_dict : TYPE
        DESCRIPTION.

    """
    with open(file_path) as f: 
        pose_dict = json.load(f)
        
    return pose_dict

def scene_setting(start_frame:int, end_frame:int) -> None:
    """
    

    Parameters
    ----------
    start_frame : int
        DESCRIPTION.
    end_frame : int
        DESCRIPTION.

    Returns
    -------
    None
        DESCRIPTION.

    """
    scene = bpy.context.scene

    # Set frame range:
    scene.frame_start = start_frame
    scene.frame_end   = end_frame
    return

def pose_reset(armature_name:str="Armature.001"):
    """
    this function will result all the pose into the rest pose. 
    armature_name = "Armature.001" 
    Returns
    -------
    None.
    """
    
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
    # kinematic reset is done is pose mode
    bpy.ops.object.mode_set(mode='POSE')
    
    # Reset all bones to rest pose
    for bone in armature.pose.bones:
        bone.location = (0, 0, 0)
        bone.rotation_quaternion = (1, 0, 0, 0)
        bone.scale = (1, 1, 1)
    bpy.ops.object.mode_set(mode='OBJECT') # swith the mode into object for      
    return armature

def kinematic_smoothing(factor:int=1, sigma:float=0.33, filter_width:int=6) -> None: 
    # Find the Graph Editor area
    """
    function to use gaussian smoothing for kinamtic( deprecated)
    """
    
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    graph_area = next((area for area in bpy.context.screen.areas if area.type == 'GRAPH_EDITOR'), None)
    if graph_area:
        # Override the context to the Graph Editor
        with bpy.context.temp_override(area=graph_area):
            bpy.ops.graph.gaussian_smooth(factor =factor, sigma=sigma, filter_width=filter_width)
    else:
        print("Graph Editor area not found.")
    bpy.ops.object.mode_set(mode='OBJECT') # don't forget this step for normal rendering
    return

def load_kinematics(armature, start_frame:int, end_frame:int, reconstruction_project:str,epoch_index:int = 0, if_membrane_opt:bool=True, scale_factor:float=1.0) -> None:
    """
    function to load all the kinematics from raw reconstruction json

    Parameters
    ----------
    start_frame : int
        DESCRIPTION.
    end_frame : int
        DESCRIPTION.
    reconstruction_project : name of the reconstruction sequence
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    None
        DESCRIPTION.

    """
    #obj = bpy.context.selected_objects[0]
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
        if if_membrane_opt:
            json_file_path = f"/home/yihao19/PhDProject_real_data/{reconstruction_project}/rearrange_pose/{file_index}/membrane_output_{epoch_index}.json"
        else:
            json_file_path = f"/home/yihao19/PhDProject_real_data/{reconstruction_project}/rearrange_pose/{file_index}/output_smoothed.json"
        pose_dict = read_json_file(json_file_path)
        #template_displacement = [math.floor(pose_dict['template_displacement'][0]*1e4)/1e4, math.ceil(pose_dict['template_displacement'][1]*1e4)/1e4,math.ceil(pose_dict['template_displacement'][2]*1e4)/1e4]
        
        
        bone_rest_rotation = read_json_file('/home/yihao19/3D_Flying_Bats_Kinematic_Membrane_Reconstruction/3D_bat_reconstruction/membrane_kinematic_optimization_model/model_template/template_bone_rest_rotation.json')
        bone_rest_rotation = bone_rest_rotation['bone_rest_rotation']
        
       
        
        scale = 0.005
        
        pose_array = pose_dict['pose']
        
        if(len(pose_dict['pose']) == 34): 
            # this is the paper one design, append 6 more bones
            pose_dict['pose'].append([0,0,0])
            pose_dict['pose'].append([0,0,0])
            pose_dict['pose'].append([0,0,0])
            pose_dict['pose'].append([0,0,0])
            pose_dict['pose'].append([0,0,0])
            pose_dict['pose'].append([0,0,0])
        # Get the active bone. If you want to operator on a specific bone, use it's name instead. arm.pose.bones['Bone']
        #bones = context.selected_objects[0].pose.bones
        # if the 5_7 the location needs to be rescaled by 0.35 
       
        pose_dict['joints'][0] = np.array(pose_dict['joints'][0]) * scale_factor

        for bone_index, bone in enumerate(armature.pose.bones):
            bone_name = bone.name
            parts = bone_name.split('.')
            if(len(parts) == 1): 
                bone_index = 0
            else: 
                bone_index = int(parts[1])
            quat = rodrigues(pose_array[bone_index])
            mat = quat_to_rotmat(quat) 
            eu = Matrix(mat)
            bone_rest = bone_rest_rotation[bone_index]
            bone_rest = Matrix(np.array(bone_rest))
            
    
            final_rotation = eu @ bone_rest 
            
    
            #print(l+Vector(template_displacement)*285)
            joints = np.array(pose_dict['joints'][0]) 
           
            
            #joint = Vector(joints[bone_index])
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

def membrane_cloth_setting(tension_stiffness:float,
                           start_frame:int, 
                           end_frame:int,
                           mesh_obj_name:str = "Icosphere.001", 
                           opt_quality:int=5,
                           mass:float=0.000001,
                           air_damping:float=1, 
                           bending_model:str="ANGULAR") -> None:
    
   
    obj =  bpy.data.objects.get(mesh_obj_name)

    cloth_modifier = obj.modifiers.new(name="Cloth", type='CLOTH')
       
    point_cache = cloth_modifier.point_cache
    point_cache.frame_start = start_frame  # Set the start frame
    point_cache.frame_end = end_frame  # Set the end frame
    #cloth_modifier = obj.modifiers["Cloth"]
    settings = cloth_modifier.settings
    vertex_group = obj.vertex_groups.get("pin_group")
    if vertex_group:
        settings.vertex_group_mass = "pin_group"
    else:
        print("Pinned vertex group not found.")
    settings.bending_model = bending_model
    settings.gravity = Vector((0.0, 0.0, 0.0))
    
    
    settings.quality = opt_quality # Simulation quality
    settings.mass =mass  # Cloth mass
    settings.air_damping = air_damping# Air resistance
    
    settings.tension_stiffness =tension_stiffness # Tension stiffness
    settings.compression_stiffness = 0
    settings.bending_stiffness = 0
    
    settings.tension_damping = 0
    settings.compression_damping = 0
    settings.bending_damping = 0
  
    collision_settings = cloth_modifier.collision_settings
    collision_settings.use_collision = False
    collision_settings.use_self_collision = False

    return

def spawn(self, frame_index):
    """Render the scene and return image as buffer.

    Returns
    -------
    image: HxWxD array
        where D is 4 when `mode=='RGBA'` else 3.
    """
    mesh_data = {}

    obj = bpy.data.objects['Icosphere.001'] 
    if not os.path.exists(os.path.join(self.obj_save_root, self.test_name, "blender_render")):
        os.makedirs(os.path.join(self.obj_save_root, self.test_name, "blender_render"))
    save_path = os.path.join(self.obj_save_root, self.test_name, "blender_render",f"{frame_index}.obj")
    bpy.ops.wm.obj_export(filepath=save_path)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_obj.to_mesh()
    # Access the vertex coordinates
    mesh_data = {
        'vertices': np.array([v.co for v in evaluated_mesh.vertices]),
        'faces': np.array([f.vertices for f in obj.data.polygons]),
        'object_index': obj.pass_index
    }
    
    return mesh_data

def mesh_render(obj_save_root:str,
                reconstruction_project:str,
                start_frame:int,
                end_frame:int,
                mesh_name:str='Icosphere.001',
                scale_factor:float=1.0) -> None:
    scene = bpy.context.scene
    for frame_index in range(start_frame, end_frame +1): 
        scene.frame_set(frame_index)
        mesh_data = {}
    
        obj = bpy.data.objects[mesh_name] 
        if not os.path.exists(os.path.join(obj_save_root, reconstruction_project, "blender_render")):
            os.makedirs(os.path.join(obj_save_root, reconstruction_project, "blender_render"))
        save_path = os.path.join(obj_save_root, reconstruction_project, "blender_render",f"{frame_index}.obj")
        bpy.ops.wm.obj_export(filepath=save_path)
        original_vertices, original_faces = load_obj(save_path)
        rectified_vertices = y_forward_z_up(original_vertices, scale_factor=scale_factor)
        if(frame_index == end_frame):
            save_path = os.path.join(obj_save_root, reconstruction_project, "membrane_optimized_mesh", f"{frame_index}.obj")
            save_obj(rectified_vertices, original_faces,save_path)
        else:
            continue
    return None

def original_mesh_render(obj_save_root:str,
                reconstruction_project:str,
                start_frame:int,
                end_frame:int,
                mesh_name:str='Icosphere.001', ) -> None:
    """
    apply kinematic smooth only to the mesh template
    """
    scene = bpy.context.scene
    for frame_index in range(start_frame, end_frame +1): 
        scene.frame_set(frame_index)
        if not os.path.exists(os.path.join(obj_save_root, reconstruction_project, "blender_render")):
            os.makedirs(os.path.join(obj_save_root, reconstruction_project, "blender_render"))
        save_path = os.path.join(obj_save_root, reconstruction_project, "blender_render",f"{frame_index}.obj")
        print(save_path)
        bpy.ops.wm.obj_export(filepath=save_path)
        original_vertices, original_faces = load_obj(save_path)
        rectified_vertices = y_forward_z_up(original_vertices)
        if(frame_index == end_frame):
            save_path = os.path.join(obj_save_root, reconstruction_project, "original_kinematic_smooth", f"{frame_index}.obj")
            save_obj(rectified_vertices, original_faces,save_path)
        else:
            continue
    return None

def membrane_reconstruction_render(config) -> None: 
    """
    the main function to load the kinematics from json file and enable membrane cloth, render the final membrane optimized mesh

    Parameters
    ----------
    start_frame : int
        DESCRIPTION.
    end_frame : int
        DESCRIPTION.
    reconstruction_name : str
        DESCRIPTION.

    Returns
    -------
    None
        DESCRIPTION.

    """
    
    #1. reset the pose
    start_frame = config['start_frame']
    end_frame = config['end_frame']
    project_root = config['project_root']
    epoch_index = int(config['epoch_index'])
    reconstruction_project = config['reconstruction_project']
    tension_stiffness = config['tension_stiffness']
    if_membrane_opt = config['if_membrane_opt']
    if("5_7" in reconstruction_project):
        scale_factor = 0.328
    else:
        scale_factor = 1.0
    print(scale_factor)
    #scene setting
    scene_setting(start_frame, end_frame)
    armature = pose_reset()
    #2. enable the mesh template cloth
    if if_membrane_opt:
        # if optimize the membrane
        membrane_cloth_setting(tension_stiffness, start_frame, end_frame)
        #3. load kinematic  of the project
        load_kinematics(armature, start_frame, end_frame, reconstruction_project, epoch_index = epoch_index, if_membrane_opt=if_membrane_opt,scale_factor=scale_factor)
        #4. smooth the loaded kinematic
        #kinematic_smoothing()
        #5. render the membrane enabled mesh
        mesh_render(project_root, reconstruction_project, start_frame, end_frame, scale_factor=1/scale_factor)
          # quit blender
    else: 
        #3. load the kinematics
        load_kinematics(armature, start_frame, end_frame, reconstruction_project, epoch_index = epoch_index, if_membrane_opt=if_membrane_opt)
        #4. smooth the loaded kinematic
        #kinematic_smoothing()
        #5. render the membrane enabled mesh
        original_mesh_render(project_root, reconstruction_project, start_frame, end_frame)
    bpy.ops.wm.quit_blender()
    return None


if __name__=="__main__":


# Find the index of '--' to separate Blender's arguments from custom ones
    if '--' in sys.argv:
        custom_args = sys.argv[sys.argv.index('--') + 1:]
    else:
        custom_args = []
    PROJECT_YEAR = "2023"
    project_root =  custom_args[0]
    reconstruction_project = custom_args[1]
    start_frame = custom_args[2]
    end_frame = custom_args[3]
    tension_stiffness = custom_args[4]
    epoch_index = custom_args[5]
    if_membrane_opt = custom_args[6]
    print("Custom arguments:", custom_args)

    blender_config = {"start_frame":int(start_frame), 
                      "end_frame":int(end_frame), 
                      "project_root":str(project_root),
                      #"reconstruction_project":str(f"Brunei_{PROJECT_YEAR}_"+reconstruction_project),
                      "reconstruction_project":str(reconstruction_project),
                      "tension_stiffness":float(tension_stiffness),
                      "epoch_index":int(epoch_index),
                      "if_membrane_opt":True if if_membrane_opt.lower()=='true' else False}
    membrane_reconstruction_render(blender_config)

    #./blender /home/yihao19/3D_Flying_Bats_Kinematic_Membrane_Reconstruction/3D_bat_reconstruction/membrane_kinematic_optimization_model/membrane_blender/bat_15_1/bat_15_1.blend -b --python-use-system-env --python /home/yihao19/3D_Flying_Bats_Kinematic_Membrane_Reconstruction/3D_bat_reconstruction/membrane_kinematic_optimization_model/blender_script_template.blend.py -- /home/yihao19/PhDProject_real_data bat_15_1 102 110 0.018193772992058585 0 True
    #./blender /home/yihao19/3D_Flying_Bats_Kinematic_Membrane_Reconstruction/3D_bat_reconstruction/membrane_kinematic_optimization_model/membrane_blender/2024/bat.blend -b --python-use-system-env --python /home/yihao19/3D_Flying_Bats_Kinematic_Membrane_Reconstruction/3D_bat_reconstruction/membrane_kinematic_optimization_model/blender_script_template.blend.py -- /home/yihao19/PhDProject_real_data Brunei_2024_HIPCER021_FlightTest1_5_7 3030 3040 0.018193772992058585 0 True
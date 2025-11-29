import torch
import torch.nn as nn
import os
import numpy as np
import pickle
import math
import soft_renderer as sr
import math
from LBS import LBS

PROJECT_ROOT = "/home/yihao19/"
class Kinematic_model(nn.Module):
    def __init__(self, 
                 bone_skining_matrix_name:str, 
                 template_obj_path = os.path.join(PROJECT_ROOT,"3D_Flying_Bats_Kinematic_Membrane_Reconstruction/3D_bat_reconstruction/membrane_kinematic_optimization_model/dummy.obj"),
                 train_skining_matrix = False,
                 use_previous = True, 
                 opposite_direction=False, 
                 template_flip=False):
        super(Kinematic_model, self).__init__()
        # set template mesh
        # the mesh object no need to change since the vertices will move with the
        # joints
        # put the mesh of the model in rest pose in OBJ file and bone and default skining
        # matrix in the corresponding pkl file
        #self.estimated_location_file = estimated_location_file
        self.opposite_direction = opposite_direction
        self.use_previous = use_previous
        self.template_flip = template_flip
        self.template_mesh = sr.Mesh.from_obj(template_obj_path, load_texture=False, texture_res = 5, texture_type='surface')
        with open(os.path.join(PROJECT_ROOT,"3D_Flying_Bats_Kinematic_Membrane_Reconstruction/3D_bat_reconstruction/membrane_kinematic_optimization_model/model_template", bone_skining_matrix_name), 'rb') as f:
            data = pickle.load(f)
        # generate the .obj file from params.pkl
        self.template_mesh.vertices = torch.tensor(data['v_template']).float().unsqueeze(0).cuda()
        self.template_mesh.faces = torch.tensor(data['faces']).unsqueeze(0).cuda()
        #self.template_mesh.face_vertices =  
        #print(self.template_mesh.face_vertices)
        #self.template_mesh.faces = torch.tensor(data['faces']).unsqueeze(0).cuda()
        joints = data['joints_matrix'][:3, :].transpose()
        joints_tail = data['joints_matrix'][3:, :].transpose()
        
        #print(np.linalg.norm(joints_tail[7] - joints_tail[5]))

        self.joint_number = joints.shape[0]
        skining = data['weights']
        #kintree_table = data['kintree']  # numpy array that define the kinematic tree of the skeleton
        if(train_skining_matrix == False):
            # use the default 
            skining_tensor  = torch.tensor(skining).unsqueeze(0).cuda()
            self.register_buffer('skining',skining_tensor)
            self.register_buffer("skining_adjust",torch.zeros_like(skining_tensor))
        else: 
            # make skinging_tensor a trainable parameters
            # make the skining matrix trainable just like the joint rotation
            skining_tensor  = torch.tensor(skining).unsqueeze(0).cuda()
            self.register_buffer('skining',skining_tensor)
            self.register_parameter("skining_adjust", nn.Parameter(torch.zeros_like(skining_tensor)))
            #self.skining = skining_tensor
        # importing the bones and skining matrix of a bat model
        # make the skining matrix the registered param
        # and joints the registered param
        # first, test the 
        joints_tensor = torch.tensor(joints).unsqueeze(0).cuda()
        joints_tail_tensor = torch.tensor(joints_tail).unsqueeze(0).cuda()
        
        # define the kintree of the skeleton of the bat
        # define in the Blender 
        kintree_table = np.array([[ -1, 0, 1, 2, 2, 4, 5, 6, 7, 8, 9,  7,  11, 12, 7 , 14, 15, 2,  17, 18, 19, 20, 21, 22, 20, 24, 25, 20, 27, 28, 0,  30, 0,  32, 10, 13, 16, 23, 26, 29],
                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]])

        
        
        # define the pose matrix for the joints in the passing list
        # empty pose for all bones
        #cuda all the parameters
        #trainable parameters 
        self.kintree_table = torch.tensor(kintree_table).cuda()
        self.register_buffer('joints',joints_tensor)
        self.register_buffer('joints_tail', joints_tail_tensor)
        self.register_buffer('vertices', self.template_mesh.vertices)
        self.register_buffer('faces', self.template_mesh.faces)
        self.parents = self.kintree_table[0].type(torch.LongTensor)
        
        # for each bone, then 
        self.training_skining_weight = torch.clone(self.skining)
        if(train_skining_matrix == True):
            weights_mask =  torch.tensor(data['weights_mask']).unsqueeze(0).cuda()
            
            training_skining_weight_scope = (self.skining + self.skining_adjust) * weights_mask
            # replace all zero value to -inf
            self.training_skining_weight =training_skining_weight_scope - torch.where(training_skining_weight_scope != 0, torch.zeros_like(training_skining_weight_scope), torch.ones_like(training_skining_weight_scope) * float('inf'))
            self.training_skining_weight = torch.softmax(self.training_skining_weight, dim = 2)
        
        
    
        
        self.LBS_model = LBS(self.joints, self.parents, self.training_skining_weight)# define the LBS model
        self.LBS_model_joint_tail = LBS(self.joints_tail, self.parents, self.training_skining_weight)
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
        self.pose_tensor = torch.zeros((40, 3)).cuda()
        
        self.laplacian_smoothing = sr.LaplacianLoss(self.vertices[0].cpu(), self.faces[0].cpu())
        
        #print(self.laplacian_loss.laplacian)
        #laplacian_loss = self.laplacian_loss(self.vertices).mean()
        #print(laplacian_loss)
        #self.flatten_loss = sr.FlattenLoss(self.faces[0].cpu())
        #
    
    '''
    # model's forward function'
    
    '''
    def forward(self, batch_size, estimated_location, use_previous = False, prev_pose = None):
        
        # define how the vertices is going to change
        # add the displacement first
        # rotation the model next
        # rotate the template vertices and the skeleton
        #vertices = torch.bmm(self.vertices,rotation) 
        #joints = torch.bmm(self.joints, rotation.double())
        # figure out the joint rotation matrix third
        #np.savetxt('./test_1.txt', vertices.detach().cpu().numpy()[0])
        
        # using sigmoid / tanh function to limit the rotation degree
        # limit the bone rotation to 45 degree maximum pi/4
        # sigmoid x - > (0, 1) 
        # tanh    x - > (-1, 1)
        # assign all bones with fill DOF but a limited rotation angle
        # body rotation matrix
        if(use_previous == True):
            
            self.pose_tensor[0][:] = (prev_pose[0][:] + math.pi / 6 * torch.tanh(self.joint_0))
            
            #self.pose_tensor[1][0] = ( prev_pose[1][0] + math.pi / 3 * torch.tanh(self.joint_1[0][0])) 
            self.pose_tensor[2][0] = ( prev_pose[2][0] + math.pi / 6 * torch.tanh(self.joint_2[0][0])) 
            #self.pose_tensor[3][0] = ( prev_pose[3][0] + math.pi / 18 * torch.tanh(self.joint_3[0][0]))
            
            
            #self.pose_tensor[1][:] = math.pi / 4 * torch.tanh(self.joint_1)
            #self.pose_tensor[2][:] = prev_pose[2][:] + math.pi / 9 * torch.tanh(self.joint_2)  # make the neck bone trainable(slightly)
            # for the shoulder using the symmetric deformation
            #self.pose_tensor[5][0] = (prev_pose[5][0] + math.pi / 9 * torch.tanh(self.joint_5[0][0]))
            low_bound_y = -math.pi/9
            high_bound_y = math.pi/9
            self.pose_tensor[5][1] = max(low_bound_y, min(high_bound_y,(prev_pose[5][1] + math.pi / 3 * torch.tanh(self.joint_5[0][1]))))
            self.pose_tensor[5][2] =  (prev_pose[5][2] + math.pi / 3 * torch.tanh(self.joint_5[0][2]))
            
            #self.pose_tensor[18][0] = (prev_pose[18][0] + math.pi / 9 * torch.tanh(self.joint_5[0][0]))
            self.pose_tensor[18][1] = max(low_bound_y, min(high_bound_y,(prev_pose[18][1] + math.pi / 3 * torch.tanh(self.joint_18[0][1]))))
            self.pose_tensor[18][2] =  (prev_pose[18][2] + math.pi / 3 * torch.tanh(self.joint_18[0][2]))
            
            #self.pose_tensor[4][0] =  torch.max(torch.min( prev_pose[4][0] + math.pi / 3 * torch.tanh(self.joint_4[0][0]), upper_bound_2),  lower_bound_2)
            #self.pose_tensor[4][1] =  torch.max(torch.min( prev_pose[4][1] + math.pi / 3 * torch.tanh(self.joint_4[0][1]), upper_bound_2),  lower_bound_2)
            self.pose_tensor[4][2] =  ( prev_pose[4][2] + math.pi / 9 * torch.tanh(self.joint_4[0][2]))
            
            #self.pose_tensor[17][0] =  torch.max(torch.min( prev_pose[17][0] + math.pi / 3 * torch.tanh(self.joint_4[0][0]), upper_bound_2), lower_bound_2)
            #self.pose_tensor[17][1] =  torch.max(torch.min( prev_pose[17][1] - math.pi / 3 * torch.tanh(self.joint_4[0][1]), upper_bound_2), lower_bound_2)
            self.pose_tensor[17][2] = ( prev_pose[17][2] - math.pi / 9 * torch.tanh(self.joint_4[0][2]))
            
            low_bound_x = -math.pi/9
            high_bound_x = math.pi/9

            low_bound_y = -math.pi/3
            high_bound_y = math.pi/3

            low_bound_z = -math.pi/3
            high_bound_z = math.pi/3
            
            self.pose_tensor[6][0] =    max(low_bound_x,min(high_bound_x,prev_pose[6][0]  + math.pi / 9 * torch.tanh(self.joint_6[0][0])))
            self.pose_tensor[6][1]  =   max(low_bound_y,min(high_bound_y,prev_pose[6][1]  + math.pi / 6 * torch.tanh(self.joint_6[0][1])))
            self.pose_tensor[6][2]  =   max(low_bound_z, min(high_bound_z,prev_pose[6][2]  + math.pi / 6 * torch.tanh(self.joint_6[0][2])))
            
            self.pose_tensor[19][0] =   max(low_bound_x, min(high_bound_x,prev_pose[19][0] + math.pi / 9 * torch.tanh(self.joint_19[0][0])))
            self.pose_tensor[19][1] =   max(low_bound_y, min(high_bound_y,prev_pose[19][1] + math.pi / 6 * torch.tanh(self.joint_19[0][1])))
            self.pose_tensor[19][2] =   max(low_bound_z,min(high_bound_z,prev_pose[19][2] + math.pi / 6 * torch.tanh(self.joint_19[0][2])))
           
            
            #self.pose_tensor[7][0] = ( prev_pose[7][0]   + math.pi / 3 * torch.tanh(self.joint_7[0][0]))
            low_bound_y = -math.pi/18
            high_bound_y = math.pi/18

            self.pose_tensor[7][1] = max(low_bound_y, min(high_bound_y, ( prev_pose[7][1]   + math.pi / 6 * torch.tanh(self.joint_7[0][1]))))
            self.pose_tensor[7][2] = max(( prev_pose[7][2]   + math.pi / 6 * torch.tanh(self.joint_7[0][2])),0)
            
            #self.pose_tensor[20][0] = ( prev_pose[20][0] + math.pi / 3 * torch.tanh(self.joint_20[0][0]))
            self.pose_tensor[20][1] = max(low_bound_y, min(high_bound_y, ( prev_pose[20][1] + math.pi / 6 * torch.tanh(self.joint_20[0][1]))))
            self.pose_tensor[20][2] = min((prev_pose[20][2] + math.pi / 6 * torch.tanh(self.joint_20[0][2])),0)
            
            #self.pose_tensor[15][:] = ( prev_pose[15][:] +math.pi / 18 * torch.tanh(self.joint_15))
            #self.pose_tensor[16][:] = ( prev_pose[16][:] +math.pi / 18 * torch.tanh(self.joint_16))

            self.pose_tensor[8][0] = max(0, ( prev_pose[8][0]   + math.pi / 18 * torch.tanh(self.joint_8[0][0])))
            high_bound_y = math.pi/18
            self.pose_tensor[8][1] = min(high_bound_y,( prev_pose[8][1]   + math.pi / 18 * torch.tanh(self.joint_8[0][1])))
            #self.pose_tensor[8][2] = ( prev_pose[8][2]   + math.pi / 18 * torch.tanh(self.joint_8[0][2]))
            self.pose_tensor[9][0] =  min(0, ( prev_pose[9][0]   - 0.8 * math.pi / 18 * torch.tanh(self.joint_8[0][0])))
            self.pose_tensor[10][0] = min(0, (prev_pose[10][0] - 0.5 * math.pi / 18 * torch.tanh(self.joint_8[0][0])))
            self.pose_tensor[34][0] = min(0, ( prev_pose[34][0] - 0.3 * math.pi / 18 * torch.tanh(self.joint_8[0][0])))
            
            self.pose_tensor[11][0] = min(0, ( prev_pose[11][0] + math.pi / 18 * torch.tanh(self.joint_11[0][0])))
            self.pose_tensor[11][1] = ( prev_pose[11][1] + 0.5 * math.pi / 18 * torch.tanh(self.joint_8[0][1]))
            #self.pose_tensor[11][2] = ( prev_pose[11][2] + math.pi / 18 * torch.tanh(self.joint_11[0][2]))
            self.pose_tensor[12][0] = min(0, ( prev_pose[12][0] - 0.8 * math.pi / 18 * torch.tanh(self.joint_11[0][0])))
            self.pose_tensor[13][0] = min(0, ( prev_pose[13][0] - 0.5 * math.pi / 18 * torch.tanh(self.joint_11[0][0])))
            self.pose_tensor[35][0] = min(0, ( prev_pose[35][0] - 0.3 * math.pi / 18 * torch.tanh(self.joint_11[0][0])))
            
            self.pose_tensor[14][0] = min(0, ( prev_pose[14][0] + math.pi / 18 * torch.tanh(self.joint_14[0][0])))
            self.pose_tensor[14][1] = ( prev_pose[14][1] + 0.3 * math.pi / 18 * torch.tanh(self.joint_8[0][1]))
            #self.pose_tensor[14][2] = ( prev_pose[14][2] + math.pi / 18 * torch.tanh(self.joint_14[0][2]))
            self.pose_tensor[15][0] = min(0, ( prev_pose[15][0] - 0.8 * math.pi / 18 * torch.tanh(self.joint_14[0][0])))
            self.pose_tensor[16][0] = min(0, ( prev_pose[16][0] - 0.5 * math.pi / 18 * torch.tanh(self.joint_14[0][0])))
            self.pose_tensor[36][0] = min(0, ( prev_pose[36][0] - 0.3 * math.pi / 18 * torch.tanh(self.joint_14[0][0])))
            
            self.pose_tensor[21][0] = max( 0, prev_pose[21][0] + math.pi / 18 * torch.tanh(self.joint_21[0][0]))
            high_bound_y = -math.pi/18
            self.pose_tensor[21][1] = max(high_bound_y,( prev_pose[21][1] + math.pi / 18 * torch.tanh(self.joint_21[0][1])))
            #self.pose_tensor[21][2] = ( prev_pose[21][2] + math.pi / 18 * torch.tanh(self.joint_21[0][2]))
            
            self.pose_tensor[22][0] = min(0, ( prev_pose[22][0] - 0.8 * math.pi / 18 * torch.tanh(self.joint_21[0][0])))
            self.pose_tensor[23][0] = min(0, ( prev_pose[23][0] - 0.5 * math.pi / 18 * torch.tanh(self.joint_21[0][0])))
            self.pose_tensor[37][0] = min(0, ( prev_pose[37][0] - 0.3 * math.pi / 18 * torch.tanh(self.joint_21[0][0])))
            
            self.pose_tensor[24][0] = min(0, ( prev_pose[24][0] + math.pi / 18 * torch.tanh(self.joint_24[0][0])))
            self.pose_tensor[24][1] = ( prev_pose[24][1] + 0.5 * math.pi / 18 * torch.tanh(self.joint_21[0][1]) )
            #self.pose_tensor[24][2] = ( prev_pose[24][2] + math.pi / 18 * torch.tanh(self.joint_24[0][2]) )
            self.pose_tensor[25][0] = min(0, ( prev_pose[25][0] - 0.8 * math.pi / 18 * torch.tanh(self.joint_24[0][0])))
            self.pose_tensor[26][0] = min(0, ( prev_pose[26][0] - 0.5 * math.pi / 18 * torch.tanh(self.joint_24[0][0])))
            self.pose_tensor[38][0] = min(0, ( prev_pose[38][0] - 0.5 * math.pi / 18 * torch.tanh(self.joint_24[0][0])))
            
            self.pose_tensor[27][0] = min(0, ( prev_pose[27][0] + math.pi / 18 * torch.tanh(self.joint_27[0][0])))
            self.pose_tensor[27][1] = ( prev_pose[27][1] + 0.3 * math.pi / 18 * torch.tanh(self.joint_21[0][1]))
            #self.pose_tensor[27][2] = ( prev_pose[27][2] + math.pi / 18 * torch.tanh(self.joint_27[0][2]))
            self.pose_tensor[28][0] = min(0, ( prev_pose[28][0] - 0.8 * math.pi / 18 * torch.tanh(self.joint_27[0][0])))
            self.pose_tensor[29][0] = min(0, ( prev_pose[29][0] - 0.5 * math.pi / 18 * torch.tanh(self.joint_27[0][0])))
            self.pose_tensor[39][0] = min(0, ( prev_pose[29][0] - 0.3 * math.pi / 18 * torch.tanh(self.joint_27[0][0])))
            
            self.pose_tensor[30][0] = ( prev_pose[30][0] + math.pi / 18 * torch.tanh(self.joint_30[0][0]))
            self.pose_tensor[30][2] = ( prev_pose[30][2] + math.pi / 18 * torch.tanh(self.joint_30[0][2]))
            #self.pose_tensor[31][:] = ( prev_pose[31][:] + math.pi / 18 * torch.tanh(self.joint_31))
            
            self.pose_tensor[32][0] = ( prev_pose[32][0] + math.pi / 18 * torch.tanh(self.joint_32[0][0]))
            self.pose_tensor[32][2] = ( prev_pose[32][2] + math.pi / 18 * torch.tanh(self.joint_32[0][2]))
            #self.pose_tensor[33][:] = ( prev_pose[33][:] + math.pi / 18 * torch.tanh(self.joint_33))
    
            # since the template is fully streched, some angle value can only be negative
        else: 
            if(self.template_flip == True and self.opposite_direction == True):
                self.pose_tensor[0][0] = math.pi / 2 * torch.tanh(self.joint_0[0][0])
                self.pose_tensor[0][1] = math.pi / 2 * torch.tanh(self.joint_0[0][1])
                self.pose_tensor[0][2] = math.pi / 2 * torch.tanh(self.joint_0[0][2]) + 1 * math.pi
            elif(self.template_flip == True and self.opposite_direction == False):
                self.pose_tensor[0][0] = math.pi / 2 * torch.tanh(self.joint_0[0][0]) + 1 * math.pi
                self.pose_tensor[0][1] = math.pi / 2 * torch.tanh(self.joint_0[0][1])
                self.pose_tensor[0][2] = math.pi / 2 * torch.tanh(self.joint_0[0][2])
            elif(self.template_flip == False and self.opposite_direction == True):
                self.pose_tensor[0][0] = math.pi / 2 * torch.tanh(self.joint_0[0][0]) 
                self.pose_tensor[0][1] = math.pi / 2 * torch.tanh(self.joint_0[0][1]) + 1 * math.pi
                self.pose_tensor[0][2] = math.pi / 2 * torch.tanh(self.joint_0[0][2])
            else: 
                self.pose_tensor[0][0] = math.pi / 2 * torch.tanh(self.joint_0[0][0]) 
                self.pose_tensor[0][1] = math.pi / 2 * torch.tanh(self.joint_0[0][1]) 
                self.pose_tensor[0][2] = math.pi / 2 * torch.tanh(self.joint_0[0][2])

            
            #self.pose_tensor[0][2] = math.pi / 2 * torch.tanh(self.joint_0[0][2]) +math.pi
            #self.pose_tensor[1][:] = math.pi / 9 * torch.tanh(self.joint_1)
            #self.pose_tensor[2][:] = math.pi / 9 * torch.tanh(self.joint_2)
            #self.pose_tensor[3][:] = math.pi / 18 * torch.tanh(self.joint_3)
            
            self.pose_tensor[1][0] = math.pi / 3 * torch.tanh(self.joint_1[0][0])
            #self.pose_tensor[1][0] = ( prev_pose[1][0] + math.pi / 3 * torch.tanh(self.joint_1[0][0])) 
            self.pose_tensor[2][0] = ( math.pi / 6 * torch.tanh(self.joint_2[0][0])) 
            #self.pose_tensor[3][0] = ( prev_pose[3][0] + math.pi / 18 * torch.tanh(self.joint_3[0][0]))
            
            
            #self.pose_tensor[1][:] = math.pi / 4 * torch.tanh(self.joint_1)
            #self.pose_tensor[2][:] = prev_pose[2][:] + math.pi / 9 * torch.tanh(self.joint_2)  # make the neck bone trainable(slightly)
            # for the shoulder using the symmetric deformation
            #self.pose_tensor[5][0] = (prev_pose[5][0] + math.pi / 9 * torch.tanh(self.joint_5[0][0]))
            self.pose_tensor[5][1] = (math.pi / 3 * torch.tanh(self.joint_5[0][1]))
            self.pose_tensor[5][2] =  (math.pi / 3 * torch.tanh(self.joint_5[0][2]))
            
            #self.pose_tensor[18][0] = (prev_pose[18][0] + math.pi / 9 * torch.tanh(self.joint_5[0][0]))
            self.pose_tensor[18][1] =  (math.pi / 3 * torch.tanh(self.joint_18[0][1]))
            self.pose_tensor[18][2] =  (math.pi / 3 * torch.tanh(self.joint_18[0][2]))
            
            #self.pose_tensor[4][0] =  torch.max(torch.min( prev_pose[4][0] + math.pi / 3 * torch.tanh(self.joint_4[0][0]), upper_bound_2),  lower_bound_2)
            #self.pose_tensor[4][1] =  torch.max(torch.min( prev_pose[4][1] + math.pi / 3 * torch.tanh(self.joint_4[0][1]), upper_bound_2),  lower_bound_2)
            self.pose_tensor[4][2] =  (math.pi / 9 * torch.tanh(self.joint_4[0][2]))
            
            #self.pose_tensor[17][0] =  torch.max(torch.min( prev_pose[17][0] + math.pi / 3 * torch.tanh(self.joint_4[0][0]), upper_bound_2), lower_bound_2)
            #self.pose_tensor[17][1] =  torch.max(torch.min( prev_pose[17][1] - math.pi / 3 * torch.tanh(self.joint_4[0][1]), upper_bound_2), lower_bound_2)
            self.pose_tensor[17][2] = ( -math.pi / 9 * torch.tanh(self.joint_4[0][2]))
            
            #self.pose_tensor[6][0] =    math.pi / 9 * torch.tanh(self.joint_6[0][0])
            self.pose_tensor[6][1]  =   math.pi / 6 * torch.tanh(self.joint_6[0][1])
            #self.pose_tensor[6][2]  =   math.pi / 6 * torch.tanh(self.joint_6[0][2])
            
            self.pose_tensor[19][0] =   math.pi / 9 * torch.tanh(self.joint_19[0][0])
            self.pose_tensor[19][1] =   math.pi / 6 * torch.tanh(self.joint_19[0][1])
            self.pose_tensor[19][2] =   math.pi / 6 * torch.tanh(self.joint_19[0][2])
           
            
            #self.pose_tensor[7][0] = ( prev_pose[7][0]   + math.pi / 3 * torch.tanh(self.joint_7[0][0]))
            self.pose_tensor[7][1] = (  math.pi / 6 * torch.tanh(self.joint_7[0][1]))
            self.pose_tensor[7][2] = max((  math.pi / 6 * torch.tanh(self.joint_7[0][2])),0)
            
            #self.pose_tensor[20][0] = ( prev_pose[20][0] + math.pi / 3 * torch.tanh(self.joint_20[0][0]))
            self.pose_tensor[20][1] = ( prev_pose[20][1] + math.pi / 6 * torch.tanh(self.joint_20[0][1]))
            self.pose_tensor[20][2] = min(( math.pi / 6 * torch.tanh(self.joint_20[0][2])),0)
            
            #self.pose_tensor[15][:] = ( prev_pose[15][:] +math.pi / 18 * torch.tanh(self.joint_15))
            #self.pose_tensor[16][:] = ( prev_pose[16][:] +math.pi / 18 * torch.tanh(self.joint_16))
            
            self.pose_tensor[8][0] = (  math.pi / 18 * torch.tanh(self.joint_8[0][0]))
            self.pose_tensor[8][1] = (  math.pi / 18 * torch.tanh(self.joint_8[0][1]))
            #self.pose_tensor[8][2] = ( prev_pose[8][2]   + math.pi / 18 * torch.tanh(self.joint_8[0][2]))
            self.pose_tensor[9][0] =  (    - 0.8 * math.pi / 18 * torch.tanh(self.joint_8[0][0]))
            self.pose_tensor[10][0] = (  - 0.5 * math.pi / 18 * torch.tanh(self.joint_8[0][0]))
            self.pose_tensor[34][0] = (  - 0.3 * math.pi / 18 * torch.tanh(self.joint_8[0][0]))
            
            self.pose_tensor[11][0] = (  math.pi / 18 * torch.tanh(self.joint_11[0][0]))
            self.pose_tensor[11][1] = (  0.5 * math.pi / 18 * torch.tanh(self.joint_8[0][1]))
            #self.pose_tensor[11][2] = ( prev_pose[11][2] + math.pi / 18 * torch.tanh(self.joint_11[0][2]))
            self.pose_tensor[12][0] = (  - 0.8 * math.pi / 18 * torch.tanh(self.joint_11[0][0]))
            self.pose_tensor[13][0] = (  - 0.5 * math.pi / 18 * torch.tanh(self.joint_11[0][0]))
            self.pose_tensor[35][0] = (  - 0.3 * math.pi / 18 * torch.tanh(self.joint_11[0][0]))
            
            self.pose_tensor[14][0] = (  + math.pi / 18 * torch.tanh(self.joint_14[0][0]))
            self.pose_tensor[14][1] = (  + 0.3 * math.pi / 18 * torch.tanh(self.joint_8[0][1]))
            #self.pose_tensor[14][2] = ( prev_pose[14][2] + math.pi / 18 * torch.tanh(self.joint_14[0][2]))
            self.pose_tensor[15][0] = (  - 0.8 * math.pi / 18 * torch.tanh(self.joint_14[0][0]))
            self.pose_tensor[16][0] = (  - 0.5 * math.pi / 18 * torch.tanh(self.joint_14[0][0]))
            self.pose_tensor[36][0] = (  - 0.3 * math.pi / 18 * torch.tanh(self.joint_14[0][0]))
            
            self.pose_tensor[21][0] = (   math.pi / 18 * torch.tanh(self.joint_21[0][0]))
            self.pose_tensor[21][1] = (   math.pi / 18 * torch.tanh(self.joint_21[0][1]))
            #self.pose_tensor[21][2] = ( prev_pose[21][2] + math.pi / 18 * torch.tanh(self.joint_21[0][2]))
            
            self.pose_tensor[22][0] = ( - 0.8 * math.pi / 18 * torch.tanh(self.joint_21[0][0]))
            self.pose_tensor[23][0] = (  - 0.5 * math.pi / 18 * torch.tanh(self.joint_21[0][0]))
            self.pose_tensor[37][0] = (  - 0.3 * math.pi / 18 * torch.tanh(self.joint_21[0][0]))
            
            self.pose_tensor[24][0] = (  math.pi / 18 * torch.tanh(self.joint_24[0][0]) )
            self.pose_tensor[24][1] = (  0.5 * math.pi / 18 * torch.tanh(self.joint_21[0][1]) )
            #self.pose_tensor[24][2] = ( prev_pose[24][2] + math.pi / 18 * torch.tanh(self.joint_24[0][2]) )
            self.pose_tensor[25][0] = (  - 0.8 * math.pi / 18 * torch.tanh(self.joint_24[0][0]))
            self.pose_tensor[26][0] = (  - 0.5 * math.pi / 18 * torch.tanh(self.joint_24[0][0]))
            self.pose_tensor[38][0] = (  - 0.5 * math.pi / 18 * torch.tanh(self.joint_24[0][0]))
            
            self.pose_tensor[27][0] = (  math.pi / 18 * torch.tanh(self.joint_27[0][0]))
            self.pose_tensor[27][1] = (  0.3 * math.pi / 18 * torch.tanh(self.joint_21[0][1]))
            #self.pose_tensor[27][2] = ( prev_pose[27][2] + math.pi / 18 * torch.tanh(self.joint_27[0][2]))
            self.pose_tensor[28][0] = (  - 0.8 * math.pi / 18 * torch.tanh(self.joint_27[0][0]))
            self.pose_tensor[29][0] = (  - 0.5 * math.pi / 18 * torch.tanh(self.joint_27[0][0]))
            self.pose_tensor[39][0] = (  - 0.3 * math.pi / 18 * torch.tanh(self.joint_27[0][0]))
            
            self.pose_tensor[30][0] = (  math.pi / 18 * torch.tanh(self.joint_30[0][0]))
            self.pose_tensor[30][2] = (  math.pi / 18 * torch.tanh(self.joint_30[0][2]))
            #self.pose_tensor[31][:] = ( prev_pose[31][:] + math.pi / 18 * torch.tanh(self.joint_31))
            
            self.pose_tensor[32][0] = (  math.pi / 18 * torch.tanh(self.joint_32[0][0]))
            self.pose_tensor[32][2] = (  math.pi / 18 * torch.tanh(self.joint_32[0][2]))
            #self.pose_tensor[33][:] = ( prev_pose[33][:] + math.pi / 18 * torch.tanh(self.joint_33))
    
           
            
        #self.pose_tensor[1] = self.joint_0
        
        #self.pose_tensor = self.pose_tensor.unsqueeze(0)
        # model will deform the mesh and then add the predetermined offset and learned displacement
        # apply the small adjustment on template first
        template_default_scale = 0.0035#8* 1.53
        displacement_range = 0.1
        vertex_displacement_range = 0.1
        

        vertices = self.vertices  + vertex_displacement_range * torch.tanh(self.local_adjust.cuda())

        


    
        vertices, joints = self.LBS_model(vertices,self.joints, self.pose_tensor, to_rotmats=True)
        
        
        _, joints_tail = self.LBS_model(vertices,self.joints_tail, self.pose_tensor, to_rotmats=True)
        #self.pose_tensor = self.pose_tensor.squeeze()
        
        estimated_location = estimated_location.unsqueeze(dim = 1)
        
       
        
        vertices =  template_default_scale * self.scale * vertices + estimated_location[0].repeat(1, self.vertices_number, 1).cuda() + displacement_range * torch.tanh(self.displacement.repeat(1, self.vertices_number, 1)).cuda() 
        joints =   template_default_scale* self.scale * joints + estimated_location[0].repeat(1, self.joint_number, 1).cuda() + displacement_range * torch.tanh(self.displacement.repeat(1, self.joint_number, 1)).cuda() 
        
        
        joints_tail =  template_default_scale * self.scale * joints_tail + estimated_location[0].repeat(1, self.joint_number, 1).cuda() + displacement_range * torch.tanh(self.displacement.repeat(1, self.joint_number, 1)).cuda() 
        #self.vertices = vertices #+ self.random_dis.repeat(1, self.vertices_number, 1).cuda()
        
        #self.joints = joints #+ self.random_dis.repeat(1, self.joint_number, 1).cuda()
        
        

        
        
        #return
        #np.savetxt('./test_2.txt', verts.detach().cpu().numpy()[0])
        
        # apply Laplacian and flatten geometry constraints
        
        #flatten_loss = self.flatten_loss(vertices).mean()
        # add l2 regularization for small wing bones
        skining_matrix_adjust_l2 = torch.norm(self.skining_adjust)
        # define the return package, including the pose_tensor, vertices, faces, joints location, initial displacement, and local positon adjustment
        
        
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
        
        
        bone_prior_1 = 0.1 * (torch.norm(self.pose_tensor[4][2]) +torch.norm(self.pose_tensor[17][2]) + torch.norm(self.pose_tensor[5])+torch.norm(self.pose_tensor[18]))
        
                                                                                                           
        bone_prior_2 = 0.1 * (torch.norm(self.pose_tensor[6]) + torch.norm(self.pose_tensor[19]) + 
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
                        
                        
          
                          
        #bone_symmetric = 0.1 * bone_symmetric_1 + 0.3 * bone_symmetric_2 + 0.5 * bone_symmetric_3
        bone_symmetric = 0.5 * bone_symmetric_1 + 0.5* bone_symmetric_2 + 0.01 * bone_symmetric_3
        
        
        
        laplacian_loss = self.laplacian_smoothing(vertices).mean()
        return sr.Mesh(vertices.repeat(batch_size, 1, 1),self.faces.repeat(batch_size, 1, 1)), \
                       laplacian_loss, \
                       reg, \
                       bone_prior, \
                       bone_symmetric, \
                       self.pose_tensor, \
                       self.scale,       \
                       self.displacement, \
                       vertex_displacement_range * torch.tanh(self.local_adjust), \
                       vertices,            \
                       joints,              \
                       joints_tail,         \
                       estimated_location[0] +displacement_range * torch.tanh(self.displacement), \
                       self.training_skining_weight
                       
    def render_original(self,location,current_pose):

            
        template_default_scale = 0.005/1.349#8* 1.53  #blender precision difference
     
        vertices = self.vertices

        
        vertices, joints = self.LBS_model(vertices,self.joints, current_pose, to_rotmats=True)
        
        
        _, joints_tail = self.LBS_model(vertices,self.joints_tail, current_pose, to_rotmats=True)
        #self.pose_tensor = self.pose_tensor.squeeze()
        
        estimated_location = location.unsqueeze(dim = 1)
        
      
        vertices =  template_default_scale * vertices + estimated_location[0].repeat(1, self.vertices_number, 1).cuda()  
        joints =   template_default_scale  * joints + estimated_location[0].repeat(1, self.joint_number, 1).cuda() 
        
        
        return sr.Mesh(vertices.repeat(1, 1, 1),self.faces.repeat(1, 1, 1))
                          
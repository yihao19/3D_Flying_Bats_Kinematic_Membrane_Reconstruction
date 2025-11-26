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
from kinematic_model import Kinematic_model
'''
trying to make the model learning euler angle and displacement
by using camera matrix

'''
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, './')

'''
define the dataset, it contains the silhouette images 
and the camera matrix

camera_list: passing the camera index to control the number of passed camera
and angles: only read the camera meta that the silouette contains part of the bat

'''
class image_dataset(Dataset):
    def __init__(self, camera_meta_path, camera_list_path, silouette_image_path, current_pose, use_previous):
        self.camera_meta_path = camera_meta_path
        self.camera_list_path = camera_list_path
        self.silouette_image_path = silouette_image_path
    
        self.use_previous = use_previous
        self.current_pose = current_pose
        # read the file
        camera_list_file = os.path.join(self.camera_list_path, "camera.txt")
        camera_list= []
        camera_list_file = open(camera_list_file)
        camera_list_string = camera_list_file.read()
        camera_list_string = camera_list_string[1: len(camera_list_string)-1]
        camera_list = camera_list_string.split(', ')
       
        if(len(camera_list) <= 0):
            raise Exception("The number of silouettee image is: {}, not enough images!".format(len(camera_list)))

       
        
        entire_camera_matrix = np.loadtxt(os.path.join(self.camera_meta_path, 'camera_meta.txt'))
        self.image_list = []  # store the image path
        
       
        counter = 0
        glitch_camera_index = []#['2', '6', '11', '12', '13']  # glithes camera index that need to get rid of
        camera_list = set(camera_list).difference(set(glitch_camera_index))
        self.camera_number = len(camera_list)
        camera_matrix = np.zeros((self.camera_number, 12))
        print("camera list: ", camera_list)
        for index in camera_list:
            index = int(index)
            image_name = "camera{}.png".format(index)
            self.image_list.append(image_name)
            camera_matrix[counter] = entire_camera_matrix[index-1]
            counter += 1
        camera_matrix = np.reshape(camera_matrix, (self.camera_number, 3, 4))
        self.camera_matrix = camera_matrix # camera matrix initialization
        
    def __len__(self):
        return self.camera_number
    def __getitem__(self, idx):
        # return the data sample indicated by the passed index

        index = idx % self.camera_number 

        camera_matrix = self.camera_matrix[index]
        
        image_path = os.path.join(self.silouette_image_path, self.image_list[index])
        
        mask_image = cv.imread(image_path).astype('float32')[:, :, 0] / 255.
        mask_image = np.expand_dims(mask_image, -1)
        #cv.imwrite("test.png",255 * mask_image)
        mask_image = mask_image.transpose((2, 0, 1))
        
        
        if(self.use_previous == False):
            prev_pose = 0
            pre_local_adjust = 0
            estimated_location_file = open(os.path.join(self.silouette_image_path, 'estimated_location.txt'))
            estimated_location_string = estimated_location_file.read()
            parts = estimated_location_string.split(' ')
            x_average = float(parts[0])
            y_average = float(parts[1])
            z_average = float(parts[2])
            estimated_location = np.array([x_average, y_average, z_average]).astype('float32') # randomly assign offset for

        else:
            prev_pose_index = self.current_pose - 1
            current_path = Path(self.silouette_image_path)
            curr_root = current_path.parent
            prev_output_path = os.path.join(curr_root, str(prev_pose_index), "output.json")
            file = open(prev_output_path)
            prev_output = json.load(file) 
            prev_pose = np.array(prev_output['pose'])
            pre_local_adjust = np.array(prev_output['local_adjust'])
            estimated_location = np.array(prev_output['template_displacement']).astype('float32')
        sample = {'mask': mask_image, 
                  'camera_matrix':camera_matrix.astype('float32'), 
                  'prev_pose':prev_pose, 
                  'pre_local_adjust': pre_local_adjust,
                  'estimated_location':estimated_location}
        return sample
    
'''
param: template_obj_path: path for template of model in rest pose(obj file)
       bone_skining_matrix_path: path for self-designed bone and default skining_matrix
       joint_list: determine which bone's rotation matrix that you want to trained to get
       train_skining: determine whether you want to train the skining matrix or using 
                      default matrix as hyper-params
'''

'''
IOU loss define the  
'''                  
def neg_iou_loss(predict, target):
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + 1e-6
    return 1. - (intersect / union ).sum() / intersect.nelement()


def main(project_name, test_name, passed_pose_index, epoch, use_previous, opposite_direction):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--template-mesh', type=str,
                        default=os.path.join(data_dir, './aero_dynamic_template.obj'))
    parser.add_argument('-o', '--image-output-dir', type=str,
                        default=os.path.join(data_dir, f'E:/{project_name}/{test_name}/temp/'))
    args = parser.parse_args()
    
    # make the model data loader style of input 
    os.makedirs(args.image_output_dir, exist_ok=True)

 
    # start 
    pose_index = passed_pose_index
    image_size = (1024,1280)
    output_path = 'D:/{}/{}/rearrange_pose/{}/'.format(project_name, test_name, pose_index)
    #camera_list = [1]
    camera_meta_path = 'D:/{}/{}/rearrange_pose/'.format(project_name, test_name)
    camera_list_path = 'D:/{}/{}/rearrange_pose/{}/'.format(project_name, test_name, pose_index)
    silouettee_image_path = 'D:/{}/{}/rearrange_pose/{}/'.format(project_name, test_name, pose_index)
    #estimated_location_file = 'G:/PhDProject_real_data/{}/rearrange_pose/{}/estimated_location.txt'.format(test_name, pose_index)
    args.output_dir  = 'D:/{}/{}/reconstruction/'.format(project_name, test_name)
    #args.image_output_dir = 'G:/PhDProject_real_data/{}/reconstruction/'.format(test_name)
    current_pose = pose_index
    # if use_previous == True
    # load the previous pose matrix as a starting point for the current pose reconstruction
    dataset = image_dataset(camera_meta_path, camera_list_path, silouettee_image_path, current_pose, use_previous)
    
    batch_size =dataset.camera_number 
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    
    #return

    model = Kinematic_model(args.template_mesh, use_previous=use_previous, opposite_direction=opposite_direction).cuda()
    

    optimizer = torch.optim.Adam(model.parameters(), 0.005,betas=(0.5, 0.99))

    #renderer.transform.set_eyes_from_angles(camera_distances, elevations, viewpoints)

    epoch = tqdm.tqdm(list(range(0,epoch)))
    for i in epoch:
        
        for training_sample in train_dataloader:
      
            images_gt = training_sample['mask'].cuda()
            camera_matrix = training_sample['camera_matrix'].cuda()
            prev_pose = training_sample['prev_pose'].cuda()
            prev_local_adjust = training_sample['pre_local_adjust'].cuda()
            estimated_location = training_sample['estimated_location'].cuda()
            #images_gt = torch.from_numpy(images).cuda()
            # at the begining, train the model orientation first
            
            if(i >= 0):
                
                for name, param in model.named_parameters():
                        param.requires_grad = True
                        
            mesh, laplacian_loss,wing_tip_reg, bone_prior, bone_symmetric, current_pose, scale, displacement, local_adjust, vertices, joints, joints_tail, displacement,weight_tensor = model(batch_size, estimated_location, use_previous, prev_pose[0])
            
            
            renderer = sr.SoftRenderer(image_height=image_size[0], image_width=image_size[1],sigma_val=1e-6,
                                       camera_mode='projection', P = camera_matrix ,orig_height=image_size[0], orig_width=image_size[1], 
                                       near=0, far=100)
            
            # check the mesh vertices and the projection          
            images_pred = renderer.render_mesh(mesh)
            #print("image_pred shape: ", images_pred.shape)
            # optimize mesh with silhouette reprojection error and
            # geometry constraints
            # silhouette image predicted will in the 4th element of the vector 
            #print("pred_image shape: ", images_pred.shape)
            IOU_loss = neg_iou_loss(images_pred[:, -1], images_gt[:, 0])
            loss = IOU_loss #+ 1 * l2_norm 
            #print("Laplacian_loss: ", 5 * laplacian_loss)
            pose_loss = torch.tensor(0)
            l2_adjust = torch.tensor(0)
            if(use_previous == True):
                # only the body orientation is considered
                pose_loss = 0.005 * torch.norm(current_pose[:][:] - prev_pose[:][:]) #+ 0.1 * torch.norm(current_pose[1:][:] - prev_pose[1:][:])
                l2_adjust = 0.02 * torch.norm(local_adjust - prev_local_adjust)
            
            
           
            loss = IOU_loss + pose_loss  + 0 * laplacian_loss + 0.00 * wing_tip_reg  + 0.005 * bone_prior + 0.005 * bone_symmetric + 0*l2_adjust
            epoch.set_description('IOU Loss: %.4f   Pose Loss: %.4f  Wingtip_reg: %.4f  Bone prior: %.4f  Bone symmetry: %.4f  L2 adjust: %.4f' % (IOU_loss.item(),pose_loss.item(), wing_tip_reg.item(), 0.1 * bone_prior.item(), bone_symmetric.item(), l2_adjust.item()))
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        
       
        
        if i % 1 == 0:
            #print("pred_image shape: ",images_pred.detach().cpu().numpy()[0].shape )
            #image = images_pred.detach().cpu().numpy()[1].transpose((1 , 2, 0))
           
            for counter in range(1):
            
                image = images_pred.detach().cpu().numpy()[counter].transpose((1 , 2, 0))
                imageio.imsave(os.path.join(args.image_output_dir, 'pred_camera_{}_{}.png'.format(counter, 2)), (255*image[..., 1]).astype(np.uint8))
            image_gt = images_gt.detach().cpu().numpy()[counter].transpose((1, 2, 0))
        
      
    output_mesh,laplacian_loss, wing_tip_reg,bone_prior, bone_symmetric, current_pose, current_scale, current_displacement, local_adjust, vertices, joints, joints_tail,  displacement,weight_tensor = model(1, estimated_location,use_previous, prev_pose[0])
  
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
    output_mesh.save_obj(os.path.join(args.output_dir, '{}_bat_{}.obj'.format(test_name, pose_index)), save_texture=False)
    
    
    #n_digits = 4
    
    
    #print(output_skining_weight.shape)
    #print("Displacement: ", template_displacement)
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
    with open(os.path.join(silouettee_image_path, "output.json"), "w") as output:
        json.dump(save_dict, output)
        


if __name__ == '__main__':
    start_pose =232
    end_pose = 232
    epoch = 100
    interval = 1
    project_name = "PhDProject_real_data"
    test_name = "brunei_2023_bat_test_13_2"
    for pose_index in range(start_pose, end_pose + interval, interval):
        print("working on pose: ", pose_index)

        # developing the use_previous to provide extra supervision
        main(project_name, test_name, pose_index, epoch, use_previous = False, opposite_direction =False)
    
    
    
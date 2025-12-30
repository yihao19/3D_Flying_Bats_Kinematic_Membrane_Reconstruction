import scipy.io as sio
import numpy as np
import os # to manage file path

# Assuming 'data.mat' was created in MATLAB and is in the current directory
mat_contents = sio.loadmat('./calibration_mat/2to4_June_24.mat')

# loadmat returns a dictionary. The keys are the variable names saved in MATLAB.
# 'my_variable' is the name you used when saving in MATLAB

class ArrayRectify(): 
    def __init__(self,
                 calibration_file:str="./calibration_mat/2to4_June_24.mat",
                 ground_camera_index:list = [2, 3, 10, 11],
                 y_forward:list = [10,2],
                 x_forward:list = [10, 11]):
        #for 2-4,21,22,23,24,25,31,32,35,41,42,43,44,45, ground: 23, 24, 43, 44 (2, 3, 10, 11)
        #for 5-7,52,53,54,55,61,64,65,71,72,73,74,75, ground: 53, 54, 64, 73, 74 (1 ,2 ,5 ,9 ,10)
        self.calibration_file = calibration_file
        self.ground_camera_index = ground_camera_index
        self.y_forward = y_forward
        self.x_forward = x_forward
        self.mat_contents = sio.loadmat(self.calibration_file)      
        self.mat_contents = sio.loadmat(self.calibration_file)
        self.camera_locs = np.array(self.mat_contents['Ce']).T
        return
    
    def rectifying(self): 
        """Figure out the R|T for rectifying the camera array so that gravity is -z and forward is y
        """
        x_forward_vector = (np.array(self.camera_locs[self.x_forward[1]]) - np.array(self.camera_locs[self.x_forward[0]]))/np.linalg.norm(np.array(self.camera_locs[self.x_forward[1]]) - np.array(self.camera_locs[self.x_forward[0]]))
        y_forward_vector = (np.array(self.camera_locs[self.y_forward[1]]) - np.array(self.camera_locs[self.y_forward[0]]))/np.linalg.norm(np.array(self.camera_locs[self.y_forward[1]]) - np.array(self.camera_locs[self.y_forward[0]]))

        up= np.cross(x_forward_vector, y_forward_vector)
        z_up = np.array([0,0,1])
        axis = np.cross(up, z_up)
        axis_norm = np.linalg.norm(axis)

        if axis_norm < 1e-8:
            # up is already aligned or opposite to z_up
            if np.dot(up, z_up) > 0:
                R = np.eye(3)
            else:
                # 180° rotation around any perpendicular axis
                R = np.array([
                    [1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]
                ])
        else:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(up, z_up), -1, 1))

            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])

            R = (
                np.eye(3)
                + np.sin(angle) * K
                + (1 - np.cos(angle)) * (K @ K)
            )
        print(R)
        return
    
if __name__=="__main__":
    array_rectifier = ArrayRectify()
    array_rectifier.rectifying()

    
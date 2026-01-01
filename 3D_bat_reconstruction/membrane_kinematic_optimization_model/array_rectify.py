import scipy.io as sio
import numpy as np
import os # to manage file path
import matplotlib.pyplot as plt
# Assuming 'data.mat' was created in MATLAB and is in the current directory


# loadmat returns a dictionary. The keys are the variable names saved in MATLAB.
# 'my_variable' is the name you used when saving in MATLAB

class ArrayRectify(): 
    def __init__(self,name = "2_4"):
        #for 2-4,21,22,23,24,25,31,32,35,41,42,43,44,45, ground: 23, 24, 43, 44 (2, 3, 10, 11)
        #for 5-7,52,53,54,55,61,64,65,71,72,73,74,75, ground: 53, 54, 64, 73, 74 (1 ,2 ,5 ,9 ,10)
        self.name = name
        if(self.name == "2_4"):
            self.calibration_file = "../../calibration_mat/2to4_June_24.mat"
            self.camera_names = [21,22,23,24,25,31,32,35,41,42,43,44,45]
            self.ground_camera_index = [2, 3, 10, 11]
            self.y_forward:list = [2,10]
            self.x_forward:list = [10,11]
            self.scaler = 4.80
            self.reference_camera_index = 2
        elif(self.name == "5_7"):
            self.calibration_file = "../../calibration_mat/5to7.mat"
            self.camera_names = [52,53,54,55,61,64,65,71,72,73,74,75]
            self.ground_camera_index = [1,2, 5, 9, 10]
            self.y_forward:list = [9,1]
            self.x_forward:list = [10,9]
            self.scaler = 1.32
            self.reference_camera_index = 2
        elif(self.name == "2023"):
            pass
        self.mat_contents = sio.loadmat(self.calibration_file)      
        self.camera_locs = np.array(self.mat_contents['in']['Ce'][0][0]).T
       
    def camera_loc_rectify(self, tf:np.ndarray=np.eye(4), if_plot:bool=False):
        
        ones = np.ones((self.camera_locs.shape[0], 1))
        camera_loc_rectified = (tf @ np.hstack([self.camera_locs, ones]).T).T
        x = self.scaler * (camera_loc_rectified[:, 0] - camera_loc_rectified[self.reference_camera_index, 0])
        y = self.scaler * (camera_loc_rectified[:, 1] - camera_loc_rectified[self.reference_camera_index, 1])
        z = self.scaler * (camera_loc_rectified[:, 2] - camera_loc_rectified[self.reference_camera_index, 2])
        if if_plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d') # or use plt.axes(projection='3d')
        # 3. Create the 3D plot
           
            ax.scatter(x, y, z, c='b', marker='o',s=10) # 'c' for color, 'marker' for style
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            for i in range(len(camera_loc_rectified)):
                ax.text(x[i], y[i], z[i], f'{self.camera_names[i]}')
            #ax.view_init(elev=0, azim=90)
            plt.savefig(f"calibration_rectified_{self.name}.svg")
            plt.close()

        #print(np.linalg.norm(camera_loc_rectified[6] - camera_loc_rectified[3]))
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = tf[:3,:3]

        transformation_matrix[:3, 3] = [-camera_loc_rectified[self.reference_camera_index, 0],-camera_loc_rectified[self.reference_camera_index, 1],-camera_loc_rectified[self.reference_camera_index, 2]]
        return  self.scaler * transformation_matrix, [x, y, z]
    
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
        homo_tf = np.eye(4)
        homo_tf[:3,:3] = R
        transformation, rectified_camera_locs = self.camera_loc_rectify(homo_tf, if_plot=False)

        return transformation, rectified_camera_locs,self.camera_names 
        #return transformation, rectified_camera_locs,self.camera_names 
    
if __name__=="__main__":
    array_rectifier = ArrayRectify("5_7")
    array_rectifier.rectifying()

    
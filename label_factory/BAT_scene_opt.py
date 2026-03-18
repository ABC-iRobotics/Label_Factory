import os
import cv2
import yaml
import time
import random
import requests
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
from matplotlib.widgets import Button
from matplotlib.backend_bases import MouseButton
from scipy.spatial.transform import Rotation as R
from label_factory.Pose_optim_tool import QtPoseTool
from label_factory.utils import *

import sys
from PyQt5 import QtCore, QtGui, QtWidgets

class Image_data():
    def __init__(self,image_index,image_blender,image_real,Px_coords,distance_from_cam,World_coords,Faces):
        self.image_index = image_index
        self.image_blender = image_blender
        self.image_real = image_real
        self.Px_coords = Px_coords
        self.distance_from_cam = distance_from_cam
        self.World_coords = World_coords
        self.Faces = Faces

class BAT_Optimizer():
    def __init__(self):
        self.Width = 72
        self.path = subprocess.check_output("ros2 pkg prefix label_factory",shell = True, text = True)
        self.path = self.path.split("/install",1)[0]
        # Open config
        with open(os.path.join(self.path,'label_factory/config.yaml'), 'r') as file:
                self.config = yaml.safe_load(file)
        # Set render image path
        self.render_path = self.config['path_to_rendered_img']        
        if not os.path.exists(self.path + '/label_factory/Calibration/Blender_Images'):
            os.makedirs(self.path + '/label_factory/Calibration/Blender_Images')
        for i in self.config['camera_indexes']:
            if not os.path.exists(self.path + '/label_factory/Calibration/Blender_Images/camera_' + str(i)):
                os.makedirs(self.path + '/label_factory/Calibration/Blender_Images/camera_' + str(i))

    def Define_cam_poses(self):
        if os.path.isfile(self.path + "/label_factory/Calibration/robot_poses.csv"):
            # Read robot poses
            self.robot_poses = np.loadtxt(self.path + "/label_factory/Calibration/robot_poses.csv", delimiter=",", dtype=float)
        else:
            print(closed_text("The robot_poses.csv is not found!",self.Width,"red","left"))
            return False
            
        # Create empty arrays for the poses
        self.camera_poses = np.zeros((len(self.config['camera_indexes']),self.robot_poses.shape[0],6))
        self.camera_pose_matrix = np.zeros((len(self.config['camera_indexes']),self.robot_poses.shape[0],4,4))
        
        # Calculate camera poses in matrix and in [position, euler_angles(xyz)] form 
        for i in range(len(self.config['camera_indexes'])):
            camera_trafs = np.reshape(np.array(self.config['cameras']['camera_' + str(self.config['camera_indexes'][i])]['Tmatrix_Flange_Camera']),(4,4))
            for j in range(self.robot_poses.shape[0]):
                robot_ori = R.from_quat(self.robot_poses[j,3:]).as_matrix() 
                robot_posi = np.array([[self.robot_poses[j,0]],[self.robot_poses[j,1]],[self.robot_poses[j,2]]])
                robot_mat = np.append(np.append(robot_ori,robot_posi,axis=1),np.array([[0.0,0.0,0.0,1.0]]),axis=0)
                Rotate_x_180 = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                cam_mat = np.matmul(np.matmul(robot_mat,camera_trafs),Rotate_x_180)
                self.camera_pose_matrix[i,j,:,:] = cam_mat
                self.camera_poses[i,j,:] = np.append(cam_mat[0:3,3],R.from_matrix(cam_mat[0:3,0:3]).as_euler('xyz',degrees=False))
        return True
        
    def Setup_camera(self,camera_index = 0):
        try:
            requests.get('http://localhost:12345/frame')
        except:
            print(closed_text("The Blender HTTP Remote Interface is not found!",self.Width,"red","left"))
            return False
        
        self.camera_index = self.config['camera_indexes'][int(camera_index)]
        Content_type = {'Content-Type': 'application/json',}
        setup_camera = {
        'camera': {
            'sensor_width': self.config['sensor_width_' + str(self.camera_index)],
            'fx':self.config['cameras']['camera_' + str(self.camera_index)]['CameraIntrinsic'][0],
            'fy':self.config['cameras']['camera_' + str(self.camera_index)]['CameraIntrinsic'][4],
            'cx':self.config['cameras']['camera_' + str(self.camera_index)]['CameraIntrinsic'][2],
            'cy':self.config['cameras']['camera_' + str(self.camera_index)]['CameraIntrinsic'][5],
            'k1': self.config['cameras']['camera_' + str(self.camera_index)]['DistCoeffs'][0],
            'k2': self.config['cameras']['camera_' + str(self.camera_index)]['DistCoeffs'][1],
            'p1': self.config['cameras']['camera_' + str(self.camera_index)]['DistCoeffs'][2],
            'p2': self.config['cameras']['camera_' + str(self.camera_index)]['DistCoeffs'][3],
            'k3': self.config['cameras']['camera_' + str(self.camera_index)]['DistCoeffs'][4],
            'k4': 0
            },
        }
        requests.post('http://localhost:12345', headers=Content_type, json=setup_camera)
        time.sleep(0.5)
        return True
    
    def Set_Object_pose(self,object_name):
        self.object_name = object_name
        try:
            answer =  requests.get('http://localhost:12345/object?name=' + str(object_name))
            print(answer)
            if answer.json()['status'] == "failed":
                print(closed_text("The object is found! Its name: " + str(object_name) + ".",self.Width,"red","left"))
                return False
            else:
                try: 
                    self.object_pose = self.config['object_pose_' + object_name]
                except:
                    print(closed_text("The object pose is not defined in the config file!",self.Width,"yellow","left"))
                    loc = answer.json()['location']
                    rot = [angles * np.pi / 180 for angles in answer.json()['rotation']]
                    update_pos = {'location' : loc}
                    update_ori = {'rotation' : rot} 
                    if ('object_pose_' + self.object_name) not in self.config:
                        self.config['object_pose_' + self.object_name] = {}
                    self.config['object_pose_' + self.object_name].update(update_pos)
                    self.config['object_pose_' + self.object_name].update(update_ori)
                    with open(self.path + '/label_factory/config.yaml','w') as yamlfile:
                        yaml.safe_dump(self.config, yamlfile, default_flow_style = False)
                    print(closed_text("The object's pose is defined by the blender scene, the config file is updated!",self.Width,"yellow","left"))
        except:
            print(closed_text("The object is found! Its name: " + str(object_name) + ".",self.Width,"red","left"))
            return False


        Content_type = {'Content-Type': 'application/json',}
        object_pose = {
        'pose': {
            'name': object_name,
            'location': self.config['object_pose_' + object_name]['location'],
            'rotation': self.config['object_pose_'+  object_name]['rotation'],
            }
        }
        requests.post('http://localhost:12345', headers=Content_type, json=object_pose)
        return True

    def Render_images(self, Object_name = "object", img_number = 10):
        if os.path.isfile(self.render_path):
            os.remove(self.render_path) 
        self.All_image_datas = []
        Content_type = {'Content-Type': 'application/json',}
        render = {'render': {'render': True,},}
        if int(img_number) >= self.camera_poses.shape[1]:
            img_number = self.camera_poses.shape[1]


        rand_imgs = range(0,self.camera_poses.shape[1],int(self.camera_poses.shape[1]/int(img_number)))


        self.camera_pose_mtx = self.camera_pose_matrix[self.config['camera_indexes'].index(self.camera_index),rand_imgs,:,:]
        for i in rand_imgs:
            camera_pose = {
                'pose': {
                    'name': 'Camera',
                    'location': self.camera_poses[self.config['camera_indexes'].index(self.camera_index),i,:3].tolist(),
                    'rotation': self.camera_poses[self.config['camera_indexes'].index(self.camera_index),i,3:].tolist(),
                }
            }
            requests.post('http://localhost:12345', headers=Content_type, json=camera_pose)
            requests.post('http://localhost:12345', headers=Content_type, json=render)
            while True:
                try: 
                    if os.path.isfile(self.render_path):
                        time.sleep(0.3)
                        im_blender = cv2.imread(self.render_path)
                        cv2.imwrite(self.path + '/label_factory/Calibration/Blender_Images/camera_' + str(self.camera_index) + "/" + str(rand_imgs.index(i)).zfill(4) + ".png", im_blender)
                        time.sleep(0.1)
                        os.remove(self.render_path) 
                        break
                    else:
                        time.sleep(0.1)
                except:
                    time.sleep(0.1)
            resp =  requests.get('http://localhost:12345/mesh?name='+ Object_name).json()
            while True:
                if resp['status'] == 'success':
                    break
                else:
                    time.sleep(0.1)
            World_coords = np.array(resp['vertices'])
            Faces = np.array(resp['faces'])
            Px_coords = np.array(resp['px_coords'])
            Distances = np.array(resp['dist'])
            im_real = cv2.imread(self.path + "/label_factory/Calibration/Real_Images/camera_" + str(self.config['camera_indexes'][self.config['camera_indexes'].index(self.camera_index)]) + "/" + str(i).zfill(4) + ".png", cv2.IMREAD_COLOR)
            current_image_data = Image_data(i, im_blender, im_real, Px_coords, Distances, World_coords, Faces)           
            self.All_image_datas.append(current_image_data)

    def Create_window(self):
        app_qt = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        w = QtPoseTool(self.All_image_datas, self.config, self.path, self.camera_pose_mtx, self.camera_index, self.object_name, self.Width)
        w.show()
        app_qt.exec_()

    def Colorize_labels(self, mask, background = 0, bgr = True):
        """
        Convert a 2D label mask to a color image with distinct colors per label.
        - mask: 2D array of ints (uint8/uint16/etc). Label 0 stays black.
        - background: label value to render as black (default 0).
        - bgr: return BGR (OpenCV default) if True, else RGB.

        Returns: HxWx3 uint8 image.
        """
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D array")
        lab = mask.astype(np.uint32)

        # Collect unique labels excluding background
        labels = np.unique(lab)
        labels = labels[labels != background]
        h, w = lab.shape

        # Early exit: only background
        out = np.zeros((h, w, 3), dtype=np.uint8)
        if labels.size == 0:
            return out if bgr else out[..., ::-1]

        # Generate distinct colors via evenly spaced HSV -> BGR
        n = labels.size
        hues = np.linspace(0, 179, n, endpoint=False, dtype=np.uint8)  # OpenCV hue range
        hsv = np.stack([hues, np.full(n, 200, np.uint8), np.full(n, 255, np.uint8)], axis=1)
        hsv = hsv.reshape(1, n, 3)                     # shape (1,n,3) for cv2.cvtColor
        bgr_colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(n, 3)  # uint8

        # Assign colors per label
        flat = lab.ravel()
        out_flat = out.reshape(-1, 3)
        for i, val in enumerate(labels):
            mask_i = (flat == val)
            if mask_i.any():
                out_flat[mask_i] = bgr_colors[i]

        # Return in requested channel order
        return out if bgr else out[..., ::-1]

    def Show_results(self,image_number = 15, object_name = 'object'):
        from data_generator.exr_reader import OpenEXRReader
        import shutil

        self.Define_cam_poses()

        Content_type = {'Content-Type': 'application/json',}
        render = {'render': {'annotation': True,},}

        '''if int(image_number) == -1 or int(image_number)>=self.camera_poses.shape[1]:
            rand_imgs = range(self.camera_poses.shape[1])
        else:
            rand_imgs = random.sample(range(self.camera_poses.shape[1]), int(image_number))
        '''
        rand_imgs = [3,10,18,22,24,27,30,35,42,45]

        if not os.path.exists(self.path + '/label_factory/Calibration/Results/Annotations'):
            os.makedirs(self.path + '/label_factory/Calibration/Results/Annotations')
        if not os.path.exists(self.path + '/label_factory/Calibration/Results'):
            os.makedirs(self.path + '/label_factory/Calibration/Results')
        
        print(object_name)

        self.Set_Object_pose(object_name)
        time.sleep(0.5)
        for i in rand_imgs:
            camera_pose = {
                'pose': {
                    'name': 'Camera',
                    'location': self.camera_poses[self.config['camera_indexes'].index(self.camera_index),i,:3].tolist(),
                    'rotation': self.camera_poses[self.config['camera_indexes'].index(self.camera_index),i,3:].tolist(),
                }
            }
            requests.post('http://localhost:12345', headers=Content_type, json=camera_pose)
            requests.post('http://localhost:12345', headers=Content_type, json=render)
            while True:
                if os.path.exists(self.render_path.rsplit("/",1)[0] + "/annotations/0001.exr"):
                    time.sleep(0.5)
                    shutil.copyfile(self.render_path.rsplit("/",1)[0] + "/annotations/0001.exr", self.path + '/src/Calibration/Results/Annotations/' + str(rand_imgs.index(i)).zfill(4) + '.exr')
                    if os.path.exists(self.path + '/label_factory/Calibration/Results/Annotations/' + str(rand_imgs.index(i)).zfill(4) + '.exr'):
                        time.sleep(0.5)
                        os.remove(self.render_path.rsplit("/",1)[0] + "/annotations/0001.exr")
                        break
                else:
                    time.sleep(0.1)

        for i in range(len(rand_imgs)):
            real_img = cv2.imread(self.path + '/label_factory/Calibration/Real_Images/camera_' +  str(self.camera_index) + '/' + str(rand_imgs[i]).zfill(4) + '.png')
            try:
                chstr = 'i'
                with OpenEXRReader(self.path + '/label_factory/Calibration/Results/Annotations/' + str(i).zfill(4) + '.exr', chstr) as exr:
                    mask = np.reshape(exr.i,(exr.resolution))
                    Annotation = self.Colorize_labels(mask)
            except AttributeError:
                print('Could not access channel(s) "{}", because they are not loaded!'.format(chstr))
            

            RESULTS = np.zeros(real_img.shape,dtype=real_img.dtype)
            RESULTS[:,:,:] = (0.5 * real_img[:,:,:]) + ((0.5) * Annotation[:,:,:])
            cv2.imwrite(self.path + '/label_factory/Calibration/Results/' + str(i).zfill(4) + '.png', RESULTS)

    '''#####################################################################################
    #################### Show Measurement System errors in Blender ######################
    #####################################################################################
    def MMS_B_T(self,image_number = -1, postion_mean_error = 0.003709910615278368, orientation_mean_error =  0.7301386995005932, position_std = 0.0047379104608484286, orientation_std = 0.6546477237618156):
        from data_generator.exr_reader import OpenEXRReader
        import shutil

        self.Define_cam_poses()
        
        Content_type = {'Content-Type': 'application/json',}
        render = {'render': {'annotation': True,},}

        if int(image_number) == -1 or int(image_number)>=self.camera_poses.shape[1]:
            rand_imgs = range(self.camera_poses.shape[1])
        else:
            rand_imgs = random.sample(range(self.camera_poses.shape[1]), int(image_number))

        rand_imgs = range(0,20)

        ## camera_index_ pose_of_the_image parameters 
        Param_distorted = np.zeros((len(rand_imgs),6))

        # Generate random measurement errors
        errors = np.genfromtxt(self.path + "/src/Calibration/final_errors.csv", delimiter=',')

        posi_errors_x = np.random.normal(loc=errors[0,0], scale=errors[1,0], size=(len(rand_imgs),1)) 
        posi_errors_y = np.random.normal(loc=errors[0,1], scale=errors[1,1], size=(len(rand_imgs),1))
        posi_errors_z = np.random.normal(loc=errors[0,2], scale=errors[1,2], size=(len(rand_imgs),1))
        oris_errors_x = np.random.normal(loc=errors[0,3], scale=errors[1,3], size=(len(rand_imgs),1)) * np.pi / 180
        oris_errors_y = np.random.normal(loc=errors[0,4], scale=errors[1,4], size=(len(rand_imgs),1)) * np.pi / 180
        oris_errors_z = np.random.normal(loc=errors[0,5], scale=errors[1,5], size=(len(rand_imgs),1)) * np.pi / 180

        mm_errors = np.hstack((posi_errors_x,posi_errors_y,posi_errors_z,oris_errors_x,oris_errors_y,oris_errors_z))
        
        Param_distorted = self.camera_poses[:,rand_imgs,:] + mm_errors

        if not os.path.exists(self.path + '/src/Calibration/Results'):
            os.makedirs(self.path + '/src/Calibration/Results')
        if not os.path.exists(self.path + '/src/Calibration/Results/Annotations_before'):
            os.makedirs(self.path + '/src/Calibration/Results/Annotations_before')
        if not os.path.exists(self.path + '/src/Calibration/Results/Annotations_after'):
            os.makedirs(self.path + '/src/Calibration/Results/Annotations_after')
        if not os.path.exists(self.path + '/src/Calibration/Results/Annotations_results'):
            os.makedirs(self.path + '/src/Calibration/Results/Annotations_results')

        time.sleep(0.5)
        for cam_index in range(len(self.config['camera_indexes'])):
            for i in range(len(rand_imgs)):
                camera_pose = {
                    'pose': {
                        'name': 'Camera',
                        'location': self.camera_poses[cam_index,rand_imgs[i],:3].tolist(),
                        'rotation': self.camera_poses[cam_index,rand_imgs[i],3:].tolist(),
                    }
                }
                requests.post('http://localhost:12345', headers=Content_type, json=camera_pose)
                requests.post('http://localhost:12345', headers=Content_type, json=render)
                while True:
                    if os.path.exists(self.render_path.rsplit("/",1)[0] + "/annotations/0001.exr"):
                        time.sleep(0.5)
                        shutil.copyfile(self.render_path.rsplit("/",1)[0] + "/annotations/0001.exr", self.path + '/src/Calibration/Results/Annotations_before/' + str(cam_index*len(rand_imgs)+i).zfill(4) + '.exr')
                        if os.path.exists(self.path + '/src/Calibration/Results/Annotations_before/' + str(cam_index*len(rand_imgs)+i).zfill(4) + '.exr'):
                            time.sleep(0.5)
                            os.remove(self.render_path.rsplit("/",1)[0] + "/annotations/0001.exr")
                            break
                    else:
                        time.sleep(0.1)

        time.sleep(0.5)
        for cam_index in range(len(self.config['camera_indexes'])):
            for i in range(len(rand_imgs)):
                camera_pose = {
                    'pose': {
                        'name': 'Camera',
                        'location': Param_distorted[cam_index,rand_imgs[i],:3].tolist(),
                        'rotation': Param_distorted[cam_index,rand_imgs[i],3:].tolist(),
                    }
                }
                requests.post('http://localhost:12345', headers=Content_type, json=camera_pose)
                requests.post('http://localhost:12345', headers=Content_type, json=render)
                while True:
                    if os.path.exists(self.render_path.rsplit("/",1)[0] + "/annotations/0001.exr"):
                        time.sleep(0.5)
                        shutil.copyfile(self.render_path.rsplit("/",1)[0] + "/annotations/0001.exr", self.path + '/src/Calibration/Results/Annotations_after/' + str(cam_index*len(rand_imgs)+i).zfill(4) + '.exr')
                        if os.path.exists(self.path + '/src/Calibration/Results/Annotations_after/' + str(cam_index*len(rand_imgs)+i).zfill(4) + '.exr'):
                            time.sleep(0.5)
                            os.remove(self.render_path.rsplit("/",1)[0] + "/annotations/0001.exr")
                            break
                    else:
                        time.sleep(0.1)
        
        all_ious = []
        mean_ious = []
        for i in range(1,len(rand_imgs)*len(self.config['camera_indexes'])):
            try:
                chstr = 'i'
                with OpenEXRReader(self.path + '/src/Calibration/Results/Annotations_before/' + str(i).zfill(4) + '.exr', chstr) as exr:
                    mask_before = np.reshape(exr.i,(exr.resolution))
            except AttributeError:
                print('Could not access channel(s) "{}", because they are not loaded!'.format(chstr))
            try:
                chstr = 'i'
                with OpenEXRReader(self.path + '/src/Calibration/Results/Annotations_after/' + str(i).zfill(4) + '.exr', chstr) as exr:
                    mask_after = np.reshape(exr.i,(exr.resolution))
            except AttributeError:
                print('Could not access channel(s) "{}", because they are not loaded!'.format(chstr))


            # Find unique labels except background
            labels_before = np.unique(mask_before)
            labels_before = labels_before[labels_before != 0]

            # Find unique labels except background
            labels_after = np.unique(mask_after)
            labels_after = labels_after[labels_after != 0]
            
            if not np.all(labels_before == labels_after):
                print(closed_text("The labels are not the same before and after the measurement errors!",self.Width,"red","left"))
                return False
            else:
                H, W = mask_before.shape
                K = len(labels_before)

            # Preallocate binary stack
            Stack_before = np.zeros((H, W, K), dtype=np.uint8)
            Stack_after = np.zeros((H, W, K), dtype=np.uint8)
            
            # Fill stack
            for q, lab in enumerate(labels_before):
                Stack_before[:, :, q] = (mask_before == lab).astype(np.uint8)
                #plt.imshow((mask_before == lab).astype(np.uint8))
                #plt.show()
                Stack_after[:, :, q] = (mask_after == lab).astype(np.uint8)
                #plt.imshow((mask_after == lab).astype(np.uint8))
                #plt.show()
            
            if Stack_before.shape != Stack_after.shape:
                raise ValueError("Stacks must have the same shape")
        
            K = Stack_before.shape[2]
            ious = np.zeros(K, dtype=float)
            for k in range(K):
                A = Stack_before[:,:,k].astype(bool)
                B = Stack_after[:,:,k].astype(bool)
                inter = np.logical_and(A, B)
                union = np.logical_or(A, B)
                ious[k] = inter.sum() / union.sum() if union.sum() > 0 else 0.0

                fig, axs = plt.subplots(2,2)
                
                axs[0,0].set_title("Label Before")
                axs[0,0].imshow((A))

                axs[0,1].set_title("Label After")
                axs[0,1].imshow((B))

                axs[1,0].set_title("Intersection")
                axs[1,0].imshow((inter))

                axs[1,1].set_title("Union")
                axs[1,1].imshow((union))

                fig.suptitle('IOU Value:' + str(ious[k]))
                plt.savefig(self.path + '/src/Calibration/Results/Annotations_results/' + str(i).zfill(4) + '_label_' + str(int(labels_before[k])) + '.png')
                plt.close()

            all_ious.append(ious)
            mean_iou = np.mean(ious)
            mean_ious.append(mean_iou)
        all_ious = np.array(all_ious)
        mean_ious = np.array(mean_ious)

        plt.hist(mean_ious.flatten(),bins = 50)
        plt.title("Mean IOU values")
        plt.savefig(self.path + '/src/Calibration/Results/Annotations_results/HIST.png')

        print(closed_text("Final average IOU: " + str(np.average(mean_ious)),self.Width,"white","separated"))
        print(closed_text("The standard deviation of the IOU: " + str(np.std(mean_ious)),self.Width,"white","separated"))
        
        for valami in range(all_ious.shape[1]):
            plt.hist(all_ious[:,valami].flatten(),bins = 50)
            plt.title("IOU values for label: " + str(valami+1))
            plt.savefig(self.path + '/src/Calibration/Results/Annotations_results/HIST_object'+ str(valami) +'.png')
            print(closed_text("Final average IOU for label " + str(valami+1) + ": " + str(np.average(all_ious[:,valami])),self.Width,"white","separated"))
            print(closed_text("The standard deviation of the IOU for label " + str(valami+1) + ": " + str(np.std(all_ious[:,valami])),self.Width,"white","separated"))
        return True'''






    



        



        

        
        
        
            
            
'''if __name__ == "__main__":
    valami = BAT_Optimizer()
    valami.Define_cam_poses()
    valami.Setup_camera(8)
    valami.Set_Object_pose('Cube')
    valami.Render_images('Cube','all', 8)
    valami.Create_window()
    valami.Show_results(15)'''
    

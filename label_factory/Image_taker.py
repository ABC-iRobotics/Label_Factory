#!/usr/bin/env python3
import os 
import cv2
import yaml
import time
import rclpy
import subprocess
import numpy as np
from scipy import optimize
from threading import Thread
from colorama import Fore, Style
from data_generator.moveit2 import MoveIt2
from scipy.spatial.transform import Rotation as R
from data_generator.generate_poses import PoseGenerator
from data_generator.ee_pose_subs import EE_Pose_Subscriber
from data_generator.joint_state_subs import Joint_state_Subscriber

def format_text(text, width, text_color="white", align="left"):
    def strip_ansi(s):
        result = ""
        i = 0
        while i < len(s):
            if s[i] == "\033" and i + 1 < len(s) and s[i + 1] == "[":
                i += 2
                while i < len(s) and not s[i].isalpha():
                    i += 1
                i += 1
            else:
                result += s[i]
                i += 1
        return result
    def visible_length(s):
        return len(strip_ansi(s))
    color_map = {
        "black": Fore.BLACK, "red": Fore.RED, "green": Fore.GREEN, "yellow": Fore.YELLOW,
        "blue": Fore.BLUE, "magenta": Fore.MAGENTA, "cyan": Fore.CYAN, "white": Fore.WHITE
    }
    color = color_map.get(text_color.lower(), Fore.WHITE)
    final_lines = []
    for line in text.split("\n"):
        words = line.split()
        current_line = ""
        for word in words:
            if visible_length(current_line) + visible_length(word) + (1 if current_line else 0) <= width:
                current_line += (" " if current_line else "") + word
            else:
                final_lines.append(current_line)
                current_line = word
        if current_line:
            final_lines.append(current_line)
    formatted_output = []
    for line in final_lines:
        clean_len = visible_length(line)
        padding = max(width - clean_len, 0)
        if align == "center":
            left = padding // 2
            right = padding - left
            formatted = f"{Fore.WHITE}*** {' ' * left}{color}{line}{' ' * right} {Fore.WHITE}***{Style.RESET_ALL}"
        elif align == "right":
            formatted = f"{Fore.WHITE}*** {' ' * padding}{color}{line} {Fore.WHITE}***{Style.RESET_ALL}"
        elif align == "separated":
            if ":" in line:
                key, value = line.split(":", 1)
                key_part = key.strip() + ":"
                value_part = value.strip()
                spacing = width - visible_length(key_part) - visible_length(value_part)
                spacing = max(spacing, 1)
                formatted = f"{Fore.WHITE}*** {color}{key_part}{' ' * spacing}{value_part} {Fore.WHITE}***{Style.RESET_ALL}"
            else:
                # fallback to left-align if no colon is found
                right = max(width - visible_length(line), 0)
                formatted = f"{Fore.WHITE}*** {color}{line}{' ' * right} {Fore.WHITE}***{Style.RESET_ALL}"
        else:  # default to left
            formatted = f"{Fore.WHITE}*** {color}{line}{' ' * padding} {Fore.WHITE}***{Style.RESET_ALL}"
        formatted_output.append(formatted)
    return "\n".join(formatted_output)

def closed_text(text,Width,color,align):
    Line_b =  "*" * (Width + 8) + "\n"
    Line_a =  "\n" + "*" * (Width + 8) 
    return '\n' + Fore.WHITE +  Line_b  + format_text(text,Width,color,align)  + Fore.WHITE +  Line_a


def averageQuaternions(Q):
    import numpy.matlib as npm
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = np.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = np.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return np.real(eigenVectors[:,0].A1)


class Image_taker():
    def __init__(self, print_ = False):
        '''
        Initialize the Image taker class, the moveit2
            print_:  This boolian parameter enables printing during the processes
        '''
        self.print_ = print_
        self.photo_poses = None
        self.Width = 72
        
        # Get config_file
        self.path = subprocess.check_output("ros2 pkg prefix data_generator",shell = True, text = True)
        self.path = self.path.split("/install",1)[0]
        with open(os.path.join(self.path,'src/config.yaml'), 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Create folders if not exists
        if not os.path.exists(self.path + '/src/Calibration'):
            os.makedirs(self.path + '/src/Calibration')
        if not os.path.exists(self.path + '/src/Calibration/Calibration_Images'):
            os.makedirs(self.path + '/src/Calibration/Calibration_Images')
        if not os.path.exists(self.path + '/src/Calibration/Real_Images'):
            os.makedirs(self.path + '/src/Calibration/Real_Images')
            
        rclpy.init()
        
        self.node = rclpy.node.Node("Move_ee_to_Cartesian_pose")
        self.callback_group = rclpy.callback_groups.ReentrantCallbackGroup()
        
        # Initialize Joint state, and End-Effector Pose publisher 
        self.EE_Pose_sub = EE_Pose_Subscriber()
        self.Joint_state_sub = Joint_state_Subscriber()
        
        # Create MoveIt 2 interface
        self.Moveit2_Robot = MoveIt2(
            node=self.node,
            joint_names = self.Joint_state_sub.msg.name,
            base_link_name = self.config['moveit_configs']['base_link_name'],
            end_effector_name = self.config['moveit_configs']['end_effector_name'],
            group_name = self.config['moveit_configs']['move_group_arm'],
            callback_group = self.callback_group,
            )        

    def Move_robot_n_take_imgs(self, mm_param = [3,10,3,10,1,0.25,1,5,1,5,None,5], calibration = False):
        '''
        Inputs: 
            mm_param: This list contains the necessary parameters for the movemet of the robot(see the generate_poses.py for more) and
                the camera buffer size [orbit_x,orbit_ang_x,orbit_y,orbit_ang_y,spheres,r_step,ori_x,ori_ang_x,ori_y,ori_ang_y,radius,camera_buffer]
        '''
        
        # Get joint names for MoveIt2
        rclpy.spin_once(self.Joint_state_sub)
        
        # Init Planner
        self.Moveit2_Robot.planner_id = ("RRTConnectkConfigDefault")
        
        # Spin the node in background thread(s) and wait a bit for initialization
        executor = rclpy.executors.MultiThreadedExecutor(2)
        executor.add_node(self.node)
        executor_thread = Thread(target=executor.spin, daemon=True, args=())
        executor_thread.start()
        self.node.create_rate(1.0).sleep()
        
        # Scale down velocity and acceleration of joints (percentage of maximum)
        self.Moveit2_Robot.max_velocity = 0.5
        self.Moveit2_Robot.max_acceleration = 0.5
        # Set parameters for cartesian planning
        self.Moveit2_Robot.cartesian_avoid_collisions = False
        self.Moveit2_Robot.cartesian_jump_threshold = 0.0
        
        # Create folders if not exists and define cameras, stop the program if the cameras are not detected
        camera_not_found = False
        Cam_subs = []
        for i in self.config['camera_indexes']:
            curr_cam = cv2.VideoCapture(i)
            if curr_cam.isOpened():
                curr_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                curr_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) 
                Cam_subs.append(curr_cam)
                if not os.path.exists(self.path + '/src/Calibration/Calibration_Images' + "/camera_" + str(i)):
                    os.makedirs(self.path + '/src/Calibration/Calibration_Images' + "/camera_" + str(i))
                if not os.path.exists(self.path + '/src/Calibration/Real_Images' + "/camera_" + str(i)):
                    os.makedirs(self.path + '/src/Calibration/Real_Images' + "/camera_" + str(i))
            else:
                print(closed_text("Camera number " + str(i) +  " is not found!",self.Width,"red","left"))
                return False
        # Get starting pose
        rclpy.spin_once(self.EE_Pose_sub)
        

        ee_pose_avg_pos = []
        ee_pose_avg_ori = []
        for j in range(50):
            rclpy.spin_once(self.EE_Pose_sub)
            act_ee_pose_pos = [self.EE_Pose_sub.ee_pose.pose.position.x,
                            self.EE_Pose_sub.ee_pose.pose.position.y,
                            self.EE_Pose_sub.ee_pose.pose.position.z]
                            
            act_ee_pose_ori = [self.EE_Pose_sub.ee_pose.pose.orientation.w,
                            self.EE_Pose_sub.ee_pose.pose.orientation.x,
                            self.EE_Pose_sub.ee_pose.pose.orientation.y,
                            self.EE_Pose_sub.ee_pose.pose.orientation.z]
            ee_pose_avg_pos.append(act_ee_pose_pos)
            ee_pose_avg_ori.append(act_ee_pose_ori)
        
        ee_pose_avg_pos = np. array(ee_pose_avg_pos)
        ee_pose_avg_ori = np.array(ee_pose_avg_ori)
        
        start_pose_pos = np.average(ee_pose_avg_pos,axis=0)
        start_pose_ori = averageQuaternions(ee_pose_avg_ori)
        start_pose_ori = np.array([start_pose_ori[1],start_pose_ori[2],start_pose_ori[3],start_pose_ori[0]])
        start_pose = np.append(start_pose_pos,start_pose_ori)
        if self.print_:
            print(closed_text(Fore.CYAN + "Start pose\n" + Fore.WHITE + "pose: " + str(start_pose[:3]) + "\nori: " + str(start_pose[3:]),self.Width,"white","left"))
        
        # Define home position
        start_pose_rot = np.zeros((6,))
        start_pose_rot[:3] = start_pose[:3]
        # Rotate -90° in z axis
        Rotated = np.array([[0,1,0],[-1,0,0],[0,0,1]]) @ R.from_quat(start_pose[3:]).as_matrix()
        start_pose_rot[3:] = R.from_matrix(Rotated).as_euler('xyz')
        
        calibration_photo_poses = []
        move_ee_pose = start_pose_rot
        for i in range(5):    
            pose_generator = PoseGenerator()
            # orbiting_poses: Sphere layer calculated so far, layer spacing, number of layers
            # 3D grid; the number of orientations in the given position (add one to the even)
            orbiting_poses = pose_generator.generate_orbitings(start_pose=move_ee_pose , num_pos_x = mm_param[0], num_pos_y = mm_param[2], angle_pos_x = mm_param[1]/180*np.pi, angle_pos_y = mm_param[3]/180*np.pi, radius = mm_param[10])
            sphere_poses = pose_generator.generate_spheres(orbiting_poses=orbiting_poses, r_step = mm_param[5], steps = mm_param[4])
            generated_pose = pose_generator.generate_poses(poses=sphere_poses, num_ori_x = mm_param[6], num_ori_y = mm_param[8], ori_x_angle = mm_param[7]/180*np.pi, ori_y_angle = mm_param[9]/180*np.pi)
            calibration_photo_poses.append(generated_pose)
            # Rotate +45° in z axis
            Rotated = np.array([[1/np.sqrt(2),-1/np.sqrt(2),0],[1/np.sqrt(2),1/np.sqrt(2),0],[0,0,1]]) @ R.from_euler('xyz',move_ee_pose [3:]).as_matrix()
            move_ee_pose [3:] = R.from_matrix(Rotated).as_euler('xyz')
            
        calibration_photo_poses = np.reshape(np.array(calibration_photo_poses),(-1,6))
        calibration_photo_poses = np.append(calibration_photo_poses,np.zeros((calibration_photo_poses.shape[0],1)),axis = 1)
        for i in range(calibration_photo_poses.shape[0]):
            q = R.from_euler('xyz', [calibration_photo_poses[i,3],calibration_photo_poses[i,4],calibration_photo_poses[i,5]],degrees=False).as_quat()
            calibration_photo_poses[i,3:] = q
        calibration_photo_poses = np.append(calibration_photo_poses,np.array([start_pose]),axis = 0)
        np.savetxt(self.path + "/src/Calibration/calibration_photo_poses.csv", calibration_photo_poses , delimiter=",")
        
        # Logged End-Effector poses
        ee_poses = []
        for i in range(calibration_photo_poses.shape[0]):
            if self.print_ :
                print(closed_text(Fore.CYAN + "Moving to pose\n" + Fore.WHITE + "position: " + str(calibration_photo_poses[i,0:3]) + "\norientation: " + str(calibration_photo_poses[i,3:7]),self.Width,"white","left"))
            # Get position and orientation for the robot
            position = np.array([calibration_photo_poses[i,0],calibration_photo_poses[i,1],calibration_photo_poses[i,2]])
            orientation = np.array([calibration_photo_poses[i,3],calibration_photo_poses[i,4],calibration_photo_poses[i,5],calibration_photo_poses[i,6]])
            # Move the robot to the defined pose in cartesian space, linear movement 
            self.Moveit2_Robot.move_to_pose(
                position = position,
                quat_xyzw = orientation,
                cartesian = True,
                cartesian_max_step = 0.000025,
                cartesian_fraction_threshold = 0.0,
            )
            
            self.Moveit2_Robot.wait_until_executed()   
            # Get the Joint state, if the velocity is small enough, and the acceleration is small, the robot stopped moving
            rclpy.spin_once(self.Joint_state_sub)
            previous_velocities = np.abs(np.array(self.Joint_state_sub.msg.velocity))
            while np.any(previous_velocities>=1e-10) or np.any(previous_velocities-np.array(self.Joint_state_sub.msg.velocity)>1e-10):
                time.sleep(0.1)
                rclpy.spin_once(self.Joint_state_sub)
                previous_velocities = np.abs(np.array(self.Joint_state_sub.msg.velocity))

            # Get the End-Effector pose from the robot controller using a moving average(5)
            ee_pose_avg_pos = []
            ee_pose_avg_ori = []
            for j in range(30):
                rclpy.spin_once(self.EE_Pose_sub)
                act_ee_pose_pos = [self.EE_Pose_sub.ee_pose.pose.position.x,
                        self.EE_Pose_sub.ee_pose.pose.position.y,
                        self.EE_Pose_sub.ee_pose.pose.position.z]
                        
                act_ee_pose_ori = [self.EE_Pose_sub.ee_pose.pose.orientation.w,
                            self.EE_Pose_sub.ee_pose.pose.orientation.x,
                            self.EE_Pose_sub.ee_pose.pose.orientation.y,
                            self.EE_Pose_sub.ee_pose.pose.orientation.z]
                
                ee_pose_avg_pos.append(act_ee_pose_pos)
                ee_pose_avg_ori.append(act_ee_pose_ori)
            
            ee_pose_avg_pos = np. array(ee_pose_avg_pos)
            ee_pose_avg_ori = np.array(ee_pose_avg_ori)
            value_pos = np.average(ee_pose_avg_pos,axis=0)
            value_ori = averageQuaternions(ee_pose_avg_ori)
            value_ori = np.array([value_ori[1],value_ori[2],value_ori[3],value_ori[0]])
            ee_pose_avg = np.append(value_pos,value_ori)
            ee_pose_avg
            ee_poses.append(np.array(ee_pose_avg).tolist())
            
            for j in range(len(Cam_subs)):
                for k in range(mm_param[11]):
                    ret, image = Cam_subs[j].read()
                if ret:
                    if calibration:
                        cv2.imwrite(os.path.join(self.path + '/src/Calibration/Calibration_Images/camera_' + str(self.config['camera_indexes'][j]) + "/", str(i).zfill(4) + ".png"), image)
                    else:
                        cv2.imwrite(os.path.join(self.path + '/src/Calibration/Real_Images/camera_' + str(self.config['camera_indexes'][j]) + "/", str(i).zfill(4) + ".png"), image)
                if self.print_:
                    print(closed_text("Image taken with camera: " + str(self.config['camera_indexes'][j]),self.Width,"white","left"))
        # Save the photo positions
        np.savetxt(self.path + "/src/Calibration/robot_poses.csv", ee_poses , delimiter=",")
        self.photo_poses = np.genfromtxt(self.path + "/src/Calibration/robot_poses.csv", delimiter=',')
        
        return True
        
    def Calibrate_cameras(self):
        if self.photo_poses is None:
            self.photo_poses = np.genfromtxt(self.path + "/src/Calibration/robot_poses.csv", delimiter=',')
        # Define the ChArUco board and the detector for that 
        board = cv2.aruco.CharucoBoard((37,25), 0.03, 0.023, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))	
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),  cv2.aruco.DetectorParameters())
        
        # Check if the cameras dict is in the config, if not create a cameras dict
        new_yaml_data_dict = {'cameras': {}}
        if not "cameras" in self.config:
            self.config.update(new_yaml_data_dict)
            with open(self.path + '/src/config.yaml','w') as yamlfile:
                yaml.safe_dump(self.config, yamlfile, default_flow_style = False)
        
        # Calibrate the cameras using the ChArUco board 
        for cam_index in self.config['camera_indexes']:
            allCorners = []
            allIds = []
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
            for i in range(self.photo_poses.shape[0]):
                frame = cv2.imread(self.path + '/src/Calibration/Calibration_Images/camera_' + str(cam_index) + "/" + str(i).zfill(4) + ".png", cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                if len(corners)>0:               
                    for corner in corners:
                        cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria) 
                    res2 = 	cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                    if res2[1] is not None and res2[2] is not None:
                        allCorners.append(res2[1])
                        allIds.append(res2[2])
            (_, camera_matrix, distortion_coefficient, _, _) = cv2.aruco.calibrateCameraCharuco(
                charucoCorners = allCorners,
                charucoIds = allIds,
                board = board,
                imageSize = gray.shape,
                cameraMatrix = np.array([[ 1000, 0, gray.shape[0]/2.],
                                        [0, 1000, gray.shape[1]/2.],
                                        [0, 0, 1]]),
                distCoeffs = np.zeros((4,1)),
                flags = (cv2.CALIB_FIX_K3),
                criteria = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
            if self.print_:
                print(closed_text("Camera Number " + str(cam_index),self.Width,"cyan","center") + "\n" + format_text("The camera intrinsic matrix:",self.Width,"yellow","left") + "\n" + format_text(np.array2string(camera_matrix, precision=5, separator=','),self.Width,"white","left") + "\n" + format_text("The camera distorsion coefficients:",self.Width,"yellow","left") + "\n" + format_text(np.array2string(distortion_coefficient.flatten(),precision=5,separator=','),self.Width,"white","left") + "\n" + '*'*(self.Width+8))

            # Update the config file, giving  camera_<index_number> to the cameras dict with the camera_matrix and distCoeffs datas
            new_yaml_data_dict = {'camera_'+ str(cam_index): {
            'CameraIntrinsic' : camera_matrix.astype(float).flatten().tolist(),
            'DistCoeffs' : distortion_coefficient.astype(float).flatten().tolist()
                }
            }
            self.config['cameras'].update(new_yaml_data_dict)
            with open(self.path + '/src/config.yaml','w') as yamlfile:
                yaml.safe_dump(self.config, yamlfile, default_flow_style = False)
        

    def Validate_cam_calib(self):
        print("Start Validation")
        if self.photo_poses is None:
            self.photo_poses = np.genfromtxt(self.path + "/src/Calibration/robot_poses.csv", delimiter=',')
        # Define the ChArUco board and the detector for that 
        board = cv2.aruco.CharucoBoard((37,25), 0.03, 0.023, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))	
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),  cv2.aruco.DetectorParameters())

        print("Calculate camera params")
        # Calibrate the cameras using the ChArUco board 
        for cam_index in self.config['camera_indexes']:
            allCorners = []
            allIds = []
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
            
            for i in range(self.photo_poses.shape[0]):

                frame = cv2.imread(self.path + '/src/Calibration/Calibration_Images/camera_' + str(cam_index) + "/" + str(i).zfill(4) + ".png", cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                if len(corners)>0:               
                    for corner in corners:
                        cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria) 
                    res2 = 	cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                    if res2[1] is not None and res2[2] is not None:
                        allCorners.append(res2[1])
                        allIds.append(res2[2])
            (_, camera_matrix, distortion_coefficient, rvecs, tvecs) = cv2.aruco.calibrateCameraCharuco(
                charucoCorners = allCorners,
                charucoIds = allIds,
                board = board,
                imageSize = gray.shape,
                cameraMatrix = np.array([[ 1000, 0, gray.shape[0]/2.],
                                        [0, 1000, gray.shape[1]/2.],
                                        [0, 0, 1]]),
                distCoeffs = np.zeros((4,1)),
                flags = (cv2.CALIB_FIX_K3),
                criteria = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
            
            print("Check camera params")
            mean_error, errors_per_image = self.compute_reprojection_error(
    allCorners, allIds, board, camera_matrix, distortion_coefficient, rvecs, tvecs
)
            print(f"Mean reprojection error: {mean_error:.4f} pixels")
            for i, err in enumerate(errors_per_image):
                print(f"Image {i}: {err:.4f} px")

    def compute_reprojection_error(self,all_corners, all_ids, board, cameraMatrix, distCoeffs, rvecs, tvecs):
        total_error = 0
        total_points = 0
        errors_per_image = []

        # Precompute the full set of 3D object points
        obj_points_all = board.getChessboardCorners()

        for i in range(len(all_corners)):
            if all_ids[i] is not None and len(all_ids[i]) > 0:
                # Select the 3D points that correspond to the detected IDs
                obj_points = obj_points_all[all_ids[i].flatten(), :]

                # Project them using the calibration parameters
                img_points, _ = cv2.projectPoints(
                    obj_points, rvecs[i], tvecs[i], cameraMatrix, distCoeffs
                )

                detected = all_corners[i].reshape(-1, 2)
                projected = img_points.reshape(-1, 2)

                # RMS error for this image
                error = cv2.norm(detected, projected, cv2.NORM_L2) / len(projected)
                errors_per_image.append(error)

                total_error += (error**2) * len(projected)
                total_points += len(projected)

        mean_error = np.sqrt(total_error / total_points)
        return mean_error, errors_per_image


    def SVD_pose_diff_estimation(self,Points_Blender, Points_estimated):
        Points_Blender_mean = np.mean(Points_Blender,axis=0)
        Points_estimated_mean = np.mean(Points_estimated,axis=0)
        Blender_diff = Points_Blender-Points_Blender_mean
        estimated_diff = Points_estimated-Points_estimated_mean
        H = Blender_diff.T @ estimated_diff  
        U, _, Vt = np.linalg.svd(H)
        Rot = Vt.T @ U.T
        det = np.linalg.det(Rot)
        if det < 0:
            S = np.eye(3)
            S[-1, -1] = -1
            Rot = Vt.T @ S @ U.T
        t = Points_estimated_mean.T - Rot @ Points_Blender_mean.T   
        Tr = np.append(np.append(Rot,np.reshape(t,(t.shape[0],1)),axis=1),np.array([[0,0,0,1]]),axis=0)
        return Tr

    def Calibrate_flange_Cam(self):
        if self.photo_poses is None:
            self.photo_poses = np.genfromtxt(self.path + "/src/Calibration/robot_poses.csv", delimiter=',')
        # Define the ChArUco board and the detector for that 
        board = cv2.aruco.CharucoBoard((37,25), 0.03, 0.023, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))	
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),  cv2.aruco.DetectorParameters())
        
        T_Base_Flange = np.zeros((self.photo_poses.shape[0],4,4))
        for i in range(self.photo_poses.shape[0]):
            T_Base_Flange[i,0:3,3] = self.photo_poses[i,0:3]
            T_Base_Flange[i,0:3,0:3] = R.from_quat(self.photo_poses[i,3:]).as_matrix()
            T_Base_Flange[i,3,3] = 1
        
        reject = []
        # Collect Camera to board transformations
        Cam_to_board = np.zeros((len(self.config['camera_indexes']),self.photo_poses.shape[0],4,4))
        for cam_index in self.config['camera_indexes']:
            camera_matrix = np.reshape(np.array(self.config['cameras']['camera_' + str(cam_index)]['CameraIntrinsic']),(3,3))
            distortion_coefficient = np.reshape(np.array(self.config['cameras']['camera_' + str(cam_index)]['DistCoeffs']),(-1,1)) 
            for i in range(self.photo_poses.shape[0]):
                frame = cv2.imread(self.path + '/src/Calibration/Calibration_Images/camera_' + str(cam_index) + "/" + str(i).zfill(4) + ".png", cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(cv2.undistort(frame, camera_matrix, distortion_coefficient), cv2.COLOR_RGB2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                for corner in corners:
                    cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001))                        
                if len(corners) > 0:
                    _, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                    _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, distortion_coefficient,None,None)
                if rvec is not None and tvec is not None:
                    Rotmat = R.from_rotvec(np.reshape(rvec,(3,))).as_matrix()
                    Cam_to_board[self.config['camera_indexes'].index(cam_index),i,0:3,0:3] = Rotmat
                    Cam_to_board[self.config['camera_indexes'].index(cam_index),i,0:3,3] = np.reshape(np.array([tvec[0],tvec[1],tvec[2]]),(3,))
                    Cam_to_board[self.config['camera_indexes'].index(cam_index),i,3,3] = 1
                else:
                    reject.append(i)  

        # Define the initial Rotation matrices From Flnage to each Camera 
        T_Flange_Cam = np.zeros((len(self.config['flange_camera_orientation']),4,4))
        R_Flange_Cam = np.zeros((len(self.config['flange_camera_orientation']),3,3))
        for i in range(len(self.config['flange_camera_orientation'])):
            R_Flange_Cam[i,:,:] = R.from_euler('xyz',self.config['flange_camera_orientation'][i],degrees= True).as_matrix()
            T_Flange_Cam[i,0:3,0:3] = R.from_euler('xyz',self.config['flange_camera_orientation'][i],degrees= True).as_matrix()
            T_Flange_Cam[i,0:3,3] = np.array(self.config['flange_camera_translation'][i])
            T_Flange_Cam[i,3,3] = 1            

        Tra_next = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],4,4))
        for i in range(Cam_to_board.shape[0]):
            for j in range(Cam_to_board.shape[1]):
                Tra_next[i*Cam_to_board.shape[1]+j,:,:] = T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:]

        Inv_Cam_board = np.zeros(Cam_to_board.shape)
        for i in range(Cam_to_board.shape[0]):
            for j in range(Cam_to_board.shape[1]):
                Inv_Cam_board[i,j,:,:] = np.linalg.inv(Cam_to_board[i,j,:,:])

        Points_cam = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],3))
        for i in range(Cam_to_board.shape[0]):
            for j in range(Cam_to_board.shape[1]):
                Points_cam[i*Cam_to_board.shape[1] + j,:] = Inv_Cam_board[i,j,0:3,3]

        Points_Robot = np.zeros((Tra_next.shape[0],3))
        for i in range(Tra_next.shape[0]):
            Points_Robot[i,:] = Tra_next[i,0:3,3]
        
        Fin_T = self.SVD_pose_diff_estimation(Points_Robot,Points_cam)
            
        Tra_error = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],4,4))
        Pos_error = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],3))
        Ori_error = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],3))
        for i in range(Cam_to_board.shape[0]):
            for j in range(Cam_to_board.shape[1]):
                Tra_error[i*Cam_to_board.shape[1]+j,:,:] = T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:] @ Cam_to_board [i,j,:,:] @ Fin_T
                Pos_error[i*Cam_to_board.shape[1]+j,:] = (T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:] @ Cam_to_board [i,j,:,:] @ Fin_T)[0:3,3]
                Ori_error[i*Cam_to_board.shape[1]+j,:] = R.from_matrix((T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:] @ Cam_to_board [i,j,:,:] @ Fin_T)[0:3,0:3]).as_rotvec() 
        
        cam_trans = np.array(self.config['flange_camera_translation'])
        cam_ori_eul = np.array(self.config['flange_camera_orientation'])
        parameters = np.append(cam_trans.flatten(),cam_ori_eul.flatten())

        parameters = optimize.fmin(func = self.Optimizer, x0 = parameters, args = (T_Base_Flange, Cam_to_board, Points_cam), ftol = 10e-10, xtol=10e-10, maxiter = 8000, disp = True)
        
        print(parameters)

        parameters = np.reshape(parameters,(len(self.config['flange_camera_orientation']),6))
        T_Flange_Cam = np.zeros((len(self.config['flange_camera_orientation']),4,4))
        for i in range(len(self.config['flange_camera_orientation'])):
            T_Flange_Cam[i,0:3,0:3] = R.from_euler('xyz',parameters[1,i*3:(i+1)*3],degrees= True).as_matrix()
            T_Flange_Cam[i,0:3,3] = np.array(parameters[0,i*3:(i+1)*3])
            T_Flange_Cam[i,3,3] = 1   
        
        # megnézi hogy ezzel hol vannak a kamera pózok
        Tra_next = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],4,4))
        for i in range(Cam_to_board.shape[0]):
            for j in range(Cam_to_board.shape[1]):
                Tra_next[i*Cam_to_board.shape[1]+j,:,:] = T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:]
        
        # kiszámolja a kamera pozíciókat így 
        Points_Robot = np.zeros((Tra_next.shape[0],3))
        for i in range(Tra_next.shape[0]):
            Points_Robot[i,:] = Tra_next[i,0:3,3]

        Fin_T = self.SVD_pose_diff_estimation(Points_Robot,Points_cam)

        Tra_error = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],4,4))
        Pos_error = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],3))
        Ori_error = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],3))
        for i in range(Cam_to_board.shape[0]):
            for j in range(Cam_to_board.shape[1]):
                Tra_error[i*Cam_to_board.shape[1]+j,:,:] = T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:] @ Cam_to_board [i,j,:,:] @ Fin_T
                Pos_error[i*Cam_to_board.shape[1]+j,:] = (T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:] @ Cam_to_board [i,j,:,:] @ Fin_T)[0:3,3]*1000
                Ori_error[i*Cam_to_board.shape[1]+j,:] = R.from_matrix((T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:] @ Cam_to_board [i,j,:,:] @ Fin_T)[0:3,0:3]).as_rotvec(degrees = True) 
        
        print(np.mean(np.linalg.norm(Pos_error,axis=1)))
        print(np.mean(np.linalg.norm(Ori_error,axis=1)))
        import matplotlib.pyplot as plt
        plt.hist(Pos_error[:,0],bins = 50)
        plt.title("Mean position errors")
        plt.xlabel("Error [mm]")
        plt.ylabel("Counts")
        plt.show()

        plt.hist(Pos_error[:,1],bins = 50)
        plt.title("Mean position errors")
        plt.xlabel("Error [meter]")
        plt.ylabel("Counts")
        plt.show()

        plt.hist(Pos_error[:,2],bins = 50)
        plt.title("Mean position errors")
        plt.xlabel("Error [meter]")
        plt.ylabel("Counts")
        plt.show()

        plt.hist(Ori_error[:,0],bins = 50)
        plt.title("Mean orientation errors")
        plt.xlabel("Error [deg]")
        plt.ylabel("Counts")
        plt.show()

        plt.hist(Ori_error[:,1],bins = 50)
        plt.title("Mean orientation errors")
        plt.xlabel("Error [deg]")
        plt.ylabel("Counts")
        plt.show()

        plt.hist(Ori_error[:,2],bins = 50)
        plt.title("Mean orientation errors")
        plt.xlabel("Error [deg]")
        plt.ylabel("Counts")
        plt.show()

        T_Flange_Cam[i,:,:].flatten().tolist() 
        # Update the config files with the optimized transfomations
        for i in range(len(self.config['camera_indexes'])):
            if self.print_:
                print(closed_text("Camera Number " + str(cam_index),self.Width,"cyan","center") + "\n" + format_text("The Optimized Flange-Camera Transformation matrix:",self.Width,"yellow","left") + "\n" + format_text(np.array2string(T_Flange_Cam[i,:,:], precision=5, separator=','),self.Width,"white","left"))
            data_to_list = T_Flange_Cam[i,:,:].flatten().tolist() 
            new_yaml_data_dict = {'Tmatrix_Flange_Camera' : data_to_list}
            self.config['cameras']['camera_' + str(self.config['camera_indexes'][i])].update(new_yaml_data_dict)
            with open(self.path + '/src/config.yaml','w') as yamlfile:
                yaml.safe_dump(self.config, yamlfile, default_flow_style = False)

        return 0

    def Optimizer(self,parameters,T_Base_Flange,Cam_to_board,Points_cam):
        parameters = np.reshape(parameters,(len(self.config['flange_camera_orientation']),6))
        #paraméterekből kiszámolja a Flange-camera trafót
        T_Flange_Cam = np.zeros((len(self.config['flange_camera_orientation']),4,4))
        for i in range(len(self.config['flange_camera_orientation'])):
            T_Flange_Cam[i,0:3,0:3] = R.from_euler('xyz',parameters[1,i*3:(i+1)*3],degrees= True).as_matrix()
            T_Flange_Cam[i,0:3,3] = np.array(parameters[0,i*3:(i+1)*3])
            T_Flange_Cam[i,3,3] = 1
        
        # megnézi hogy ezzel hol vannak a kamera pózok
        Tra_next = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],4,4))
        for i in range(Cam_to_board.shape[0]):
            for j in range(Cam_to_board.shape[1]):
                Tra_next[i*Cam_to_board.shape[1]+j,:,:] = T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:]
        
        # kiszámolja a kamera pozíciókat így 
        Points_Robot = np.zeros((Tra_next.shape[0],3))
        for i in range(Tra_next.shape[0]):
            Points_Robot[i,:] = Tra_next[i,0:3,3]

        Fin_T = self.SVD_pose_diff_estimation(Points_Robot,Points_cam)

        Tra_error = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],4,4))
        Pos_error = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],3))
        Ori_error = np.zeros((Cam_to_board.shape[0]*Cam_to_board.shape[1],3))
        for i in range(Cam_to_board.shape[0]):
            for j in range(Cam_to_board.shape[1]):
                Tra_error[i*Cam_to_board.shape[1]+j,:,:] = T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:] @ Cam_to_board [i,j,:,:] @ Fin_T
                Pos_error[i*Cam_to_board.shape[1]+j,:] = (T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:] @ Cam_to_board [i,j,:,:] @ Fin_T)[0:3,3]*1000
                Ori_error[i*Cam_to_board.shape[1]+j,:] = R.from_matrix((T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:] @ Cam_to_board [i,j,:,:] @ Fin_T)[0:3,0:3]).as_rotvec(degrees = True) 
        
        print(np.mean(np.linalg.norm(Pos_error,axis=1)))
        print(np.mean(np.linalg.norm(Ori_error,axis=1)))
        error = np.mean(np.linalg.norm(Pos_error,axis = 1)) + np.mean(np.linalg.norm(Ori_error,axis = 1))**2
        return error
    
        # Define the relative orientation differences between the first and each of the Rotation matrices in euler angles
        # So a good initial parameters can be used for the optimization
        eulers = []
        for i in range(Cam_to_board.shape[0]):
            for j in range(Cam_to_board.shape[1]):
                Rot_error = T_Base_Flange[j,0:3,0:3] @ R_Flange_Cam[i,:,:] @ Cam_to_board[i,j,0:3,0:3] @ np.linalg.inv(Rot_error_first)
                euler = R.from_matrix(Rot_error).as_euler('xyz',degrees=True).tolist()
                eulers.append(euler)
                
        import matplotlib.pyplot as plt
        eulers = np.array(eulers)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(eulers[:,0],eulers[:,1],eulers[:,2])
        plt.show()

        # Define the average euler angles differenece from the first 
        eulers_avg = np.average(np.array(eulers),axis=0)

        # Setup the initial parameters for the orinetation optimiztaion
        Rot_Base_Board = R.from_euler('xyz',eulers_avg,degrees=True).as_matrix() @ Rot_error_first
        parameters = R.from_matrix(Rot_Base_Board).as_euler('xyz',degrees=True).tolist()
        for orients in self.config['flange_camera_orientation']:
            parameters.extend(orients)
        
        print(closed_text("It may take some time!",self.Width,"white","left"))
        time.sleep(1.5)
        
        self.print_counter = 0
        # Optimize the orientation
        Optim_oris = optimize.fmin(func = self.Fitness_ori, x0 = parameters, args = (T_Base_Flange, Cam_to_board), ftol = 1e-10, maxiter = 5000, disp = False)
        print(closed_text("Orientation optimization is finnished!",self.Width,"green","center"))
        
        # Define the Transfromation matrices, whith the optimized Rotation matrix parts
        T_Base_Board = np.zeros((4,4))
        T_Base_Board[0:3,0:3] = R.from_euler('xyz',Optim_oris[:3],degrees=True).as_matrix()
        T_Base_Board[3,3] = 1
        T_Flange_Cam = np.zeros((len(self.config['flange_camera_orientation']),4,4))
        for i in range(T_Flange_Cam.shape[0]):
            T_Flange_Cam[i,0:3,0:3] = R.from_euler('xyz',Optim_oris[(i+1)*3:(i+2)*3],degrees=True).as_matrix()
            T_Flange_Cam[i,3,3] = 1
            
        # Define the average distance, to define a good initial parameter for the optimization
        dists = []
        for i in range(Cam_to_board.shape[0]):
            for j in range(Cam_to_board.shape[1]):
                Tra_error = T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:] @ Cam_to_board[i,j,:,:] @ np.linalg.inv(T_Base_Board)
                dists.append(Tra_error[0:3,3].tolist())

        import matplotlib.pyplot as plt
        dists = np.array(dists)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(dists[:,0],dists[:,1],dists[:,2])
        plt.show()
                
        # Define the parameters for the optimization
        dist_avg = np.average(np.array(dists),axis=0)
        parameters = dist_avg.tolist()
        for translations in self. config['flange_camera_translation']:
            parameters.extend(translations)
            
        self.print_counter = 0
        # Optimize the translations in the transformation matrices
        Optim_pos = optimize.fmin(func = self.Fitness_pos, x0 = parameters, args = (T_Base_Flange, Cam_to_board,T_Base_Board,T_Flange_Cam), xtol = 1e-12, ftol = 1e-12, maxiter = 5000, disp = False)
        print(closed_text("Translation optimization is finnished!",self.Width,"green","center"))
        
        # Generate the Transformation matrices for the cameras
        T_Base_Board[0:3,3] = Optim_pos[0:3]
        for i in range(T_Flange_Cam.shape[0]):
            T_Flange_Cam[i,0:3,3] = Optim_pos[(i+1)*3:(i+2)*3]


        ################################################################################
        ############### Final error calculation and printing the results ###############
        ################################################################################
        # Calculate the final errors
        Traf_errors = np.zeros((Cam_to_board.shape[0],Cam_to_board.shape[1],4,4))
        # Number of cams
        for i in range(Cam_to_board.shape[0]):
            # Number of robot poses
            for j in range(Cam_to_board.shape[1]):
                Traf_errors[i,j,:,:] = T_Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:] @ Cam_to_board[i,j,:,:] @ np.linalg.inv(T_Base_Board)
        
        pos_errors = np.zeros((Traf_errors.shape[0],Traf_errors.shape[1],3))
        oris_errors_eul = np.zeros((Traf_errors.shape[0],Traf_errors.shape[1],3))
        oris_errors_rotvec = np.zeros((Traf_errors.shape[0],Traf_errors.shape[1],3))
        mean_pos_errors = np.zeros((Traf_errors.shape[0]*Traf_errors.shape[1],1))
        mean_oris_errors = np.zeros((Traf_errors.shape[0]*Traf_errors.shape[1],1))
        for i in range(Traf_errors.shape[0]):
            for j in range(Traf_errors.shape[1]):
                pos_errors[i,j,:] = Traf_errors[i,j,0:3,3]
                oris_errors_eul[i,j,:] = R.from_matrix(Traf_errors[i,j,0:3,0:3]).as_euler('xyz',degrees=True)
                oris_errors_rotvec[i,j,:] = R.from_matrix(Traf_errors[i,j,0:3,0:3]).as_rotvec(degrees=True)
                mean_pos_errors[i*Traf_errors.shape[1]+j] = np.linalg.norm(pos_errors[i,j,:])
                mean_oris_errors[i*Traf_errors.shape[1]+j] = np.linalg.norm(oris_errors_rotvec[i,j,:])
        
        import matplotlib.pyplot as plt
        mean_pos_errors[i*Traf_errors.shape[1]+j] = np.linalg.norm(pos_errors[i,j,:])
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.photo_poses[:,0],self.photo_poses[:,1],self.photo_poses[:,2])
        plt.show()

        for cam_index in range(len(self.config['camera_indexes'])):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(Cam_to_board[cam_index,:,0,3],Cam_to_board[cam_index,:,1,3],Cam_to_board[cam_index,:,2,3])
            plt.show()

        plt.hist(mean_pos_errors.flatten(),bins = 50)
        plt.title("Mean position errors")
        plt.xlabel("Error [meter]")
        plt.ylabel("Counts")
        plt.show()

        plt.hist(mean_oris_errors.flatten(),bins = 50)
        plt.title("Mean orientation errors")
        plt.xlabel("Error [deg]")
        plt.ylabel("Counts")
        plt.show()

        plt.hist(pos_errors[:,:,0].flatten(),bins = 50)
        plt.title("Position errors in x")
        plt.xlabel("Error [meter]")
        plt.ylabel("Counts")
        plt.show()

        plt.hist(pos_errors[:,:,1].flatten(),bins = 50)
        plt.title("Position errors in y")
        plt.xlabel("Error [meter]")
        plt.ylabel("Counts")
        plt.show()

        plt.hist(pos_errors[:,:,2].flatten(),bins = 50)
        plt.title("Position errors in z")
        plt.xlabel("Error [meter]")
        plt.ylabel("Counts")
        plt.show()

        pos_errors_save = np.array([[np.mean(pos_errors[:,:,0]),
                                    np.mean(pos_errors[:,:,1]),
                                    np.mean(pos_errors[:,:,2]),],
                                    [np.std(pos_errors[:,:,0]),
                                    np.std(pos_errors[:,:,1]),
                                    np.std(pos_errors[:,:,2])]])

        plt.hist(oris_errors_eul[:,:,0].flatten(),bins = 50)
        plt.title("Orientation errors in x")
        plt.xlabel("Error [deg]")
        plt.ylabel("Counts")
        plt.show()
    
        plt.hist(oris_errors_eul[:,:,1].flatten(),bins = 50)
        plt.title("Orientation errors in y")
        plt.xlabel("Error [deg]")
        plt.ylabel("Counts")
        plt.show()

        plt.hist(oris_errors_eul[:,:,2].flatten(),bins = 50)
        plt.title("Orientation errors in z")
        plt.xlabel("Error [deg]")
        plt.ylabel("Counts")
        plt.show()

        oris_errors_eul_save = np.array([[np.mean(oris_errors_eul[:,:,0]),
                                        np.mean(oris_errors_eul[:,:,1]),
                                        np.mean(oris_errors_eul[:,:,2]),], 
                                        [np.std(oris_errors_eul[:,:,0]),
                                        np.std(oris_errors_eul[:,:,1]),
                                        np.std(oris_errors_eul[:,:,2])]])

        saved = np.hstack((pos_errors_save,oris_errors_eul_save))
        np.savetxt(self.path + "/src/Calibration/final_errors.csv", saved , delimiter=",")


        print(closed_text("Final average position error: " + str(np.average(mean_pos_errors)) + " [meter]",self.Width,"white","separated"))
        print(closed_text("The standard deviation of the position error: " + str(np.std(mean_pos_errors)) + " [meter]",self.Width,"white","separated"))
        print(closed_text("Final average orientation error: " + str(np.average(mean_oris_errors)) + " [deg]",self.Width,"white","separated"))
        print(closed_text("The standard deviation of the orientation error: " + str(np.std(mean_oris_errors)) + " [deg]",self.Width,"white","separated"))
        

        ##############################################################################
        ########## Update the config files with the optimized transfomations #########
        ##############################################################################
        # Update the config files with the optimized transfomations
        for i in range(len(self.config['camera_indexes'])):
            if self.print_:
                print(closed_text("Camera Number " + str(cam_index),self.Width,"cyan","center") + "\n" + format_text("The Optimized Flange-Camera Transformation matrix:",self.Width,"yellow","left") + "\n" + format_text(np.array2string(T_Flange_Cam[i,:,:], precision=5, separator=','),self.Width,"white","left"))
            data_to_list = T_Flange_Cam[i,:,:].flatten().tolist() 
            new_yaml_data_dict = {'Tmatrix_Flange_Camera' : data_to_list}
            self.config['cameras']['camera_' + str(self.config['camera_indexes'][i])].update(new_yaml_data_dict)
            with open(self.path + '/src/config.yaml','w') as yamlfile:
                yaml.safe_dump(self.config, yamlfile, default_flow_style = False)
        
        
    def PoC_R2B(self):
        if self.photo_poses is None:
            self.photo_poses = np.genfromtxt(self.path + "/src/Calibration/robot_poses.csv", delimiter=',')
        # Define the ChArUco board and the detector for that 
        board = cv2.aruco.CharucoBoard((37,25), 0.03, 0.023, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))	
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),  cv2.aruco.DetectorParameters())
        
        import random
        photo_poses = self.photo_poses

        # Collect Camera to board transformations
        Cam_to_board = np.zeros((len(self.config['camera_indexes']),photo_poses.shape[0],4,4))
        Cam_to_board_params = np.zeros((len(self.config['camera_indexes']),photo_poses.shape[0],6))
        Cam_to_board_poses = np.zeros((len(self.config['camera_indexes'])*photo_poses.shape[0],7))
        for cam_index in self.config['camera_indexes']:
            print("Camera index: " + str(cam_index))
            camera_matrix = np.reshape(np.array(self.config['cameras']['camera_' + str(cam_index)]['CameraIntrinsic']),(3,3))
            distortion_coefficient = np.reshape(np.array(self.config['cameras']['camera_' + str(cam_index)]['DistCoeffs']),(-1,1)) 
            for i in range(photo_poses.shape[0]):
                frame = cv2.imread(self.path + '/src/Calibration/Real_Images/camera_' + str(cam_index) + "/" + str(i).zfill(4) + ".png", cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                corners, ids, _ = detector.detectMarkers(gray)
                for corner in corners:
                    cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001))                        
                if len(corners) > 0:
                    _, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                    _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, distortion_coefficient, None, None)
                if rvec is not None and tvec is not None:
                    Rotmat =  R.from_rotvec(rvec.flatten()).as_matrix()
                    Cam_to_board[self.config['camera_indexes'].index(cam_index),i,0:3,0:3] = Rotmat
                    Cam_to_board[self.config['camera_indexes'].index(cam_index),i,0:3,3] = np.reshape(np.array([tvec[0],tvec[1],tvec[2]]),(3,))
                    Cam_to_board[self.config['camera_indexes'].index(cam_index),i,3,3] = 1 
                    Cam_to_board[self.config['camera_indexes'].index(cam_index),i,:,:] = np.linalg.inv(Cam_to_board[self.config['camera_indexes'].index(cam_index),i,:,:])
                    Trans_rot_180_x = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                    Cam_to_board[self.config['camera_indexes'].index(cam_index),i,:,:] = Cam_to_board[self.config['camera_indexes'].index(cam_index),i,:,:] @ Trans_rot_180_x
                    Cam_to_board_params[self.config['camera_indexes'].index(cam_index),i,0:3] = Cam_to_board[self.config['camera_indexes'].index(cam_index),i,0:3,3]
                    Cam_to_board_params[self.config['camera_indexes'].index(cam_index),i,3:6] = R.from_matrix(Cam_to_board[self.config['camera_indexes'].index(cam_index),i,0:3,0:3]).as_euler('xyz',degrees=False)
                    Cam_to_board_poses[i+photo_poses.shape[0]*self.config['camera_indexes'].index(cam_index),:3] = Cam_to_board[self.config['camera_indexes'].index(cam_index),i,0:3,3]
                   
                    Cam_to_board_poses[i+photo_poses.shape[0]*self.config['camera_indexes'].index(cam_index),3:7] = R.from_matrix(Cam_to_board[self.config['camera_indexes'].index(cam_index),i,0:3,0:3]).as_quat()
                else:
                    print(i) 

        np.savetxt('/home/arminkaroly/Munka/Virt_Twin/src/Calibration/Cam_poses.csv', Cam_to_board_poses[photo_poses.shape[0]:,:], delimiter=",")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(photo_poses[:,0],photo_poses[:,1],photo_poses[:,2],c='b')
        ax.scatter(Cam_to_board_params[0,:,0],Cam_to_board_params[0,:,1],Cam_to_board_params[0,:,2],c='r')
        ax.scatter(Cam_to_board_params[1,:,0],Cam_to_board_params[1,:,1],Cam_to_board_params[1,:,2],c='g')
        plt.title("Robot flange positions")
        plt.xlabel("X [meter]")
        plt.ylabel("Y [meter]")
        ax.set_zlabel("Z [meter]")
        plt.show()


        if not os.path.exists(self.path + '/src/Calibration/Results'):
            os.makedirs(self.path + '/src/Calibration/Results')
        if not os.path.exists(self.path + '/src/Calibration/Results/Annotations_test_without_robot'):
            os.makedirs(self.path + '/src/Calibration/Results/Annotations_test_without_robot')

        print("Start rendering")

        import requests
        import shutil
        if not os.path.exists(self.path + '/src/Calibration/Results/Annotations_results'):
            os.makedirs(self.path + '/src/Calibration/Results/Annotations_results')
        Content_type = {'Content-Type': 'application/json',}
        render = {'render': {'annotation': True,},}

        for cam_index in range(len(self.config['camera_indexes'])):

            self.camera_index = self.config['camera_indexes'][int(cam_index)]
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

            self.render_path = self.config['path_to_rendered_img']
            for i in range(Cam_to_board_params.shape[1]):
                camera_pose = {
                    'pose': {
                        'name': 'Camera',
                        'location': Cam_to_board_params[cam_index,i,:3].tolist(),
                        'rotation': Cam_to_board_params[cam_index,i,3:].tolist(),
                    }
                }
                requests.post('http://localhost:12345', headers=Content_type, json=camera_pose)
                requests.post('http://localhost:12345', headers=Content_type, json=render)
                while True:
                    if os.path.exists(self.render_path.rsplit("/",1)[0] + "/annotations/0001.exr"):
                        time.sleep(0.5)
                        shutil.copyfile(self.render_path.rsplit("/",1)[0] + "/annotations/0001.exr", self.path + '/src/Calibration/Results/Annotations_test_without_robot/' + str(cam_index*Cam_to_board.shape[1]+i).zfill(4) + '.exr')
                        if os.path.exists(self.path + '/src/Calibration/Results/Annotations_test_without_robot/' + str(cam_index*Cam_to_board.shape[1]+i).zfill(4) + '.exr'):
                            time.sleep(0.5)
                            os.remove(self.render_path.rsplit("/",1)[0] + "/annotations/0001.exr")
                            break
                    else:
                        time.sleep(0.1)

                from data_generator.exr_reader import OpenEXRReader
                try:
                    chstr = 'i'
                    with OpenEXRReader(self.path + '/src/Calibration/Results/Annotations_test_without_robot/' + str(cam_index*Cam_to_board.shape[1]+i).zfill(4) + '.exr', chstr) as exr:
                        mask = np.reshape(exr.i,(exr.resolution))
                        Annotation = self.Colorize_labels(mask)
                except AttributeError:
                    print('Could not access channel(s) "{}", because they are not loaded!'.format(chstr))
                real_img = cv2.imread(self.path + '/src/Calibration/Real_Images/camera_' + str(self.config['camera_indexes'][cam_index]) + '/' + str(i).zfill(4) + '.png')
                RESULTS = np.zeros(real_img.shape,dtype=real_img.dtype)
                RESULTS[:,:,:] = (0.5 * real_img[:,:,:]) + ((0.5) * Annotation[:,:,:])
                cv2.imwrite(self.path + '/src/Calibration/Results/Annotations_results/' + str(cam_index*Cam_to_board.shape[1]+i).zfill(4) + '.png', RESULTS)
                print(cam_index*Cam_to_board.shape[1]+i)
        
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
    
    def Check_poses(self):
        import matplotlib.pyplot as plt     
        self.photo_poses = np.genfromtxt(self.path + "/src/Calibration/robot_poses.csv", delimiter=',')
        calc_poses = np.genfromtxt(self.path + "/src/Calibration/calibration_photo_poses.csv", delimiter=',')
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(calc_poses[:,0],calc_poses[:,1],calc_poses[:,2],c='b')
        ax.scatter(self.photo_poses[:,0],self.photo_poses[:,1],self.photo_poses[:,2],c='r')
        plt.show()

        error = np.abs(calc_poses[:,0:3]-self.photo_poses[:,0:3])
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(error[:,0],error[:,1],error[:,2],c='g')
        plt.show()

        eul_errors = []
        for i in range(calc_poses.shape[0]):
            photo_pose_ori = R.from_quat(self.photo_poses[i,3:]).as_matrix()
            calc_pose_ori = R.from_quat(calc_poses[i,3:]).as_matrix()
            Rot_mat_error = photo_pose_ori @ np.transpose(calc_pose_ori)
            Eul_error = R.from_matrix(Rot_mat_error).as_euler('xyz',degrees=True)
            eul_errors.append(Eul_error)
        
        eul_errors = np.array(eul_errors)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(eul_errors[:,0],eul_errors[:,1],eul_errors[:,2],c='g')
        plt.show()



    def Fitness_ori(self,parameters,Base_Flange, Cam_to_board):
        # Define Rot matrix from the parameters
        R_Board_Base = np.zeros((3,3))
        R_Board_Base = R.from_euler('xyz',parameters[0:3],degrees = True).as_matrix()
        R_Flange_Cam = np.zeros((int(len(parameters)/3-1),3,3))
        param_flange_cam = parameters[3:]
        for i in range(int(len(param_flange_cam)/3)):
            R_Flange_Cam[i,:,:] = R.from_euler('xyz',param_flange_cam[i*3:i*3+3],degrees= True).as_matrix()
        # Calculate the rotation error as the sum of the rotation vectors length
        sum_oris_error = np.array([0])
        oris_error = np.array([[0,0,0]])
        for i in range(Cam_to_board.shape[0]):
            for j in range(Cam_to_board.shape[1]):
                Rot_error = Base_Flange[j,0:3,0:3] @ R_Flange_Cam[i,:,:] @ Cam_to_board[i,j,0:3,0:3] @ np.linalg.inv(R_Board_Base)
                sum_oris_error = np.append(sum_oris_error,np.linalg.norm(R.from_matrix(Rot_error).as_rotvec(degrees = True)))
                oris_error = np.append(oris_error,np.reshape(R.from_matrix(Rot_error).as_rotvec(degrees = True),(1,3)),axis = 0)
        sum_oris_error = sum_oris_error[1:]
        oris_error = oris_error[1:,:]
        if self.print_ and self.print_counter == 150:
            print(format_text("Average orientation error: " + str(np.average(sum_oris_error)) + " [deg]",self.Width,"white","separated"))
            self.print_counter = 0
        self.print_counter += 1
        return np.average(sum_oris_error)

    def Fitness_pos(self,parameters,Base_Flange, Cam_to_board,T_Base_Board,T_Flange_Cam):
        T_Base_Board[0:3,3] = parameters[:3]
        for i in range(T_Flange_Cam.shape[0]):
            T_Flange_Cam[i,0:3,3] = parameters[(i+1)*3:(i+2)*3]
        sum_pos_error = np.array([0])
        pos_error = np.array([[0,0,0]])
        for i in range(Cam_to_board.shape[0]):
            for j in range(Cam_to_board.shape[1]):
                Trans_error = (Base_Flange[j,:,:] @ T_Flange_Cam[i,:,:] @ Cam_to_board[i,j,:,:] @ np.linalg.inv(T_Base_Board))[0:3,3]
                sum_pos_error = np.append(sum_pos_error,np.linalg.norm(Trans_error))
                pos_error = np.append(pos_error,np.array([Trans_error]),axis = 0)
        sum_pos_error = sum_pos_error[1:]
        pos_error = pos_error[1:,:]
        if self.print_ and self.print_counter == 150:
            print(format_text("Average position error: " + str(np.average(sum_pos_error)) + " [meter]",self.Width,"white","separated"))
            self.print_counter = 0
        self.print_counter += 1
        return np.average(sum_pos_error)

'''def main():
    valami = Image_taker()
    valami.Move_robot_n_take_imgs(calibration=True)
    valami.Calibrate_cameras()
    valami.Calibrate_flange_Cam()
    del valami'''

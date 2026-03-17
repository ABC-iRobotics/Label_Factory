#! /usr/bin/env python

import os
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R

class PoseGenerator:
    def __init__(self):
        pass
    def orbiting(self, pose, axis_name, alpha, radius):
        '''
        This function rotates a point around a center point given by radius. You can specify the _name and the rotation angles
        Arguments:
            - pose: position we want to multiply. This is the middle point.
            - axis_name (str): specify the axis (x or y)
            - alpha: rotation angle in radian
            - radius: rotation radius in mm
        Returns:
            - output_poses: the generated poses
        '''
        rob_pos = pose[0:3]
        rob_ori = pose[3:6]
        rot_matrix = R.as_matrix(R.from_euler('xyz', rob_ori))
        cam_ray = np.reshape(np.matmul(rot_matrix, np.array([[0], [0], [1]])), (1, 3))

        targ_pos = rob_pos + radius * cam_ray
        targ_ori = rob_ori + [np.pi, 0, 0]
        targ_rot_matrix = R.as_matrix(R.from_euler('xyz', targ_ori))

        if (axis_name == 'x'):
            rot_matrix = R.as_matrix(R.from_euler('xyz', [alpha, 0, 0]))
        elif (axis_name == 'y'):
            rot_matrix = R.as_matrix(R.from_euler('xyz', [0, alpha, 0]))

        rot_matrix = np.matmul(targ_rot_matrix, rot_matrix)
        ray = np.reshape(np.matmul(rot_matrix, np.array([[0], [0], [1]])), (1, 3))

        new_pos = targ_pos + ray * radius
        new_ori = R.from_matrix(rot_matrix).as_euler('xyz') - [np.pi, 0, 0]
        output_poses = np.append(new_pos, new_ori)

        return output_poses

    def increse_distance(self, pose, r_step, step):
        '''
        This function increases the distance of a point from the center.
        
        Arguments:
            - pose: pose whose distance we want to increase.
            - r_step: the rate of increase in mm
            - step: number of new points

        Returns:
            - output_points: the increased points
        '''

        rob_pos = pose[0:3]
        rob_ori = pose[3:6]  
        rot_matrix = R.as_matrix(R.from_euler('xyz',rob_ori))
        cam_ray = np.reshape(np.matmul(rot_matrix,np.array([[0], [0], [1]])), (1, 3))

        rob_pos = rob_pos - r_step * cam_ray * step
        output_points = np.append(rob_pos, rob_ori)

        return output_points

    def rotate(self, pose, axis_name, alpha):
        '''
        This function rotates a point around a center point in a specified axis.
        
        Arguments:
            - pose: position we want to rotate
            - axis_name (str): specify the axis (x or y)
            - alpha: rotation angle in radian

        Returns:
            - output_poses: the rotated poses
        '''

        rob_pos = pose[0:3]
        rob_ori = pose[3:6]
        mat = R.as_matrix(R.from_euler('xyz',rob_ori))

        if (axis_name == 'x'):
            rot_matrix = R.as_matrix(R.from_euler('xyz',[alpha, 0, 0]))
        elif (axis_name == 'y'):
            rot_matrix = R.as_matrix(R.from_euler('xyz',[0, alpha, 0]))

        Mat = np.matmul(rot_matrix,mat)
        rob_ori = R.from_matrix(Mat).as_euler('xyz')
        output_poses = np.append(rob_pos,rob_ori)

        return output_poses

    def generate_orbitings(self, start_pose, num_pos_x, num_pos_y, angle_pos_x, angle_pos_y, radius):
        '''
        This function generates spherical grid points around the object. All orientation points to the center of the sphere.

        Arguments:
            - start_pose (np.array): start pose [x,y,z,rx,ry,rz] mm and radian (Euler-angles)

            - num_pos_x : number of grid points in x direction (if even, add plus one)
            - num_pos_y : number of grid points in y direction (if even, add plus one)

            - angle_pos_x: x direction of the angle you turn in radian
            - angle_pos_y : y direction of the angle you turn in radian (these values are used to adjust how far apart the grid points should be)
            - radius: the selected initial spherical radius in mm (if you want to orbit the robot around the object and you are standing directly above it, you enter the z value)
        
        Returns:
            - poses: the generated spherical grid points. [x,y,z,rx,ry,rz] mm and radian (Euler-angles)
        
        '''
        # If even, add plus one
        if (num_pos_x % 2) == 0:
            num_pos_x = num_pos_x + 1
        if (num_pos_y % 2) == 0:
            num_pos_y = num_pos_y + 1
        
        poses = np.zeros((num_pos_x * num_pos_y,6))
        k = 0
        
        ## If the radius is None the radius is calculated to be equal to the distance from the x-y plain
        if radius == None:
            radius = -start_pose[2]/np.cos(start_pose[3])/np.cos(start_pose[4])
        
        for i in range(int(-(num_pos_x - 1) / 2),int((num_pos_x - 1) /2 ) + 1):
            l = 0
            for j in range(int(-(num_pos_y - 1) / 2), int((num_pos_y - 1) / 2) + 1):
                poses_curr = start_pose
                poses_curr = self.orbiting(poses_curr, 'x', angle_pos_x * i, radius)
                poses_curr = self.orbiting(poses_curr, 'y', angle_pos_y * j, radius)
                poses[k * num_pos_y + l, :] = poses_curr
                l = l + 1
            k = k + 1

        return poses

    def generate_spheres(self, orbiting_poses, r_step, steps):
        '''
        This function generates further spheres around the the original one with the same center.

        Arguments:
            - orbiting_poses (np.array): spherical poses, ez csak az eddigi pontokból képez újabb pontokat, eltávolítja a pontokat a középponttól
            - r_step: step size in mm (how much the radius increases compared to the previous one)
            - steps: number of spheres

        Returns:
            - poses_spheres: the generated sphere positions
        '''
        poses_spheres = np.zeros((orbiting_poses.shape[0] * steps, 6))
        for i in range(steps):
            for j in range(orbiting_poses.shape[0]):
                poses_spheres[i * orbiting_poses.shape[0] + j, :] = self.increse_distance(orbiting_poses[j, :], r_step, i)

        return poses_spheres

    def generate_poses(self, poses, num_ori_x, num_ori_y, ori_x_angle, ori_y_angle):
        '''
        This function generates different orientations at each position so that not all frames point towards the centre of the sphere.

        Arguments:
            - poses (np.array): spherical poses
            - num_ori_x : number of orientation changes on the x-axis (this also does it in a square grid, if even, add plus one)
            - num_ori_y : number of orientation changes on the y-axis (this also does it in a square grid, if even, add plus one)
            - ori_x_angle : angular change in x direction in radian
            - ori_y_angle: angular change in y direction in radian
        
        Returns:
            - poses_ori: generated poses with different orientations
        '''

        # If even, add plus one
        if (num_ori_x % 2) == 0:
            num_ori_x = num_ori_x + 1
        if (num_ori_y % 2) == 0:
            num_ori_y = num_ori_y + 1
        poses_ori = np.zeros((poses.shape[0] * num_ori_x * num_ori_y, 6))
        for i in range(poses.shape[0]):
            pose = poses[i,:]
            l = 0
            for j in range(int(-(num_ori_x - 1) / 2),int((num_ori_x - 1) / 2) + 1):
                m = 0
                for k in range(int(-(num_ori_y - 1) / 2),int((num_ori_y - 1) / 2) + 1):
                    poses_curr = pose
                    poses_curr = self.rotate(poses_curr, 'x', ori_x_angle * j)
                    poses_curr = self.rotate(poses_curr, 'y', ori_y_angle * k)
                    poses_ori[i * num_ori_x * num_ori_y + l * num_ori_y + m, :] = poses_curr
                    m = m + 1
                l = l + 1
        poses_ori = np.round(poses_ori, 4)

        return poses_ori

if __name__=='__main__':
    pose_generator = PoseGenerator()

    ## OE side start_pose (camera)
    start_pose = [0, 0, 0, 0, 3.1415924, 3.1415924/2] # mm and Euler-angles in radia
    
    radius = None
    
    # start pose, number of grid points in x and y direction, ...
    orbiting_poses = pose_generator.generate_orbitings(start_pose=np.array(start_pose), num_pos_x=3, num_pos_y=3, angle_pos_x=10/180*np.pi, angle_pos_y=10/180*np.pi, radius=radius)

    # orbiting_poses: Sphere layer calculated so far, layer spacing, number of layers
    sphere_poses = pose_generator.generate_spheres(orbiting_poses=orbiting_poses, r_step=70, steps=1)

    # 3D grid; the number of orientations in the given position (add one to the even)
    calibration_photo_poses = pose_generator.generate_poses(poses=sphere_poses, num_ori_x=1, num_ori_y=1, ori_x_angle=4/180*np.pi, ori_y_angle=4/180*np.pi)

    calibration_photo_poses[:,0:3] = calibration_photo_poses[:,0:3] / 1000 # switch to meter
    
    # Save generated poses to csv file
    with open(os.path.join('generated_poses_2.csv'), mode='w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        for pose in calibration_photo_poses:
            # Convert Euler-angles to quaternion
            q = R.from_euler('xyz', [pose[3],pose[4],pose[5]],degrees=False).as_quat() # [x,y,z,w] format
            pose_to_write = [pose[0], pose[1], pose[2], q[0], q[1], q[2],q[3]] # quaternion in [x,y,z,w] format
            writer.writerow(pose_to_write)

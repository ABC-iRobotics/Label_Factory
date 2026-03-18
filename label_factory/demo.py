#!/usr/bin/env python3
from Image_taker import Image_taker
from BAT_scene_opt import BAT_Optimizer
import time  
def main():
    IMG_TAKER = Image_taker(False)
    BAT_OPT = BAT_Optimizer()

    BAT_OPT.Define_cam_poses()
    BAT_OPT.Setup_camera(1)
    BAT_OPT.Set_Object_pose('Cube')
            

    for _ in range(10):
        IMG_TAKER.Move_robot_n_take_imgs(mm_param = [3,10,3,10,1,0.05,1,5,1,5,None,5], calibration = False)
        BAT_OPT.Define_cam_poses()
        BAT_OPT.Setup_camera(1)
        BAT_OPT.Set_Object_pose('Cube')
        time.sleep(30)
        path_r = '/home/arminkaroly/Munka/Virt_Twin/src/Calibration/Results'

if __name__ == '__main__':
    main()
    

#!/usr/bin/env python3
import cmd
import time
from vispy import app
from colorama import Fore, Style
import subprocess
from data_generator.BAT_scene_opt import BAT_Optimizer
from data_generator.Image_taker import Image_taker 

class MyCLI(cmd.Cmd):
    prompt = '*** >>'
    def preloop(self):
        "Clear the cmd before running the loop"
        subprocess.run("clear")
        self.Menu_id = 0
        self.Width = 72
        main_menu = closed_text("Main menu",self.Width,"green","center")
        Description = closed_text("This system was created to generate real camera images, with corresponding annotations, to teach neural networks for object detection.",self.Width,"yellow","left")
        Functions = closed_text(Fore.GREEN + "Commands:" + Fore.WHITE + "\nstart_cam_calib:" + Fore.WHITE +  " Start the camera calibration process.\nstart_cam_flange_calib:" + Fore.WHITE + " Start the Camera-Flange calibration process.\nstart_real_images:" + Fore.WHITE + " Start real image creating process.\nstart_optimization:" + Fore.WHITE + " Start the pose optimization precess.\nresults:" + Fore.WHITE + "Create some images, that shows the results.\nsetup_menu:" + Fore.WHITE + " Define the advanced parameters for the processes. \nhelp <command name>:" + Fore.WHITE + " Describe the command in a detailed form.\nclear:" + Fore.WHITE + " Clear the terminal.\nexit:" + Fore.WHITE + " Close the program.",self.Width,"cyan","separated")
        print(closed_text("Welcome to BAT Digital Twin Real Part Operator.",self.Width,'yellow',"center"))
        self.intro = main_menu + Description + Functions
        self.param1 = False
        self.param3 = [3,10,3,10,1,0.25,1,5,1,5,None,5]
        self.param4 = 1
        self.param5 = "Cube"
        self.param6 = "all"
        self.param7 = 2
        self.param8 = 15
        self.img_taker = Image_taker(self.param1)
        self.BAT_opt = BAT_Optimizer()
                
    def postloop(self):
        # Add custom cleanup or finalization here
        print(closed_text("Goodbye!",self.Width,"green","center"))
    
    def emptyline(self):
        print(Fore.RED +"No command was detected." + Fore.WHITE) 
    
    def default(self, line):
        print(Fore.RED +"The command " + line  + " is not found." + Fore.WHITE) 
    
    def do_start_cam_calib(self, line):
        print(closed_text("The image taking process is started!",self.Width,"white","left"))
        output = self.img_taker.Move_robot_n_take_imgs(mm_param = self.param3, calibration = True)
        if output:
            print(closed_text("The camera calibration process started!",self.Width,"white","left"))
            self.img_taker.Calibrate_cameras()
            print(closed_text("The config.yaml file is updated with the camera parameters!",self.Width,"green","center"))
        
    def do_start_cam_flange_calib(self,line):
        print(closed_text("The flange-camera transformation optimization process started!",self.Width,"white","left"))
        self.img_taker.Calibrate_flange_Cam()
        print(closed_text("The config.yaml file is updated with the Flange-Camera-Transformation matrices!",self.Width,"green","center"))
        
    def do_start_real_images(self,line):
        print(closed_text("The image taking process is started!",self.Width,"white","left"))        
        self.img_taker.Move_robot_n_take_imgs(mm_param = self.param3, calibration = False)
        
    def do_start_optimization(self,line):
        print(closed_text("Calculating camera poses started!",self.Width,"white","left"))
        output = self.BAT_opt.Define_cam_poses()
        if output:
            print(closed_text("Setting up the BAT camera is started!",self.Width,"white","left"))
            output = self.BAT_opt.Setup_camera(self.param4)
            if output:
                print(closed_text("Setting the pose of the object is started!",self.Width,"white","left"))
                output = self.BAT_opt.Set_Object_pose(self.param5)
                if output:
                    print(closed_text("The image rendering process is started",self.Width,"white","left"))
                    self.BAT_opt.Render_images(self.param5, self.param7)
                    print(closed_text("The optimization window is opened!",self.Width,"white","left"))
                    self.BAT_opt.Create_window()
                    
    def do_results(self,line):
        print(closed_text("Creating the result images!",self.Width,"white","left"))
        output = self.BAT_opt.Setup_camera(self.param4)
        if output:
            self.BAT_opt.Show_results(self.param8,self.param5) 

    def do_setup_menu(self,line):
        subprocess.run(["clear"])
        self.Menu_id = 1
        setup_text = Fore.GREEN + "Commands:\ndesc <parameter_name>:" + Fore.WHITE + "Describe the parameter properly.\n<parameter_name> <value>:" + Fore.WHITE + "Set the parameters to the defined value.\nreset <parameter_name>:" + Fore.WHITE + "Reset the selected parameter to default.\nreset_all:" + Fore.WHITE + "Reset all parameters to default.\nclear:" + Fore.WHITE + "Clear the terminal.\nback:"+ Fore.WHITE +" Return to the Main menu.\n"  
        paramters1 = "log_info:" + Fore.WHITE + "\t"+ str(self.param1) +"\n"
        paramters3 = "mm_param:" + Fore.WHITE + "\t"+ str(self.param3).replace(" ","") +"\n"
        paramters4 = "camera_index:" + Fore.WHITE + "\t"+ str(self.param4) +"\n"
        paramters5 = "object_name:" + Fore.WHITE + "\t"+ str(self.param5) +"\n"
        paramters6 = "vertices_num:" + Fore.WHITE + "\t"+ str(self.param6) +"\n"
        paramters7 = "opt_img:" + Fore.WHITE + "\t"+ str(self.param7) +"\n"
        paramters8 = "result_img:" + Fore.WHITE + "\t"+ str(self.param8) +"\n"
        text_1 = closed_text("Advanced setups menu",self.Width,"green","center")
        text_2 = closed_text(setup_text,self.Width,"cyan","separated")
        self.setups = closed_text(paramters1+paramters3+paramters4+paramters5+paramters6+paramters7+paramters8,self.Width,"magenta","separated")
        print(text_1 + text_2 + self.setups)
        # Image taker:          default
        #       print_ :        False
        #       mm_paramteres:  [3,10,3,10,1,0.25,1,5,1,5,None,5]
        
        # BAT_scene_opt:
        #       camera_index:   -1
        #       object_name:    "object"
        #       vertexes_num:   "all"
        #       image_number:   10
        #       result_images:  15
    def do_log_info(self,line):
        if self.Menu_id == 1:
            if line == "True" or line == "False":
                self.param1 = line
                self.do_setup_menu(line)
            else: 
                print(Fore.RED + "Cannot set the log_info parameter to " + line + "." + Fore.WHITE)
        elif self.Menu_id == 0:
            print(Fore.RED +"The command log_info " + line  + " is not found." + Fore.WHITE) 
            
    def do_mm_param(self,line):
        if self.Menu_id == 1:
            separated = line.replace(" ","")
            separated = separated[1:]
            separated = separated[:-1]
            separated = separated.split(",")
            n = len(separated)
            my_bool = True
            for i in separated:
                try:
                    if separated.index(i) == 10 and i == 'None':
                        my_bool = my_bool and True
                        separated[10] = None
                    elif separated.index(i) in [0,2,4,6,8,11]:
                        separated[separated.index(i)] = int(i)
                    else:
                        separated[separated.index(i)] = float(i)
                except:
                    my_bool = False
            if n == 12 and my_bool :
                self.param3 = separated
                self.do_setup_menu(separated)
            else: 
                print(Fore.RED + "Cannot set the mm_param parameter to " + line + "." + Fore.WHITE)
        elif self.Menu_id == 0:
            print(Fore.RED +"The command mm_param " + line  + " is not found." + Fore.WHITE) 
    
    def do_camera_index(self,line):
        if self.Menu_id == 1:
            if line.isdigit() or line[1:].isdigit():
                self.param4 = line
                self.do_setup_menu(line)
            else: 
                print(Fore.RED + "Cannot set the camera_index parameter to " + line + "." + Fore.WHITE)
        elif self.Menu_id == 0:
            print(Fore.RED +"The command camera_index " + line  + " is not found." + Fore.WHITE) 
    
    def do_object_name(self,line):
        if self.Menu_id == 1:
            if line != "":
                self.param5 = line
                self.do_setup_menu(line)
            else: 
                print(Fore.RED + "Cannot set the object_name parameter to '" + line + "'." + Fore.WHITE)
        elif self.Menu_id == 0:
            print(Fore.RED +"The command object_name " + line  + " is not found." + Fore.WHITE) 
    
    def do_vertices_num(self,line):
        if self.Menu_id == 1:
            separated = line.split(',')
            my_bool = True
            for i in separated:
                my_bool = my_bool and i.isdigit()
            if line == "all":
                self.param6 = line
                self.do_setup_menu(line)
            elif my_bool and len(separated)>0:
                self.param6 = line
                self.do_setup_menu(line)
            else: 
                print(Fore.RED + "Cannot set the vertices_num parameter to " + line + "." + Fore.WHITE)
        elif self.Menu_id == 0:
            print(Fore.RED +"The command vertices_num" + line  + " is not found." + Fore.WHITE) 
    
    def do_opt_img(self,line):
        if self.Menu_id == 1:
            if line.isdigit():
                self.param7 = line
                self.do_setup_menu(line)
            else: 
                print(Fore.RED + "Cannot set the opt_img parameter to " + line + "." + Fore.WHITE)
        elif self.Menu_id == 0:
            print(Fore.RED +"The command opt_img " + line  + " is not found." + Fore.WHITE) 
    
    def do_result_img(self,line):
        if self.Menu_id == 1:
            if line.isdigit():
                self.param8 = line
                self.do_setup_menu(line)
            else: 
                print(Fore.RED + "Cannot set the result_img parameter to " + line + "." + Fore.WHITE)
        elif self.Menu_id == 0:
            print(Fore.RED +f"The command result_img {line} is not available in this menu." + Fore.WHITE) 
    
    def do_reset(self, line):
        if self.Menu_id != 1:
            print(Fore.RED + f"The command reset {line} is not available in this menu." + Fore.WHITE)
            return
        param_name = line.strip()
        default_values = {
            "log_info": False,
            "mm_param": [3, 10, 3, 10, 1, 0.25, 1, 5, 1, 5, None, 5],
            "camera_index": -1,
            "object_name": "object",
            "vertices_num": "all",
            "opt_img": 10,
            "result_img": 15
        }
        param_mapping = {
            "log_info": "param1",
            "mm_param": "param3",
            "camera_index": "param4",
            "object_name": "param5",
            "vertices_num": "param6",
            "opt_img": "param7",
            "result_img": "param8"
        }
        if param_name in param_mapping:
            setattr(self, param_mapping[param_name], default_values[param_name])
            self.do_setup_menu(line)
        else:
            print(Fore.RED + f"The parameter '{param_name}' cannot be reset (not recognized)." + Fore.WHITE)

    def do_desc(self, line):
        if self.Menu_id != 1:
            print(Fore.RED + f"The command desc {line} is not available in this menu." + Fore.WHITE)
            return
        param_name = line.strip()
        param_descriptions = {
            "log_info": "boolian\nEnable or disable logging of information during processes.",
            "mm_param": "list:[int,float,int,float,int,float,int,float,int,float,None/float,int]\nMost of the parameters can be seen in the PoseGenerator class.\nSet the parameters for the movement of the robot.\nThe following order of the parameters is the same like in the list.\nOrbiting number in x axis: int.\nOrbiting degree in x axis: float[°].\nOrbiting number in y axis: int.\nOrbiting degree in y axis: float[°].\nThe number of sphere surfaces: int.\nThe distance between sphere surfaces: float[m].\nNumber of orientations in x axis: int.\nRotation degree in the x axis: float[°].\nNumber of orientations in y axis: int.\nRotation degree in the y axis: float[°].\nThe radius of the first sphere: float[m]/None if the radius is equal to the distance from the z=0 surface.\nThe camera image buffer size: int.",
            "camera_index": "integer\nIndex of the USB camera to be used.",
            "object_name": "string\nName of the object to be annotated.",
            "vertices_num": "string:'all' or list without brackets: int,int,int, ...\nIndex number of vertices to use during the optimization.",
            "opt_img": "integer\nNumber of images used during optimization.",
            "result_img": "integer\nNumber of result images."
        }
        if param_name in param_descriptions:
            description = f"{Fore.CYAN + param_name + Fore.WHITE}: {param_descriptions[param_name]}"
            print(closed_text(description, self.Width, "white", "left"))
        else:
            error_message = f"No description available for parameter '{param_name}'."
            print(Fore.RED + error_message + Fore.WHITE)

    def do_reset_all(self,line):
        if self.Menu_id == 1:
            self.param1 = False
            self.param3 = [3,10,3,10,1,0.25,1,5,1,5,None,5]
            self.param4 = -1
            self.param5 = "object"
            self.param6 = "all"
            self.param7 = 10
            self.param8 = 15 
            self.do_setup_menu(line)
        elif self.Menu_id == 0:
            print(Fore.RED +"The command reset_all is not found." + Fore.WHITE) 
    
    def do_back(self,line):
        if self.Menu_id == 1: 
            subprocess.run(["clear"])
            self.Menu_id = 0
            print(self.intro)
            self.img_taker.print_ = self.param1
        elif self.Menu_id == 0:
            print(Fore.RED +"The command back is not found." + Fore.WHITE) 
            
    def do_clear(self,line):
        subprocess.run("clear")
        if self.Menu_id == 0:
            print(self.intro)   
        elif self.Menu_id == 1:
            self.do_setup_menu(line)
    
    def do_exit(self, line):
        return True

    def do_MMS_B_T(self,line):
        self.BAT_opt.MMS_B_T()

    def do_Valid_cam(self,line):
        self.img_taker.Validate_cam_calib()

    def do_PoC_R2B(self,line):
        self.img_taker.PoC_R2B()

    def do_Robot_Pose_Check(self,line):
        self.img_taker.Check_poses()

    def do_Demo(self,line):
        for _ in range(10):
            self.do_start_real_images(line)
            self.do_results(line)

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
        
def main():
    MyCLI().cmdloop()

if __name__ == '__main__':
    app.use_app('pyqt5')
    main()

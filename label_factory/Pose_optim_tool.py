import sys
import numpy as np
import cv2
import yaml
from PyQt5 import QtCore, QtGui, QtWidgets
from vispy import app, scene
from colorama import Fore, Style
from scipy.spatial.transform import Rotation as R
from vispy.geometry import MeshData
from scipy.spatial import ConvexHull
from label_factory.utils import *

class Feature_point_data():
    def __init__(self):
        self.datas = []
        self.indexes = []
        self.axs0_points = []
        self.axs1_points = []

def SVD_pose_diff_estimation(Points_Blender, Points_estimated):
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

def split_quads_to_triangles(faces: np.ndarray) -> np.ndarray:
    """
    Converts quads into triangles (splitting each quad into two triangles).
    Each quad (4 vertices) is turned into 2 triangles.
    Returns an array of shape (2*F, 3) where F is the number of quads.
    """
    # Split each quad (4 vertices) into two triangles
    triangles = []
    for face in faces:
        if len(face) == 4:  # Quad face
            triangles.append([face[0], face[1], face[2]])
            triangles.append([face[0], face[2], face[3]])
        else:  # Triangle face
            triangles.append(face)
    
    return np.array(triangles, dtype=np.uint32)  # (2*F, 3) triangles

class ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(QtCore.QPoint)
    click_type = None  # "left" or "right"
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.click_type = "left"
        elif event.button() == QtCore.Qt.RightButton:
            self.click_type = "right"
        elif event.button() == QtCore.Qt.MiddleButton:
            self.click_type = "middle"
        self.clicked.emit(event.pos())
        super().mousePressEvent(event)

# -------------------------
# PyQt widget
# -------------------------
class QtPoseTool(QtWidgets.QWidget):
    def __init__(self, All_image_datas, config, path, camera_pose_matrix, camera_index, object_name, Width, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Object pose optimization tool")
        self.showMaximized()

        self.All_image_datas = All_image_datas
        self.config = config
        self.path = path
        self.camera_pose_matrix = camera_pose_matrix
        self.camera_index = camera_index
        self.object_name = object_name
        self.Width = Width

        self.curr_img_inx = 0
        self.Feature_datas = Feature_point_data()

        main = QtWidgets.QGridLayout(self)
        main.setContentsMargins(6, 6, 6, 6)
        main.setSpacing(6)

        # VisPy canvas
        self.canvas = scene.SceneCanvas(keys=None, size=(800, 800), bgcolor="white", show=False)
        self.canvas.create_native()  # ensure native is created using the Qt backend
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45, azimuth=0, roll=0, elevation=90, distance=2.0)
        self.mesh_visual = None
        self.sel_markers = scene.visuals.Markers()
        self.view.add(self.sel_markers)
        self.real_markers = scene.visuals.Markers()
        self.view.add(self.real_markers)
        
        # create once and reuse (prevents stray visuals)
        self.mesh_visual = scene.visuals.Mesh(parent=self.view.scene)
        self.edge_visual = scene.visuals.Line(parent=self.view.scene)  # optional overlay

        native = self.canvas.native
        if hasattr(native, "setMinimumSize"):
            native.setMinimumSize(1000, 1000)
        main.addWidget(native, 0, 0, 2, 1)

        # Blender image
        self.blender_label = ClickableLabel()
        self.blender_label.setAlignment(QtCore.Qt.AlignCenter)
        self.blender_label.setStyleSheet("background:#111;")
        main.addWidget(self.blender_label, 0, 1)

        # Real image
        self.real_label = ClickableLabel()
        self.real_label.setAlignment(QtCore.Qt.AlignCenter)
        self.real_label.setStyleSheet("background:#111;")
        main.addWidget(self.real_label, 1, 1)

        self.img_label_size = (int(960), int(540))
        self.img_label_size_small = (int(960), int(540))
        self.blender_label.setFixedSize(self.img_label_size_small[0], self.img_label_size_small[1])
        self.real_label.setFixedSize(self.img_label_size[0], self.img_label_size[1])
        self.blender_label.setScaledContents(True)
        self.real_label.setScaledContents(True)


        controls = QtWidgets.QHBoxLayout()
        self.next_btn = QtWidgets.QPushButton("Next")
        controls.addWidget(self.next_btn)

        # Add the counter label
        self.real_points_counter = QtWidgets.QLabel("Real points: 1")
        self.real_points_counter.setStyleSheet("font-weight: bold;")
        controls.addWidget(self.real_points_counter)
        controls.addStretch(1)
        main.addLayout(controls, 2, 0, 1, 2)

        # Events
        self.next_btn.clicked.connect(self.on_next)
        self.real_label.clicked.connect(self.on_real_image_click)
        self.canvas.events.mouse_press.connect(self.on_canvas_mouse_press)
        self.all_intersection_point = []
        self.all_intersection_point_dist = []
        self.sel_texts = []
        self.all_real_points = []
        

        data = self.All_image_datas[self.curr_img_inx]
        verts = np.asarray(data.World_coords, float)
        faces = np.asarray(data.Faces, int)
        faces = split_quads_to_triangles(faces)
        # (N,4) per-face RGBA
        facecolors = np.full((faces.shape[0], 4), [0.56, 0.93, 0.56, 0.75], dtype=np.float32)
        edge_color = np.array([0.5, 0.5, 0.5, 1.0])

        # create fresh visuals
        self.mesh_visual = scene.visuals.Mesh(vertices=verts, faces=faces,
                                            face_colors=facecolors, shading=None,
                                            parent=self.view.scene)
        
        self.edge_visual = scene.visuals.Mesh(vertices=verts, faces=faces,
                                            color=edge_color, mode='lines',
                                            parent=self.view.scene)

        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)
        center = (bb_min + bb_max) / 2.0
        self.view.camera.center = center
        self._draw_current()

    def on_real_image_click(self, event):
        w,h = self.All_image_datas[self.curr_img_inx].image_real.shape[0:2]
        if  self.real_label.click_type == "left":
            x, y = event.x()*h/self.img_label_size[0], event.y()*w/self.img_label_size[1]
            self.all_real_points.append([x,y])
            self.Feature_datas.axs1_points.append([x,y])
        if self.real_label.click_type == "right" and len(self.all_real_points)>0:
            self.all_real_points.pop()
            self.Feature_datas.axs1_points.pop()
            if len(self.all_real_points)>0:
                self.real_markers.set_data(np.array(self.all_real_points), face_color='red', size=8)
            else:
                self.real_markers.set_data(np.empty((0,2)), face_color='red', size=8)
        if self.real_label.click_type == "middle":
            self.Feature_datas.axs1_points.append([None,None])
            self.all_real_points.append([None,None])
            
        # Update the counter label
        self.real_points_counter.setText(f"Real points: {len(self.all_real_points)+1}")
        self._draw_current()

    def _draw_current(self):
        data = self.All_image_datas[self.curr_img_inx]
        # --- Display Blender and Real images ---
        blender_img = data.image_blender  # cv2 image
        real_img = data.image_real     # cv2 image
        if len(self.all_intersection_point) >0:
            # Update Blender image markers
            blender_2d_points = [self.project_3d_to_blender_image(p) for p in self.all_intersection_point]
            blender_img_marked = self.draw_markers_on_cv(blender_img, blender_2d_points)
            pix = self._cv_to_qpixmap(blender_img_marked)
            self.blender_label.setPixmap(pix)

        else:
            pix = self._cv_to_qpixmap(blender_img)
            self.blender_label.setPixmap(pix)

        '''if len(self.all_intersection_point) >0:
            blender_2d_points = [self.project_3d_to_blender_image(p) for p in self.all_intersection_point]
            real_img_marked = self.draw_markers_on_cv(real_img, blender_2d_points)
            pix = self._cv_to_qpixmap(real_img_marked)
            self.real_label.setPixmap(pix)
        else:
            pix = self._cv_to_qpixmap(real_img)
            self.real_label.setPixmap(pix)'''

        if len(self.all_real_points) >0:
            # Update Real image markers
            real_img_marked = self.draw_markers_on_cv(real_img, np.array(self.all_real_points))
            pix = self._cv_to_qpixmap(real_img_marked)
            self.real_label.setPixmap(pix)
        else:
            pix = self._cv_to_qpixmap(real_img)
            self.real_label.setPixmap(pix)

    def _cv_to_qpixmap(self, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        return QtGui.QPixmap.fromImage(qimg)

    def on_canvas_mouse_press(self, event):
        if event.button in (1, 2) or self.mesh_visual is None:
            if event.button == 2:  # right-click to delete last point
                self.delete_last_point()

            if event.button == 1:  # left-click to add point
                mouse_pos = np.array([event.pos[0], event.pos[1]]) # Get the 2D mouse position on the screen
                ray_origin, ray_direction = self.get_ray_from_mouse(mouse_pos) # Project the 2D mouse position into a 3D ray

                data = self.All_image_datas[self.curr_img_inx]
                vertices = np.asarray(data.World_coords, float)
                faces = np.asarray(data.Faces, int)
                faces = split_quads_to_triangles(faces)

                intersection_point = None
                t = float('inf')
                for face in faces:
                    v0, v1, v2 = vertices[face]
                    point = self.ray_triangle_intersect(ray_origin, ray_direction, v0, v1, v2)
                    if point[0] is not None and point[1] < t:
                        intersection_point = point[0]
                        t = point[1]
                if intersection_point is not None:
                    self.all_intersection_point.append(intersection_point)
                    self.Feature_datas.axs0_points.append(intersection_point)
                    if len(self.all_intersection_point)>0:
                        self.sel_markers.set_data(np.array(self.all_intersection_point), face_color='red', size=8)
                        # Add a number for this new point
                        text = scene.visuals.Text(str(len(self.all_intersection_point)), 
                                                color='black', font_size=8, bold=True, parent=self.view.scene)
                        offset = np.array([0, 0.01, 0])  # slightly above the point
                        text.transform = scene.transforms.STTransform(translate=intersection_point + offset)
                        self.sel_texts.append(text)  # store reference
            
            # Update 3D markers
            if self.all_intersection_point:
                self.sel_markers.set_data(np.array(self.all_intersection_point), face_color='red', size=8)
            else:
                self.sel_markers.set_data(np.empty((0,3)), face_color='red', size=8)
            self._draw_current()

    def delete_last_point(self):
        if not self.all_intersection_point:
            return  # nothing to delete
        # Remove last point
        self.all_intersection_point.pop()
        self.Feature_datas.axs0_points.pop()
        # Update markers
        if len(self.all_intersection_point) > 0:
            self.sel_markers.set_data(np.array(self.all_intersection_point), face_color='red', size=8)
        else:
            self.sel_markers.set_data(np.empty((0, 3)), face_color='red', size=8)
        # Remove last text label from scene
        last_text = self.sel_texts.pop()
        last_text.parent = None  # effectively removes it from the scene
        
    def get_ray_from_mouse(self, mouse_pos):
        # Convert 2D mouse position into 3D ray
        camera = self.view.camera        
        width, height = self.canvas.size
        mat_pro = np.array(camera._projection.matrix).T
        K = np.array([[0.5*mat_pro[0,0]*width,0,0.5*(mat_pro[0,2]+1)*width],[0,0.5*mat_pro[1,1]*height,0.5*(mat_pro[1,2]+1)*height],[0,0,1]])
        cam_traf = camera.transform.matrix.T @ np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        ray_origin = np.array([[0, 0, 0, 1]]).T  # Camera position (origin)
        ray_origin_world = cam_traf @ ray_origin
        
        ray_direction_world = cam_traf @ np.append(np.linalg.inv(K)@np.array([[mouse_pos[0]],[mouse_pos[1]],[1]]), [[0]], axis=0)
        ray_direction_world /= np.linalg.norm(ray_direction_world[:3])  # Normalize
        return ray_origin_world[:3], ray_direction_world[:3]

    def ray_triangle_intersect(self, ray_origin, ray_direction, v0, v1, v2):
        epsilon = 1e-6
        edge1 = v1 - v0
        edge2 = v2 - v0

        h = np.cross(ray_direction.flatten(), edge2)
        a = np.dot(edge1, h)

        if -epsilon < a < epsilon:
            return None, None  # parallel, no intersection

        f = 1.0 / a
        s = ray_origin.flatten() - v0
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None, None

        q = np.cross(s, edge1)
        v = f * np.dot(ray_direction.flatten(), q)
        if v < 0.0 or u + v > 1.0:
            return None, None

        t = f * np.dot(edge2, q)
        if t > epsilon:
            intersection_point = ray_origin.flatten() + ray_direction.flatten() * t
            return intersection_point, t
        return None, None
    
    def draw_markers_on_cv(self, img, points, radius=5, color=(0,0,255), flip = True,show_numbers=True):
        """
        Draws small circles on the cv image at given 2D points.
        img: cv2 image (numpy array)
        points: list of (x, y) tuples
        """
        i = 0
        img_copy = img.copy()
        for p in points:
            i += 1
            p = p.flatten()
            if p[0] is not None and p[1] is not None:
                x = int(p[0])
                y = int(p[1])
                cv2.circle(img_copy, (x, y), radius, color, -1)
                # Draw number
                if show_numbers:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 2
                    thickness = 2
                    text = str(i)
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                    text_x = x + radius + 2
                    text_y = y - radius - 2
                    # Ensure text is inside the image bounds
                    text_x = max(0, min(text_x, img.shape[1]-text_size[0]))
                    text_y = max(text_size[1], min(text_y, img.shape[0]))
                    cv2.putText(img_copy, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
            
        return img_copy

    def project_3d_to_blender_image(self, point_3d):        
        K = np.reshape(self.config['cameras']['camera_' + str(self.camera_index)]['CameraIntrinsic'],(3, 3))
        dist_coeffs = np.reshape(self.config['cameras']['camera_' + str(self.camera_index)]['DistCoeffs'],(1, -1))     
        Cam_pose = self.camera_pose_matrix[self.curr_img_inx,:,:]
        Rot_x_180 = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        Cam_pose = Cam_pose @ Rot_x_180
        Inv_cam_pos = np.linalg.inv(Cam_pose)
        rvec = cv2.Rodrigues(Inv_cam_pos[0:3,0:3])[0]
        tvec = Inv_cam_pos[0:3,3]
        point_cv = np.reshape(cv2.projectPoints(point_3d,rvec,tvec,K,dist_coeffs)[0],(2,-1))
        return point_cv
    
    def on_next(self):
        
        points_real = np.array(self.all_real_points)
        points_blender = np.reshape(np.array([self.project_3d_to_blender_image(p) for p in self.all_intersection_point]),(-1,2))
        
        if (len(self.Feature_datas.axs0_points) != 0 and len(self.Feature_datas.axs1_points) != 0):
            img_index = self.All_image_datas[self.curr_img_inx].image_index
            self.Feature_datas.datas.append([img_index,
                                             list(self.Feature_datas.axs0_points),
                                             list(self.Feature_datas.axs1_points)])
        self.Feature_datas.axs1_points = [] 
        self.all_real_points = []
        self.real_markers.set_data(np.empty((0,2)), face_color='red', size=8)
        if self.curr_img_inx < len(self.All_image_datas) - 1:
            self.curr_img_inx += 1
            self.real_points_counter.setText(f"Real points: {len(self.all_real_points)+1}")
            self._draw_current()
        else:
            self.fin()

    def fin(self):
        indexes_needed = []
        for data in self.Feature_datas.datas:
            for inx in range(len(data[2])):
                if data[2][inx][0] is not None and data[2][inx][1] is not None:
                    indexes_needed.append(inx)

        indexes_needed = list(set([x for x in indexes_needed if indexes_needed.count(x) > 1]))
        if len(indexes_needed) < 3:
            QtWidgets.QMessageBox.critical(self, "Error", "At least three common points are required for pose optimization.")
            return

        D3_points_needed = []
        for indx in range(len(self.Feature_datas.datas[0][1])):
            if indx in indexes_needed:
                D3_points_needed.append(self.Feature_datas.datas[0][1][indx])
        D3_points_needed = np.array(D3_points_needed)

        rays = [[] for _ in range(len(indexes_needed))]
        camera_intrinsic_matrix = np.reshape(self.config['cameras']['camera_' + str(self.camera_index)]['CameraIntrinsic'], (3, 3))
        dist_coeffs = np.reshape(self.config['cameras']['camera_' + str(self.camera_index)]['DistCoeffs'], (1, -1))

        real_points = []
        img_idxs = []
        for i in range(len(self.Feature_datas.datas)):
            img_idx, _, real_pts = self.Feature_datas.datas[i]
            real_points.append(np.array(real_pts)[indexes_needed,:])
            img_idxs.append(img_idx)
        real_points = np.array(real_points,dtype = 'f')

        rays_in_cam_coords_all = []
        for i in range(np.array(real_points).shape[1]):
            coordinates = np.copy(real_points[:,i,:])
            #d2crds = np.reshape(np.copy(coordinates[~np.isnan(coordinates)]),(-1,2))
            undist = np.reshape(cv2.undistortPoints(coordinates, camera_intrinsic_matrix, dist_coeffs),(-1,2))
            ray = np.append(undist, np.ones((undist.shape[0], 1)), axis=1).T
            rays_in_cam_coords_all.append(ray)
            for pics_numb in range(len(img_idxs)):
                Rotate_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])               
                cam_mat = self.camera_pose_matrix[pics_numb, 0:3, 0:3] @ Rotate_x_180
                ray_in_3d = cam_mat @ ray[:,pics_numb]
                ray_in_3d = np.append(np.reshape(self.camera_pose_matrix[pics_numb, 0:3, 3],(3,1)),ray_in_3d)
                if not np.all(np.isnan(ray_in_3d[3:])):
                    rays[i].append(ray_in_3d)

        each_mid_point = []
        for q in range(len(rays)):
            mid_points = []
            for i in range(len(rays[q])):
                for j in range(i + 1, len(rays[q])):
                    r1 = rays[q][i][:3]; r2 = rays[q][j][:3]
                    v1 = rays[q][i][3:]; v2 = rays[q][j][3:]
                    value1 = 1 / np.linalg.norm(np.cross(v1, v2)) ** 2
                    value2 = (np.linalg.det(np.dstack([r2, v2, np.cross(v1, v2)])) * v1 +
                            np.linalg.det(np.dstack([r1, v1, np.cross(v2, v1)])) * v2)
                    p1 = value1 * (value2 + np.dot(r1, np.cross(v1, v2)) * np.cross(v1, v2))
                    p2 = value1 * (value2 + np.dot(r2, np.cross(v1, v2)) * np.cross(v1, v2))
                    mid_points.append((p1 + p2) / 2)
            each_mid_point.append(mid_points)

        mid_points = np.zeros((len(each_mid_point), 3))
        for i in range(len(each_mid_point)):
            mid_points[i, :] = np.mean(np.array(each_mid_point[i]), axis=0)
        print("MID-POINTS")
        print(mid_points)

        Tr = SVD_pose_diff_estimation(mid_points, D3_points_needed)

        ORotation = R.from_euler('xyz', self.config['object_pose_' + self.object_name]['rotation'], degrees=False).as_matrix()
        object_hom_traf = np.zeros((4, 4))
        object_hom_traf[0:3, 0:3] = ORotation
        object_hom_traf[0:3, 3] = self.config['object_pose_' + self.object_name]['location']
        object_hom_traf[3, 3] = 1

        New_obj_hom_traf = np.linalg.inv(Tr) @ object_hom_traf
        object_pos = (New_obj_hom_traf[0:3, 3]).tolist()
        object_ori = R.from_matrix(New_obj_hom_traf[0:3, 0:3]).as_euler('xyz', degrees=False).tolist()
        
        print("OBJECT POS")
        print(object_pos)
        print("OBJECT ORI")
        print(object_ori)
        
        self.config['object_pose_' + self.object_name].update({'location': object_pos, 'rotation': object_ori})
        with open(self.path + '/label_factory/config.yaml', 'w') as yamlfile:
            yaml.safe_dump(self.config, yamlfile, default_flow_style=False)

        rays_in_cam_coords_all = np.array(rays_in_cam_coords_all)
        for i in range(rays_in_cam_coords_all.shape[0]):
            rays_in_cam_coords_all[i]








        print(closed_text("The config.yaml file is updated with the optimized object pose", self.Width, "green", "center"))
        QtWidgets.QMessageBox.information(self, "Done", "config.yaml updated with optimized object pose.")
        self.close()
        

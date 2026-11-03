import blenderproc as bproc

import os
import argparse
import cv2
import numpy as np
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--cad_path', help="The path of CAD model")
parser.add_argument('--output_dir', help="The path to save CAD templates")
parser.add_argument(
    '--templates_dir',
    help="Exact template directory (overrides <output_dir>/templates)",
)
parser.add_argument('--normalize', default=True, help="Whether to normalize CAD model or not")
parser.add_argument('--colorize', default=False, help="Whether to colorize CAD model or not")
parser.add_argument('--base_color', default=0.05, help="The base color used in CAD model")
args = parser.parse_args()

# set the cnos camera path
render_dir = os.path.dirname(os.path.abspath(__file__))
cnos_cam_fpath = os.path.join(render_dir, '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy')

bproc.init()

def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')

    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)

    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)

    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))

    return 1/(2*radius)


# load cnos camera pose
cam_poses = np.load(cnos_cam_fpath)

# calculating the scale of CAD model
if args.normalize:
    scale = get_norm_info(args.cad_path)
else:
    scale = 1

# Load the object only once. Calling bproc.clean_up() and reimporting it for
# every view leaves stale BlenderProc material wrappers behind for some STL
# files, which makes render_nocs() fail with "StructRNA ... has been removed".
# Blender's OBJ importer converts files from Y-up/-Z-forward to its own frame by
# default, which would rotate a textured OBJ relative to the STL. The pipeline
# defines the textured model in the same coordinates as the STL, so import it as is.
load_kwargs = {}
if args.cad_path.lower().endswith(".obj"):
    load_kwargs = {"forward_axis": "Y", "up_axis": "Z"}
obj = bproc.loader.load_obj(args.cad_path, **load_kwargs)[0]
obj.set_scale([scale, scale, scale])
obj.set_cp("category_id", 1)

# assigning material colors to untextured objects
if args.colorize:
    color = [args.base_color, args.base_color, args.base_color, 0.]
    material = bproc.material.create('obj')
    material.set_principled_shader_value('Base Color', color)
    obj.set_material(0, material)

# Use one reusable light and render one camera view at a time. In particular,
# do not register all views as a Blender animation: OPTIX can stall at frame
# 1/N for that path on some GPU/driver combinations.
light_scale = 2.5
light_energy = 1000
light = bproc.types.Light()
light.set_type("POINT")
light.set_energy(light_energy)

save_fpath = (
    args.templates_dir
    if args.templates_dir
    else os.path.join(args.output_dir, "templates")
)
os.makedirs(save_fpath, exist_ok=True)
bproc.renderer.set_max_amount_of_samples(50)

blender_cam_poses = []
for idx, cam_pose_source in enumerate(cam_poses):
    # Never mutate the shared predefined-pose array in place.
    cam_pose = cam_pose_source.copy()
    cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
    cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
    blender_cam_poses.append(cam_pose)

    # Reuse frame zero so render() processes exactly one view and camera frames
    # do not accumulate over the loop.
    bproc.camera.add_camera_pose(cam_pose, frame=0)
    light.set_location(light_scale * cam_pose[:3, -1], frame=0)

    data = bproc.renderer.render()
    color_bgr_0 = data["colors"][0]

    # save rgb image
    color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
    cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(idx)+'.png'), color_bgr_0)

# render_nocs() creates a temporary material which this BlenderProc version
# leaves in its Python instance registry after Blender's undo removes it.
# Calling render_nocs() again would then access that deleted material and raise
# "ReferenceError: StructRNA of type Material has been removed". Register all
# views and render the complete NOCS pass exactly once.
for frame, cam_pose in enumerate(blender_cam_poses):
    bproc.camera.add_camera_pose(cam_pose, frame=frame)

nocs_frames = bproc.renderer.render_nocs()["nocs"]
if len(nocs_frames) != len(blender_cam_poses):
    raise RuntimeError(
        f"Expected {len(blender_cam_poses)} NOCS frames, "
        f"but BlenderProc returned {len(nocs_frames)}"
    )

for idx, nocs_0 in enumerate(nocs_frames):
    # save mask
    mask_0 = nocs_0[..., -1]
    cv2.imwrite(os.path.join(save_fpath,'mask_'+str(idx)+'.png'), mask_0*255)
    
    # save nocs
    xyz_0 = 2*(nocs_0[..., :3] - 0.5)
    np.save(os.path.join(save_fpath,'xyz_'+str(idx)+'.npy'), xyz_0.astype(np.float16))

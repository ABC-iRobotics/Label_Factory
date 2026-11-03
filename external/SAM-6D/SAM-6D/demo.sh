export ROOT="$PWD"
export CAD_PATH=$ROOT/Data/MyObject_2/object.ply   # path to a given cad model(mm)
export RGB_PATH=$ROOT/Data/MyObject_2/rgb.png           # path to a given RGB image
export DEPTH_PATH=$ROOT/Data/MyObject_2/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=$ROOT/Data/MyObject_2/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=$ROOT/Data/MyObject_2/outputs         # path to a pre-defined file for saving results

# Render CAD templates
cd Render
blenderproc run render_custom_templates.py --custom-blender-path "${SAM6D_BLENDER_PATH:?Set SAM6D_BLENDER_PATH}" --output_dir "$OUTPUT_DIR" --cad_path "$CAD_PATH" #--colorize True


# Run instance segmentation model
export SEGMENTOR_MODEL=sam

cd ../Instance_Segmentation_Model
python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --stability_score_thresh 0.85


# Run pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

cd ../Pose_Estimation_Model
python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH

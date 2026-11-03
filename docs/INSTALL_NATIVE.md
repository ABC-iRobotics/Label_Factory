# Native installation

This guide installs Label Factory directly on the host, without Docker. It
reproduces what the [`Dockerfile`](../docker/Dockerfile) does. If you only want to
try the framework, use the [Docker-based installation](INSTALL_DOCKER.md).

Label Factory needs two separate Python environments, because the
reconstruction side and SAM-6D require incompatible package versions:

| Environment | Python | Purpose |
|---|---|---|
| `label-factory` | 3.10 | user interface, COLMAP/PyCOLMAP reconstruction, RealSense, consolidation, publishing |
| `sam6d` | 3.9 | SAM-6D (PyTorch 2.0), BlenderProc template and annotation rendering |

The steps were written for Ubuntu 22.04 on a machine with an NVIDIA GPU (8 GB
VRAM or more). Any camera supported by `pyrealsense2` can be used for recording.

## 1. System requirements

An NVIDIA driver, and the CUDA 11.8 toolkit with
`nvcc` (PyTorch 2.0 is built for CUDA 11.x, and the PointNet2 extension is
compiled against the local toolkit). The compiler must be supported by CUDA 11.8
(GCC 11 or older).

```bash
nvidia-smi                      # driver and GPU
/usr/local/cuda-11.8/bin/nvcc --version
gcc --version
```

Install the system packages:

```bash
sudo apt-get update
sudo apt-get install -y build-essential ninja-build git wget curl ffmpeg \
  libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 libx11-6 \
  libxi6 libxkbcommon-x11-0 libxcb-xinerama0 libegl1 libusb-1.0-0
```

## 2. Conda

Install [Miniforge](https://github.com/conda-forge/miniforge) (or
Miniconda) and open a new shell.

## 3. Reconstruction environment (`label-factory`)



```bash
conda env create -f environment.yml
conda activate label-factory
python -c "import pycolmap; print(pycolmap.__version__, pycolmap.has_cuda)"
python -c "import pyrealsense2 as rs; print('pyrealsense2 ready')"
```

`pycolmap.has_cuda` must be `True`, because the dense PatchMatch stage has no CPU
fallback. The pinned `pycolmap-cuda12` wheel needs a driver that supports
CUDA 12. If it reports `False`, install a CUDA-enabled
[COLMAP](https://colmap.github.io/install.html) and PyCOLMAP build instead.

## 4. SAM-6D environment (`sam6d`)



```bash
conda create -n sam6d python=3.9.18 pip=24.0 -y
conda activate sam6d
pip install -r docker/requirements-sam6d.txt
python -c "import numpy, torch; print(numpy.__version__, torch.__version__, torch.cuda.is_available())"
```

The file pins NumPy 1.26.4 and PyTorch 2.0.0; the last command must print
`True`. Then compile the PointNet2 CUDA extension for your GPU. The architecture
is the compute capability of the GPU (8.9 for RTX 40 series and RTX 2000 Ada,
8.6 for RTX 30 series, 8.0 for A100):

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9+PTX"
cd external/SAM-6D/SAM-6D/Pose_Estimation_Model/model/pointnet2
python setup.py install
cd -
```

## 5. Blender and BlenderProc

The template and annotation rendering uses
Blender 3.3.1. BlenderProc runs inside Blender's own Python, so its runtime
packages are installed there:

```bash
wget https://download.blender.org/release/Blender3.3/blender-3.3.1-linux-x64.tar.xz
mkdir -p ~/blender-3.3.1
tar -xJf blender-3.3.1-linux-x64.tar.xz --strip-components=1 -C ~/blender-3.3.1

conda activate sam6d
blenderproc pip --custom-blender-path ~/blender-3.3.1 install \
  wheel pyyaml==5.1.2 imageio==2.9.0 gitpython==3.1.18 \
  scikit-image==0.19.2 pypng==0.0.20 scipy==1.7.3 matplotlib==3.5.1 \
  pytz==2021.1 h5py==3.6.0 Pillow==8.3.2 opencv-contrib-python==4.5.5.64 \
  scikit-learn==1.0.2 python-dateutil==2.8.2 rich==12.6.0 trimesh==3.21.5 \
  pyrender==0.1.45
```

Pre-installing these packages avoids a long download the first time templates are
rendered. Even so, the first rendering starts Blender headless and takes
noticeably longer than the following ones.

## 6. Model weights

Download the weights from the upstream SAM-6D project
(its `download_*.py` scripts in `Instance_Segmentation_Model/` and
`Pose_Estimation_Model/` fetch them) and place them in the checkpoint folders of
the SAM-6D source tree:

```text
external/SAM-6D/SAM-6D/
├── Instance_Segmentation_Model/checkpoints/
│   ├── segment-anything/sam_vit_h_4b8939.pth
│   ├── dinov2/dinov2_vitl14_pretrain.pth
│   └── FastSAM/FastSAM-x.pt                 # optional
└── Pose_Estimation_Model/checkpoints/
    ├── sam-6d-pem-base.pth
    └── mae_pretrain_vit_base.pth
```

If the weights are already stored in a `checkpoints/ism` and `checkpoints/pem`
layout (as used by the Docker setup), link them instead of copying:

```bash
rmdir external/SAM-6D/SAM-6D/Instance_Segmentation_Model/checkpoints 2>/dev/null
ln -s "$PWD/checkpoints/ism" external/SAM-6D/SAM-6D/Instance_Segmentation_Model/checkpoints
rmdir external/SAM-6D/SAM-6D/Pose_Estimation_Model/checkpoints 2>/dev/null
ln -s "$PWD/checkpoints/pem" external/SAM-6D/SAM-6D/Pose_Estimation_Model/checkpoints
```

## 7. Environment variables

The pipeline finds the SAM-6D interpreter and
Blender through two variables (see [`.env.example`](../.env.example)). Export them
in the shell that starts the application, or add them to `~/.bashrc`:

```bash
export SAM6D_PYTHON="$(conda run -n sam6d which python)"
export SAM6D_BLENDER_PATH="$HOME/blender-3.3.1"
```

Optionally load a hardware profile with the reconstruction settings, and a
Hugging Face write token if you publish datasets:

```bash
set -a; source docker/pipeline.rtx2000.env; set +a     # or docker/pipeline.a100.env
export HF_TOKEN=hf_...                           # only when publishing
```

## 8. RealSense (only for recording)

`pyrealsense2` is installed with the
`label-factory` environment. For USB access without `sudo`, install the
RealSense udev rules from the
[librealsense installation guide](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
and reconnect the camera. Use a USB 3 port and cable.

## 9. Check and start



```bash
conda activate label-factory
streamlit run ui/app.py                          # http://localhost:8501
```

Before a long run, validate the inputs of a dataset (RGB-D frames, camera,
reconstruction and STL models); the user interface offers the same check:

```bash
python scripts/SAM6D_batch_inference.py --config scripts/configs/default.yaml \
  --python "$SAM6D_PYTHON" --check-inputs
```

## Frequent problems

| Symptom | Cause and remedy |
|---|---|
| `nvcc` not found, or `RuntimeError: ... CUDA version mismatch` while compiling PointNet2 | `CUDA_HOME` and `PATH` do not point to the CUDA 11.8 toolkit that matches PyTorch's CUDA version. |
| Compilation fails with an unsupported GCC version | Install GCC 11 and set `CC=gcc-11 CXX=g++-11`, or use the Docker image. |
| `pycolmap.has_cuda` is `False` | The PyCOLMAP build is CPU-only or the driver does not support CUDA 12; use a CUDA-enabled COLMAP build. |
| `torch.cuda.is_available()` is `False` in `sam6d` | The driver and the PyTorch CUDA build do not match; check `nvidia-smi` and reinstall the pinned PyTorch. |
| Open3D or Qt windows fail with a display error | Run on a desktop session or set `DISPLAY`; headless servers can use the command-line scripts without the viewers. |
| `Couldn't resolve requests` when recording | The RGB and depth resolution/FPS combination is not supported by the camera. |

Runtime problems of the pipeline itself are listed in the [usage guide](USAGE.md#troubleshooting).


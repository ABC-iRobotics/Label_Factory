# Docker-based installation

The Docker image contains both Python environments (`label-factory` and
`sam6d`), the CUDA build tools, Blender 3.3.1, BlenderProc and the COLMAP
bindings. It is the recommended way to install Label Factory. For an
installation without Docker see [`INSTALL_NATIVE.md`](INSTALL_NATIVE.md).

**Contents**

- [Requirements](#requirements)
- [1. Host preparation](#1-host-preparation)
- [2. Get the code and the model weights](#2-get-the-code-and-the-model-weights)
- [3. Build the image](#3-build-the-image)
- [4. Start the application](#4-start-the-application)
- [5. Verify the CUDA environments](#5-verify-the-cuda-environments)
- [Hardware profiles](#hardware-profiles)
- [Camera and display access](#camera-and-display-access)
- [Persistent data](#persistent-data)
- [Hugging Face token](#hugging-face-token)
- [Update and stop](#update-and-stop)
- [Why is the image large?](#why-is-the-image-large)
- [Troubleshooting](#troubleshooting)

## Requirements

- Linux host with an NVIDIA GPU (8 GB VRAM or more) and a current NVIDIA driver
- Docker Engine with the Compose plugin
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- about 25 GB of disk space for the image, about 5 GB for the model weights, and
  the space of the recorded scenes (a 250-frame scene with all intermediate
  results takes several GB)
- a camera supported by `pyrealsense2`, connected to a USB 3 port, only if you
  want to record your own scenes

## 1. Host preparation

```bash
docker --version && docker compose version && nvidia-smi

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# the GPU must be visible from a container
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 2. Get the code and the model weights

```bash
git clone https://github.com/ABC-iRobotics/Label_Factory.git
cd Label_Factory
mkdir -p data calibration/cameras checkpoints/ism checkpoints/pem
```

The SAM-6D weights are not part of the repository or the image. Download them
from the upstream [SAM-6D](https://github.com/JiehongLin/SAM-6D) project, under
its license, and place them on the host as follows:

```text
checkpoints/
├── ism/
│   ├── segment-anything/sam_vit_h_4b8939.pth
│   ├── dinov2/dinov2_vitl14_pretrain.pth
│   └── FastSAM/FastSAM-x.pt                 # optional
└── pem/
    ├── sam-6d-pem-base.pth
    └── mae_pretrain_vit_base.pth
```

## 3. Build the image

The PointNet2 CUDA extension of SAM-6D is compiled for one GPU generation.
Docker builds cannot see the GPU, so the architecture is given explicitly. The
default is `8.9` (Ada: RTX 40 series, RTX 2000 Ada).

```bash
docker compose build
# other GPUs:
TORCH_CUDA_ARCH_LIST="8.6" docker compose build                    # Ampere, RTX 30
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX" docker compose build    # portable
```

Rebuild with `--no-cache` after changing the GPU architecture, a dependency or
the Dockerfile.

## 4. Start the application

```bash
xhost +local:docker          # lets the container open native windows (see below)
docker compose up
```

Open <http://localhost:8501> (or `http://HOST_IP:8501` from another machine).
Run it in the background with `docker compose up -d` and follow the log with
`docker compose logs -f --tail=150 label-factory`. The next step is the
[user interface guide](UI_GUIDE.md).

## 5. Verify the CUDA environments

Both commands must print `True`:

```bash
docker compose run --rm label-factory \
  python -c "import pycolmap; print(pycolmap.__version__, pycolmap.has_cuda)"
docker compose run --rm label-factory /opt/conda/envs/sam6d/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Hardware profiles

Two environment files tune the reconstruction settings (number of PatchMatch
workers, cache sizes and so on):

| Profile | For |
|---|---|
| [`docker/pipeline.rtx2000.env`](../docker/pipeline.rtx2000.env) | 8 GB laptop GPU (RTX 2000 Ada), the configuration of the paper |
| [`docker/pipeline.a100.env`](../docker/pipeline.a100.env) | server GPU, more parallelism and host cache |

```bash
docker compose --env-file docker/pipeline.rtx2000.env up label-factory
```

Use the same profile for building and running. On an A100 40 GB, the PatchMatch
workers can be reduced temporarily with `MVS_GPU_INDICES=0,0`.

## Camera and display access

Calibration, recording and the 3D viewers open **native windows** on the host
display. On an X11 host authorize the container before starting it and revoke
the authorization afterwards:

```bash
xhost +local:docker
xhost -local:docker
```

Wayland-only, remote and headless hosts need an X server, or must use the
command-line workflow without previews ([`USAGE.md`](USAGE.md)). The browser
interface itself works from any machine.

`compose.yaml` passes `/dev/bus/usb` into the container. Close other programs
that use the camera (for example the RealSense Viewer). If access is denied,
install the RealSense udev rules on the host
([librealsense guide](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md));
do not run the application as root only to bypass the device permissions.

On some hosts the RealSense USB ioctl calls stay blocked even with the bind
mount. For local calibration and recording, start the service with the override
that gives the container privileged access:

```bash
docker compose --env-file docker/pipeline.rtx2000.env \
  -f compose.yaml -f docker/compose.capture.yaml up label-factory
```

Use this only with a locally built, trusted image. Processing and server
deployments do not need it.

If the camera list is empty: check the USB 3 cable and port, close other camera
programs, restart the container after connecting the camera, and check the host
USB permissions.

## Persistent data

These host folders are bind-mounted, so removing or rebuilding the container
does not delete them:

| Host | Container |
|---|---|
| `./data` | `/workspace/data` |
| `./calibration` | `/workspace/calibration` |
| `./checkpoints/ism`, `./checkpoints/pem` | the SAM-6D checkpoint folders (read-only) |

The **Replace …** switches of the interface can overwrite earlier results of a
dataset, so keep a copy of important `data/<dataset>` folders.

## Hugging Face token

Pass the token at run time, never in an image, a configuration file or Git:

```bash
HF_TOKEN="hf_..." docker compose up
```

Alternatively paste it into the **Review & publish** tab, where it is passed to
the upload process only.

## Update and stop

```bash
git pull
docker compose build
docker compose up          # add --env-file docker/pipeline.<profile>.env if used

docker compose down        # stop; the bind-mounted data stays
xhost -local:docker
```

## Why is the image large?

The image is about 23.5 GB. The approximate shares, from the layer sizes of
`docker history`:

| Part | Size | Why it is needed |
|---|---|---|
| CUDA 11.8 development toolkit and cuDNN | about 9.5 GB | the toolkit with `nvcc` is needed to compile the PointNet2 extension of SAM-6D, and PyTorch 2.0 and the CUDA COLMAP bindings use the libraries |
| the two Conda environments | about 10 GB | `label-factory` (PyCOLMAP with CUDA, Open3D, RealSense, Streamlit) and `sam6d` (PyTorch 2.0 with its bundled CUDA libraries, BlenderProc dependencies). They must be separate because their package versions are incompatible |
| Blender 3.3.1 and the BlenderProc packages | about 1.5 GB | template and annotation rendering |
| the repository with the SAM-6D source | about 0.7 GB | |

The model weights, the datasets and the calibrations are **not** in the image.
A smaller image would need a multi-stage Dockerfile that compiles the extension
in a development image and runs on a CUDA runtime image; this is possible but
not done in the current version.

## Troubleshooting

| Symptom | What to do |
|---|---|
| The container does not see the GPU | `docker compose run --rm label-factory nvidia-smi`; if it fails, the driver or the NVIDIA Container Toolkit is not configured |
| `pycolmap.has_cuda` is `False` | Dense PatchMatch has no CPU fallback. Rebuild the image and check the driver supports CUDA 12 |
| `torch.cuda.is_available()` is `False` in `sam6d` | check `nvidia-smi` inside the container and the driver version |
| CUDA out of memory | use the `docker/pipeline.rtx2000.env` profile on 8 GB GPUs, close other GPU programs, lower the *Maximum MVS image size*, and do not run two SAM-6D stages from two sessions at once |
| The interface is not reachable | `docker compose ps` and `docker compose logs --tail=150 label-factory`; check that port 8501 is free |
| A native window does not open | `echo "$DISPLAY"`, run `xhost +local:docker`; on remote hosts the X11 forwarding must be set up |
| Interactive shell in the container | `docker compose run --rm --entrypoint bash label-factory` |

Runtime problems of the pipeline itself (reconstruction, SAM-6D, BlenderProc)
are listed in the [usage guide](USAGE.md#troubleshooting).

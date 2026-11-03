# Docker deployment

## Prerequisites

- Linux host with an NVIDIA GPU and current NVIDIA driver
- Docker Engine with Docker Compose
- NVIDIA Container Toolkit
- Intel RealSense connected through USB 3
- SAM-6D checkpoints placed in the host `checkpoints/` layout below

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

The weights are intentionally not included in the image. Obtain them under the
terms and instructions of the upstream SAM-6D project.

## Build and start

```bash
mkdir -p data calibration/cameras checkpoints/ism checkpoints/pem
docker compose build
docker compose up
```

The image defaults to CUDA compute capability `8.9`, which matches NVIDIA Ada
GPUs such as the RTX 2000 Ada Laptop GPU. Docker builds do not expose the GPU,
so this value must be supplied explicitly while compiling SAM-6D's PointNet2
extension. For another GPU architecture, override it when building, for example:

```bash
TORCH_CUDA_ARCH_LIST="8.6" docker compose build
```

Open `http://localhost:8501`.

Verify CUDA before processing a dataset:

```bash
docker compose run --rm label-factory \
  python -c "import pycolmap; print(pycolmap.__version__, pycolmap.has_cuda)"

docker compose run --rm label-factory \
  /opt/conda/envs/sam6d/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Both CUDA checks must report `True` for the complete workflow.

## Native visualization windows

Calibration, recording, and Open3D visualization currently use native windows.
On an X11 host, authorize the local container before starting it:

```bash
xhost +local:docker
docker compose up
```

Revoke the authorization afterward:

```bash
xhost -local:docker
```

Wayland-only, remote, and headless hosts require an X server or must run the
non-interactive pipeline commands without native previews.

## RealSense access

Compose passes `/dev/bus/usb` into the container. Stop other applications using
the camera. If access is denied, install the RealSense udev rules on the host;
do not run the complete application as root solely to bypass device permissions.

## Hugging Face publication

Pass the token at runtime, never in an image or configuration file:

```bash
export HF_TOKEN=your_token
docker compose up
```

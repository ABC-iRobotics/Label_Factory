FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MINIFORGE_VERSION=24.7.1-2
ARG BLENDER_VERSION=3.3.1
ARG BLENDER_ARCHIVE=blender-3.3.1-linux-x64
# Docker builds normally cannot see the host GPU. Set the architectures used
# when compiling SAM-6D's custom PointNet2 CUDA extension explicitly.
# 8.9 is used by NVIDIA Ada GPUs, including the RTX 2000 Ada Laptop GPU.
ARG TORCH_CUDA_ARCH_LIST="8.9+PTX"

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    SAM6D_PYTHON=/opt/conda/envs/sam6d/bin/python \
    SAM6D_BLENDER_PATH=/opt/blender \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    DISPLAY=:0 \
    QT_X11_NO_MITSHM=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      bash ca-certificates curl git wget bzip2 build-essential ninja-build \
      libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 libx11-6 \
      libxi6 libxkbcommon-x11-0 libxcb-xinerama0 libegl1 libusb-1.0-0 \
      ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
      -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" && \
    rm /tmp/miniforge.sh

ENV PATH=${CONDA_DIR}/bin:${PATH}
WORKDIR /workspace

COPY environment.yml /workspace/environment.yml
COPY ui/requirements.txt /workspace/ui/requirements.txt
RUN conda env create -f /workspace/environment.yml && conda clean -afy

# Fail during the image build instead of later during capture if the
# architecture/index did not provide the RealSense Python wheel.
RUN conda run -n label-factory python -c \
      "import pyrealsense2 as rs; print('pyrealsense2 ready')"

COPY docker/requirements-sam6d.txt /tmp/requirements-sam6d.txt
RUN conda create -n sam6d python=3.9.18 pip=24.0 -y && \
    conda run -n sam6d pip install --no-cache-dir -r /tmp/requirements-sam6d.txt && \
    conda run -n sam6d python -c \
      "import numpy as np, torch; assert np.__version__ == '1.26.4'; x = torch.from_numpy(np.zeros(1, dtype=np.float32)); print('SAM-6D NumPy ABI:', np.__version__, 'tensor:', x.tolist())" && \
    conda clean -afy

RUN wget -q \
      "https://download.blender.org/release/Blender3.3/${BLENDER_ARCHIVE}.tar.xz" \
      -O /tmp/blender.tar.xz && \
    mkdir -p /opt/blender && \
    tar -xJf /tmp/blender.tar.xz --strip-components=1 -C /opt/blender && \
    rm /tmp/blender.tar.xz

# BlenderProc uses Blender's embedded Python, not the sam6d Conda interpreter.
# Preinstall its default runtime packages into the image so template rendering
# never downloads/builds pyrender, PyOpenGL, scipy, OpenCV, etc. at job time.
RUN conda run -n sam6d blenderproc pip \
      --custom-blender-path /opt/blender install \
      wheel pyyaml==5.1.2 imageio==2.9.0 gitpython==3.1.18 \
      scikit-image==0.19.2 pypng==0.0.20 scipy==1.7.3 \
      matplotlib==3.5.1 pytz==2021.1 h5py==3.6.0 Pillow==8.3.2 \
      opencv-contrib-python==4.5.5.64 scikit-learn==1.0.2 \
      python-dateutil==2.8.2 rich==12.6.0 trimesh==3.21.5 \
      pyrender==0.1.45

COPY . /workspace

# Build SAM-6D's PointNet2 CUDA extension against the image CUDA toolchain.
RUN cd /workspace/external/SAM-6D/SAM-6D/Pose_Estimation_Model/model/pointnet2 && \
    conda run -n sam6d python setup.py install

RUN chmod +x /workspace/docker/entrypoint.sh && \
    mkdir -p /workspace/data \
      /workspace/calibration/cameras \
      /workspace/external/SAM-6D/SAM-6D/Instance_Segmentation_Model/checkpoints \
      /workspace/external/SAM-6D/SAM-6D/Pose_Estimation_Model/checkpoints

EXPOSE 8501
ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

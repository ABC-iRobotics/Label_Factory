# Documentation

Start with the [main README](../README.md) for an overview of Label Factory. This
folder contains the detailed guides.

| Document | Content |
|---|---|
| [`INSTALL_DOCKER.md`](INSTALL_DOCKER.md) | Docker-based installation (recommended): build, start, hardware profiles, camera and display access, persistent data, troubleshooting |
| [`INSTALL_NATIVE.md`](INSTALL_NATIVE.md) | installation without Docker: the two Conda environments, CUDA, Blender, the PointNet2 extension |
| [`UI_GUIDE.md`](UI_GUIDE.md) | the graphical user interface step by step, with screenshots, from an empty dataset to the Hugging Face upload |
| [`USAGE.md`](USAGE.md) | the same workflow from the command line, input and output layout, YCB-Video evaluation, runtime measurement, troubleshooting |
| [`DATASETS.md`](DATASETS.md) | format of the published datasets (configurations, splits) and loading examples |

Other material:

| Path | Content |
|---|---|
| [`index.html`](index.html) | source of the [project page](https://abc-irobotics.github.io/Label_Factory/) |
| [`media/`](media) | videos and images of the project page (animation steps, 5 + 5 scene videos, annotated frames) |
| [`images/`](images) | figures and screenshots used by the documentation (light and dark variants) |
| [`../evaluation/`](../evaluation) | result tables of the paper (YCB-Video, runtime) |

**Where to start**

- *I want to install it:* [`INSTALL_DOCKER.md`](INSTALL_DOCKER.md), or
  [`INSTALL_NATIVE.md`](INSTALL_NATIVE.md) without Docker.
- *I want to annotate my own scene:* [`UI_GUIDE.md`](UI_GUIDE.md).
- *I want to script or automate it:* [`USAGE.md`](USAGE.md).
- *I want to train on the published data:* [`DATASETS.md`](DATASETS.md).

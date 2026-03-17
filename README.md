# Label Factory
#### Precise Digital Twin Alignment for Real-World Dataset Creation

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Blender](https://img.shields.io/badge/blender-4.3.2-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.0-purple) 

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [License](#license)
6. [Acknowledgements](#acknowledgements)

---

## Overview

This repository features a robot-assisted framework designed to generate annotated datasets. By aligning 3D object models with real images, it integrates synthetic and real-world data generation, enabling **semi-automatic annotation** with minimal human involvement. The framework enhances the **Blender Annotation Tool (BAT)** and **Robot Operating System 2 (ROS2)** to facilitate real-time image acquisition and annotation.

- [BAT](https://github.com/karolyartur/blender_annotation_tool)
- [ROS2](https://docs.ros.org/en/foxy/index.html)

### Key Contributions:
- **Scalable multi-modal annotation generation**: Automatically creates diverse annotations (instance segmentation, depth maps, surface normals) while maintaining physical realism and minimizing the synthetic-to-real domain gap.
- **Deterministic real-to-virtual alignment**: Employs geometric registration to ensure precise correspondence between real and virtual scenes, enabling reliable annotation projection.
- **Robot-assisted measurement infrastructure**: Combines robotic arm precision with calibrated cameras for accurate pose data and synchronized multi-view image capture.

## Features
- Real-to-virtual annotation generation using robot-assisted image acquisition.
- Pose estimation and annotation projection on real-world images.
- Scalable for large-scale dataset generation with varied annotation types.
- Supports object detection, 6D pose estimation, and other robotic perception tasks.

## Installation

Follow these steps to set up the environment and install dependencies:

### Blender Environment
- **Blender**: Requires Blender version 4.3.2 (or later). Download from [Blender's official website](https://download.blender.org/release/).
- **BAT**: Install the Blender Annotation Tool add-on from [BAT GitHub](https://github.com/karolyartur/blender_annotation_tool).

### ROS2 Environment
- **ROS2 Jazzy**: This framework requires ROS2 Jazzy. Follow the installation instructions at [ROS2 Jazzy Installation](https://docs.ros.org/en/jazzy/Installation.html).
- **MoveIt2**: Install MoveIt2 by following the [binary installation guide](https://moveit.ai/install-moveit2/binary).
- **UR Driver**: Install the [ROS2 Jazzy UR Driver](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver) for robot manipulation (tested with UR16e) and accurate camera extrinsic parameter identification, essential for the annotation pipeline.


### Clone the Repository
```bash
ros2 pkg create --build-type ament_python label_factory
```

```bash
cd label_factory
git clone https://github.com/ArminKaroly/BAT_Virtual_Twin.git
colcon build
```

## Usage


## License

This software is released under the MIT License, see [LICENSE](./LICENSE).   

## Acknowledgement
This work is related to the MedLaBotX project (2024-1.2.3-HU-RIZONT-00069).
Project 2024-1.2.3-HU-RIZONT-00069 has been implemented with support provided by the Ministry of Culture and Innovation of Hungary from the National Research, Development, and Innovation Fund, financed under the 2024-1.2.3-HU-RIZONT funding scheme.


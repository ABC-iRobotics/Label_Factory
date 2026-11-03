# Datasets

The datasets recorded and annotated with Label Factory are published on
Hugging Face:

| Dataset | Content |
|---|---|
| [`ABC-iRobotics/Hand_Tools`](https://huggingface.co/datasets/ABC-iRobotics/Hand_Tools) | hand tools, ten scenes |
| [`ABC-iRobotics/LabWare_Items`](https://huggingface.co/datasets/ABC-iRobotics/LabWare_Items) | laboratory items, ten scenes |

Every dataset card contains runnable examples as well. This page describes the
format and how to use it, and it is the format that the
**Review & publish** tab ([`UI_GUIDE.md`](UI_GUIDE.md#7-review-and-publish))
and `scripts/publish_huggingface_dataset.py` produce for your own datasets.

**Contents**

- [Configurations](#configurations)
- [Repository layout](#repository-layout)
- [Splits](#splits)
- [Loading examples](#loading-examples)

## Configurations

Each dataset repository provides four configurations, one per task:

| Configuration | Task | Main fields |
|---|---|---|
| `depth_estimation` | monocular depth estimation | RGB image, metric depth (PNG, `depth_scale`), camera intrinsics |
| `instance_segmentation` | instance segmentation | RGB image, instance-ID mask, boxes (`boxes_xywh`), instance names, annotation JSON |
| `object_pose_estimation` | 6D object pose estimation | RGB, depth, instance mask, camera and object transformations (`world_from_camera`, `camera_from_objects`, …), object names and model keys |
| `object_pose_models` | CAD models | each referenced STL once per numbered dataset, with file name, SHA-256, size, scale and the binary payload |

Pose rows contain `object_model_keys`, so the multi-megabyte meshes are not
duplicated in every frame row. RGB, depth and masks are embedded as images, so
they render in the Hub dataset viewer. Hugging Face selects the configuration
rather than a Git branch, which keeps the shared files and the task tables on
the same reviewed revision.

## Repository layout

Each captured dataset is stored below its zero-padded ID, so a later capture
never replaces the files of an earlier one:

```text
owner/repository
├── 0001/
│   ├── images/  depth_maps/  reconstruction/  sam6d_scene/  sam6d_sequences/
│   └── data/
│       ├── depth_estimation/*.parquet
│       ├── instance_segmentation/*.parquet
│       └── object_pose_estimation/
│           ├── train-*.parquet
│           ├── validation-*.parquet
│           ├── test-*.parquet
│           └── models.parquet
├── 0002/
│   └── ...
└── README.md                      generated dataset card
```

The curated upload contains the RGB-D frames, the calibration, the metric point
cloud and camera poses, the STL files, the per-frame and final world poses, the
annotations, raw segmentations, instance-mask and overlay images, the pose-fitting
plots, the overlay video and the dataset card. COLMAP databases, PatchMatch
caches, rendered templates and inference intermediates are not uploaded.

## Splits

The training, validation and test split is deterministic: a frame is assigned
by hashing the dataset ID, the frame ID and a seed (default ratios 0.8 / 0.1 /
0.1, seed 42). Uploading a new numbered dataset appends new samples without
moving existing samples between splits. The ratios and the seed are set in the
**Review & publish** tab.

All numbered captures are loaded together. To use only some of them, filter the
`dataset_id` column.

## Loading examples

The examples use the hand-tool dataset; replace the repository name to use
another one.

**Select captures**

```python
from datasets import load_dataset

poses = load_dataset("ABC-iRobotics/Hand_Tools", "object_pose_estimation")
selected_ids = {"0000", "0001", "0004"}
for split_name in poses:
    poses[split_name] = poses[split_name].filter(
        lambda row: row["dataset_id"] in selected_ids
    )
```

**Depth estimation**

```python
from datasets import load_dataset
import numpy as np

depth_dataset = load_dataset("ABC-iRobotics/Hand_Tools", "depth_estimation")
sample = depth_dataset["train"][0]
rgb = sample["image"].convert("RGB")
depth_native = np.asarray(sample["depth"])
depth_mm = depth_native.astype(np.float32) * sample["depth_scale"]
depth_m = depth_mm / 1000.0
intrinsics_3x3 = np.asarray(sample["camera_intrinsics"]).reshape(3, 3)
```

**Instance segmentation**

```python
from datasets import load_dataset
import json
import numpy as np

instances = load_dataset("ABC-iRobotics/Hand_Tools", "instance_segmentation")
sample = instances["validation"][0]
rgb = sample["image"].convert("RGB")
instance_mask = np.asarray(sample["instance_mask"])
annotations = json.loads(sample["annotations_json"])
boxes_xywh = np.asarray(sample["boxes_xywh"])
instance_names = sample["instance_names"]
```

**Object pose estimation**, including the CAD models of a sample

```python
from datasets import load_dataset
import numpy as np

poses = load_dataset("ABC-iRobotics/Hand_Tools", "object_pose_estimation")
sample = poses["test"][0]
rgb = sample["image"].convert("RGB")
depth = np.asarray(sample["depth"])
instance_mask = np.asarray(sample["instance_mask"])
world_from_camera = np.asarray(sample["world_from_camera"]).reshape(4, 4)
camera_from_world = np.asarray(sample["camera_from_world"]).reshape(4, 4)
world_from_objects = np.asarray(sample["world_from_objects"]).reshape(-1, 4, 4)
camera_from_objects = np.asarray(sample["camera_from_objects"]).reshape(-1, 4, 4)
object_names = sample["object_names"]

# Resolve the STL models referenced by this pose sample.
models = load_dataset("ABC-iRobotics/Hand_Tools", "object_pose_models")["train"]
required_keys = set(sample["object_model_keys"])
models = models.filter(lambda row: row["object_model_key"] in required_keys)
first_model = models[0]
with open(first_model["stl_file_name"], "wb") as file:
    file.write(first_model["stl"])
```

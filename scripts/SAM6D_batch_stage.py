#!/usr/bin/env python3
"""Persistent ISM/PEM worker used by SAM6D_batch_inference.py.

Each invocation loads exactly one network, processes every requested frame, and
then exits so CUDA memory is released before the next model stage starts.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import importlib
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np


def report_cuda_memory(torch, label: str) -> None:
    if not torch.cuda.is_available():
        print(f"{label}: CUDA is unavailable; using CPU where supported", flush=True)
        return
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gib = 1024 ** 3
    print(
        f"{label}: CUDA free {free_bytes / gib:.2f} GiB / "
        f"{total_bytes / gib:.2f} GiB",
        flush=True,
    )


def release_unused_cuda_memory(torch, label: str) -> None:
    """Return memory held only by dead Python objects and PyTorch's cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        report_cuda_memory(torch, label)


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1:
        raise ValueError(f"Unsupported manifest version in {path}")
    return manifest


def import_stage_module(model_dir: Path):
    sys.path.insert(0, str(model_dir))
    os.chdir(model_dir)
    return importlib.import_module("run_inference_custom")


def object_cache_signature(manifest: dict, kind: str) -> dict:
    return {
        "version": 1,
        "kind": kind,
        "object_identity": manifest["object_identity"],
    }


def load_object_feature_cache(torch, manifest: dict, kind: str, device):
    template_dir = Path(manifest["templates_dir"])
    cache_path = template_dir / f".{kind}_features.pt"
    metadata_path = template_dir / f".{kind}_features.json"
    signature = object_cache_signature(manifest, kind)
    if not cache_path.is_file() or not metadata_path.is_file():
        return None
    try:
        if json.loads(metadata_path.read_text()) != signature:
            return None
        payload = torch.load(cache_path, map_location="cpu")
        return {
            key: value.to(device, non_blocking=True)
            for key, value in payload.items()
        }
    except (AttributeError, OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return None


def save_object_feature_cache(torch, manifest: dict, kind: str, payload) -> None:
    template_dir = Path(manifest["templates_dir"])
    cache_path = template_dir / f".{kind}_features.pt"
    metadata_path = template_dir / f".{kind}_features.json"
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(
        {key: value.detach().cpu() for key, value in payload.items()},
        temporary_path,
    )
    temporary_path.replace(cache_path)
    metadata_path.write_text(
        json.dumps(object_cache_signature(manifest, kind), indent=2) + "\n"
    )


def initialize_ism(single, manifest: dict, model=None, device=None):
    torch = single.torch
    from hydra import initialize_config_dir

    ism_dir = Path(single.__file__).resolve().parent
    with initialize_config_dir(
        version_base=None,
        config_dir=str(ism_dir / "configs"),
    ):
        cfg = single.compose(config_name="run_inference.yaml")

    segmentor = manifest["segmentor_model"]
    if segmentor == "sam":
        with initialize_config_dir(
            version_base=None,
            config_dir=str(ism_dir / "configs" / "model"),
        ):
            cfg.model = single.compose(config_name="ISM_sam.yaml")
        cfg.model.segmentor_model.stability_score_thresh = manifest[
            "stability_score_thresh"
        ]
    elif segmentor == "fastsam":
        with initialize_config_dir(
            version_base=None,
            config_dir=str(ism_dir / "configs" / "model"),
        ):
            cfg.model = single.compose(config_name="ISM_fastsam.yaml")
    else:
        raise ValueError(f"Unsupported segmentor model: {segmentor}")

    if model is None:
        logging.info("Initializing ISM once for all object sequences")
        model = single.instantiate(cfg.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
            report_cuda_memory(torch, "Before loading ISM descriptor")
        try:
            model.descriptor_model.model = model.descriptor_model.model.to(device)
        except torch.cuda.OutOfMemoryError as error:
            report_cuda_memory(torch, "ISM descriptor allocation failed")
            raise RuntimeError(
                "Not enough free GPU memory to initialize DINOv2-L. Close other "
                "CUDA applications or SAM-6D jobs, then retry."
            ) from error
        model.descriptor_model.model.device = device
        if hasattr(model.segmentor_model, "predictor"):
            model.segmentor_model.predictor.model = (
                model.segmentor_model.predictor.model.to(device)
            )
        else:
            model.segmentor_model.model.setup_model(device=device, verbose=True)
    elif device is None:
        raise ValueError("device is required when reusing an initialized ISM")

    template_dir = Path(manifest["templates_dir"])
    cached_reference = load_object_feature_cache(
        torch, manifest, "ism_reference", device
    )
    if cached_reference is not None:
        model.ref_data = cached_reference
        print(f"Reusing cached ISM template features: {template_dir}")
        return model, device
    num_templates = len(list(template_dir.glob("xyz_*.npy")))
    if num_templates == 0:
        raise RuntimeError(f"No templates found in {template_dir}")

    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        image = single.Image.open(template_dir / f"rgb_{idx}.png")
        mask = single.Image.open(template_dir / f"mask_{idx}.png")
        boxes.append(mask.getbbox())
        image_tensor = torch.from_numpy(
            np.array(image.convert("RGB")) / 255
        ).float()
        mask_tensor = torch.from_numpy(
            np.array(mask.convert("L")) / 255
        ).float()
        templates.append(image_tensor * mask_tensor[:, :, None])
        masks.append(mask_tensor.unsqueeze(-1))

    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.asarray(boxes))
    processing_config = single.OmegaConf.create({"image_size": 224})
    processor = single.CropResizePad(processing_config.image_size)
    templates = processor(images=templates, boxes=boxes).to(device)
    cropped_masks = processor(images=masks, boxes=boxes).to(device)

    model.ref_data = {}
    with torch.inference_mode():
        model.ref_data["descriptors"] = (
            model.descriptor_model.compute_features(
                templates, token_name="x_norm_clstoken"
            )
            .unsqueeze(0)
            .data
        )
        model.ref_data["appe_descriptors"] = (
            model.descriptor_model.compute_masked_patch_feature(
                templates, cropped_masks[:, 0, :, :]
            )
            .unsqueeze(0)
            .data
        )

    template_poses = single.get_obj_poses_from_template_level(
        level=2, pose_distribution="all"
    )
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses, dtype=torch.float32, device=device)
    model.ref_data["poses"] = poses[
        single.load_index_level_in_level2(0, "all")
    ]

    mesh = single.trimesh.load_mesh(manifest["cad_path"])
    # trimesh.sample uses NumPy's global generator; seed it for repeatability.
    np.random.seed(1)
    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(
        model_points, device=device
    ).unsqueeze(0)
    save_object_feature_cache(
        torch, manifest, "ism_reference", model.ref_data
    )
    # The reference descriptors, poses, and point cloud are now owned by
    # model.ref_data. The source template tensors are no longer needed.
    del templates, cropped_masks, masks, boxes, poses
    del image, mask, image_tensor, mask_tensor, mesh, model_points
    # Do not empty the CUDA allocator cache here. The next frame/object can
    # reuse those blocks without expensive cudaMalloc/cudaFree synchronization.
    gc.collect()
    return model, device


def save_empty_ism(output_dir: Path) -> None:
    results = output_dir / "sam6d_results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "detection_ism.json").write_text("[]\n")


def shared_cache_is_valid(frame: dict) -> bool:
    cache_path = Path(frame["shared_ism_cache"])
    metadata_path = Path(frame["shared_ism_metadata"])
    if not cache_path.is_file() or not metadata_path.is_file():
        return False
    try:
        return json.loads(metadata_path.read_text()) == frame[
            "shared_ism_signature"
        ]
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def save_shared_ism_cache(
    frame: dict,
    proposals: dict,
) -> None:
    cache_path = Path(frame["shared_ism_cache"])
    metadata_path = Path(frame["shared_ism_metadata"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    masks = proposals["masks"].detach().cpu().numpy().astype(bool)
    boxes = proposals["boxes"].detach().cpu().numpy()
    packed_masks = np.packbits(masks.reshape(-1), bitorder="little")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            packed_masks=packed_masks,
            mask_shape=np.asarray(masks.shape, dtype=np.int64),
            boxes=boxes,
        )
    temporary_path.replace(cache_path)
    metadata_path.write_text(
        json.dumps(frame["shared_ism_signature"], indent=2) + "\n"
    )


def load_shared_ism_cache(torch, frame: dict, device) -> dict:
    with np.load(frame["shared_ism_cache"], allow_pickle=False) as payload:
        shape = tuple(int(value) for value in payload["mask_shape"])
        count = int(np.prod(shape))
        masks = np.unpackbits(
            payload["packed_masks"], count=count, bitorder="little"
        ).reshape(shape).astype(np.float32)
        boxes = np.asarray(payload["boxes"])
    return {
        "masks": torch.from_numpy(masks).to(device),
        "boxes": torch.from_numpy(boxes).to(device),
    }


def shared_feature_cache_is_valid(frame: dict) -> bool:
    cache_path = Path(frame["shared_feature_cache"])
    metadata_path = Path(frame["shared_feature_metadata"])
    if not cache_path.is_file() or not metadata_path.is_file():
        return False
    try:
        return json.loads(metadata_path.read_text()) == frame[
            "shared_feature_signature"
        ]
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def load_shared_features(torch, frame: dict, device):
    payload = torch.load(frame["shared_feature_cache"], map_location="cpu")
    return (
        payload["query_descriptors"].to(device, non_blocking=True),
        payload["query_appearance"].to(device, non_blocking=True),
    )


def save_shared_features(torch, frame: dict, descriptors, appearance) -> None:
    cache_path = Path(frame["shared_feature_cache"])
    metadata_path = Path(frame["shared_feature_metadata"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(
        {
            "query_descriptors": descriptors.detach().cpu(),
            "query_appearance": appearance.detach().cpu(),
        },
        temporary_path,
    )
    temporary_path.replace(cache_path)
    metadata_path.write_text(
        json.dumps(frame["shared_feature_signature"], indent=2) + "\n"
    )


def infer_ism_frame(single, model, device, frame: dict, visualize: bool):
    torch = single.torch
    output_dir = Path(frame["output_dir"])
    results_dir = output_dir / "sam6d_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    legacy_raw_visualization = results_dir / "vis_ism_all.png"
    if legacy_raw_visualization.is_file():
        legacy_raw_visualization.unlink()
    rgb = single.Image.open(frame["rgb_path"]).convert("RGB")
    rgb_array = np.array(rgb)

    with torch.inference_mode():
        cache_hit = shared_cache_is_valid(frame)
        if cache_hit:
            try:
                proposals = load_shared_ism_cache(
                    torch, frame, device
                )
                print(
                    "  Shared ISM found; reusing: "
                    f"{frame['shared_ism_cache']}"
                )
            except (OSError, KeyError, RuntimeError, ValueError) as error:
                print(f"  Previous shared ISM is invalid; regenerating: {error}")
                cache_hit = False
        if not cache_hit:
            proposals = model.segmentor_model.generate_masks(rgb_array)
            save_shared_ism_cache(frame, proposals)
            print(
                "  No previous shared ISM found; generating: "
                f"{frame['shared_ism_cache']}"
            )

        shared_visualization = Path(frame["shared_ism_visualization"])
        if visualize and (not cache_hit or not shared_visualization.is_file()):
            shared_visualization.parent.mkdir(parents=True, exist_ok=True)
            single.visualize_raw_sam(
                rgb, proposals, shared_visualization
            )
        detections = single.Detections(proposals)
        detections.remove_very_small_detections(
            model.post_processing_config.mask_post_processing
        )
        if len(detections) == 0:
            save_empty_ism(output_dir)
            return

        feature_hit = shared_feature_cache_is_valid(frame)
        if feature_hit:
            try:
                query_descriptors, query_appearance = load_shared_features(
                    torch, frame, device
                )
                if query_descriptors.shape[0] != len(detections):
                    raise ValueError("cached proposal count does not match")
                print("  Shared DINO features found; reusing")
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                print(f"  Shared DINO feature cache is invalid: {error}")
                feature_hit = False
        if not feature_hit:
            query_descriptors, query_appearance = (
                model.descriptor_model.forward(rgb_array, detections)
            )
            save_shared_features(
                torch, frame, query_descriptors, query_appearance
            )
            print("  No previous shared DINO features found; generating")
        (
            selected,
            object_indices,
            semantic_score,
            best_template,
        ) = model.compute_semantic_score(query_descriptors)
        detections.filter(selected)
        if len(detections) == 0:
            save_empty_ism(output_dir)
            return
        query_appearance = query_appearance[selected]
        appearance_score, reference_aux = model.compute_appearance_score(
            best_template, object_indices, query_appearance
        )
        batch = single.batch_input_data(
            frame["depth_path"], frame["camera_path"], device
        )
        image_uv = model.project_template_to_image(
            best_template, object_indices, batch, detections.masks
        )
        geometric_score, visible_ratio = model.compute_geometric_score(
            image_uv,
            detections,
            query_appearance,
            reference_aux,
            visible_thred=model.visible_thred,
        )
        final_score = (
            semantic_score
            + appearance_score
            + geometric_score * visible_ratio
        ) / (2 + visible_ratio)
        detections.add_attribute("scores", final_score)
        detections.add_attribute(
            "object_ids", torch.zeros_like(final_score)
        )
        detections.to_numpy()

    save_path = results_dir / "detection_ism"
    detections.save_to_file(
        0, frame["frame_id"], 0, str(save_path), "Custom", False
    )
    json_detections = single.convert_npz_to_json(
        0, [str(save_path) + ".npz"]
    )
    single.save_json_bop23(str(save_path) + ".json", json_detections)
    if visualize and json_detections:
        image = single.visualize(
            rgb, json_detections, results_dir / "vis_ism.png"
        )
        image.save(results_dir / "vis_ism.png")


def run_ism(single, manifest: dict, model=None, device=None):
    pending = [frame for frame in manifest["frames"] if frame["run_ism"]]
    if not pending:
        print("ISM: no pending frames")
        return model, device
    torch = single.torch
    model, device = initialize_ism(single, manifest, model, device)
    for number, frame in enumerate(pending, 1):
        print(f"ISM [{number}/{len(pending)}] frame {frame['frame_id']}")
        infer_ism_frame(
            single,
            model,
            device,
            frame,
            visualize=not manifest["skip_visualizations"],
        )
    return model, device


def initialize_pem(single, manifest: dict, model=None, cfg=None):
    torch = single.torch
    if model is None:
        config_path = Path(manifest["pem_config"])
        cfg = single.gorilla.Config.fromfile(str(config_path))
        cfg.model_name = "pose_estimation_model"
        cfg.test_dataset.n_sample_observed_point = cfg.model.fine_npoint

        print("Initializing PEM once for all object sequences")
        model_module = importlib.import_module(cfg.model_name)
        model = model_module.Net(cfg.model).cuda().eval()
        checkpoint = (
            Path(single.__file__).resolve().parent
            / "checkpoints"
            / "sam-6d-pem-base.pth"
        )
        single.gorilla.solver.load_checkpoint(
            model=model, filename=str(checkpoint)
        )
    elif cfg is None:
        raise ValueError("cfg is required when reusing an initialized PEM")

    templates_dir = manifest["templates_dir"]
    cached_reference = load_object_feature_cache(
        torch, manifest, "pem_reference", torch.device("cuda")
    )
    if cached_reference is not None:
        print(f"Reusing cached PEM template features: {templates_dir}")
        return (
            cfg,
            model,
            cached_reference["template_points"],
            cached_reference["template_features"],
        )
    all_tem, all_tem_pts, all_tem_choose = single.get_templates(
        templates_dir, cfg.test_dataset
    )
    with torch.inference_mode():
        all_tem_pts, all_tem_feat = (
            model.feature_extraction.get_obj_feats(
                all_tem, all_tem_pts, all_tem_choose
            )
        )
    save_object_feature_cache(
        torch,
        manifest,
        "pem_reference",
        {
            "template_points": all_tem_pts,
            "template_features": all_tem_feat,
        },
    )
    # Feature extraction has produced the persistent template representation;
    # RGB crops and point-selection indices are not used during inference.
    del all_tem, all_tem_choose
    gc.collect()
    return cfg, model, all_tem_pts, all_tem_feat


def save_empty_pem(output_dir: Path) -> None:
    results = output_dir / "sam6d_results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "detection_pem.json").write_text("[]\n")


def infer_pem_frame(
    single,
    cfg,
    model,
    all_tem_pts,
    all_tem_feat,
    frame: dict,
    minimum_score: float,
    visualize: bool,
) -> None:
    torch = single.torch
    segmentation = json.loads(Path(frame["ism_json"]).read_text())
    if not any(float(item.get("score", 0)) > minimum_score for item in segmentation):
        save_empty_pem(Path(frame["output_dir"]))
        return

    random.seed(cfg.rd_seed + frame["frame_id"])
    np.random.seed(cfg.rd_seed + frame["frame_id"])
    torch.manual_seed(cfg.rd_seed + frame["frame_id"])
    try:
        input_data, image, _, model_points, detections = single.get_test_data(
            frame["rgb_path"],
            frame["depth_path"],
            frame["camera_path"],
            frame["cad_path"],
            frame["ism_json"],
            minimum_score,
            cfg.test_dataset,
        )
    except RuntimeError as error:
        message = str(error)
        if (
            "NO_VALID_DEPTH_DETECTIONS" not in message
            and "stack expects a non-empty TensorList" not in message
        ):
            raise
        save_empty_pem(Path(frame["output_dir"]))
        print(
            f"PEM frame {frame['frame_id']} skipped: ISM masks contain no "
            "usable depth-supported points"
        )
        return
    instances = input_data["pts"].size(0)
    with torch.inference_mode():
        input_data["dense_po"] = all_tem_pts.repeat(instances, 1, 1)
        input_data["dense_fo"] = all_tem_feat.repeat(instances, 1, 1)
        result = model(input_data)

    pose_scores = result.get("pred_pose_score", 1) * result["score"]
    pose_scores = pose_scores.detach().cpu().numpy()
    rotations = result["pred_R"].detach().cpu().numpy()
    translations = result["pred_t"].detach().cpu().numpy() * 1000
    for idx, detection in enumerate(detections):
        detection["score"] = float(pose_scores[idx])
        detection["R"] = rotations[idx].tolist()
        detection["t"] = translations[idx].tolist()

    results_dir = Path(frame["output_dir"]) / "sam6d_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "detection_pem.json").write_text(
        json.dumps(detections) + "\n"
    )
    if visualize and len(pose_scores):
        selected = pose_scores == pose_scores.max()
        intrinsics = input_data["K"].detach().cpu().numpy()[selected]
        vis = single.visualize(
            image,
            rotations[selected],
            translations[selected],
            model_points * 1000,
            intrinsics,
            results_dir / "vis_pem.png",
        )
        vis.save(results_dir / "vis_pem.png")


def run_pem(single, manifest: dict, model=None, cfg=None):
    pending = [frame for frame in manifest["frames"] if frame["run_pem"]]
    if not pending:
        print("PEM: no pending frames")
        return model, cfg
    torch = single.torch
    cfg, model, template_points, template_features = initialize_pem(
        single, manifest, model, cfg
    )
    try:
        for number, frame in enumerate(pending, 1):
            print(f"PEM [{number}/{len(pending)}] frame {frame['frame_id']}")
            infer_pem_frame(
                single,
                cfg,
                model,
                template_points,
                template_features,
                frame,
                manifest["det_score_thresh"],
                visualize=not manifest["skip_visualizations"],
            )
    finally:
        del template_points, template_features
        gc.collect()
    return model, cfg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("ism", "pem"), required=True)
    parser.add_argument(
        "--manifest", type=Path, action="append", required=True,
        help="May be repeated; the network is loaded once for all manifests.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    manifests = [load_manifest(path) for path in args.manifest]
    pending_key = "run_ism" if args.stage == "ism" else "run_pem"
    manifests = [
        manifest for manifest in manifests
        if any(frame[pending_key] for frame in manifest["frames"])
    ]
    if not manifests:
        print(f"{args.stage.upper()}: no pending frames")
        return 0
    # Prevent multiple UI sessions or batch commands from loading the large
    # ISM/PEM networks on the same GPU at once. flock is released automatically
    # if the process exits or is interrupted.
    dataset_dirs = {
        Path(manifest["dataset_dir"]).expanduser().resolve()
        for manifest in manifests
    }
    if len(dataset_dirs) != 1:
        raise ValueError("A stage worker can only process one dataset at a time")
    dataset_dir = dataset_dirs.pop()
    lock_path = dataset_dir / ".sam6d_gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            f"Another SAM-6D worker is using the GPU; waiting for {lock_path}",
            flush=True,
        )
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
    print(f"Acquired SAM-6D GPU lock: {lock_path}", flush=True)
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    print(
        f"{args.stage.upper()}: importing PyTorch and SAM-6D dependencies "
        "(this can take a while on the first run)",
        flush=True,
    )
    try:
        single = import_stage_module(args.model_dir.resolve())
        if single.torch.cuda.is_available():
            # Stable image/template sizes benefit from cuDNN autotuning. TF32
            # uses the A100 tensor cores while retaining float32 tensors and
            # is appropriate for inference scoring.
            single.torch.backends.cudnn.benchmark = True
            allow_tf32 = os.environ.get("SAM6D_ALLOW_TF32", "1") != "0"
            single.torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            single.torch.backends.cudnn.allow_tf32 = allow_tf32
            if hasattr(single.torch, "set_float32_matmul_precision"):
                single.torch.set_float32_matmul_precision(
                    "high" if allow_tf32 else "highest"
                )
        report_cuda_memory(single.torch, f"Before {args.stage.upper()} initialization")
        model = None
        auxiliary = None
        try:
            for number, manifest in enumerate(manifests, 1):
                print(
                    f"\n{args.stage.upper()} object "
                    f"[{number}/{len(manifests)}]: {manifest['object_name']}",
                    flush=True,
                )
                if args.stage == "ism":
                    model, auxiliary = run_ism(
                        single, manifest, model, auxiliary
                    )
                else:
                    model, auxiliary = run_pem(
                        single, manifest, model, auxiliary
                    )
        finally:
            if model is not None:
                del model
            release_unused_cuda_memory(
                single.torch, f"After releasing {args.stage.upper()} model"
            )
        return 0
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Live point-cloud reconstruction pipeline entrypoint.

The script supports a two stage workflow:

* **Test mode**: bootstrap a metric map by running a VGGT TensorRT engine on
  the first 8 images.  The resulting point cloud (or TSDF) is persisted.
* **Live updates**: subsequent frames are processed with a Depth Anything v2
  TensorRT engine.  Each relative depth map is aligned to the current map,
  back-projected and fused.

The implementation keeps the heavy lifting in helper modules under
``onnx/`` and orchestrates them here.  All code is additive so existing
scripts remain untouched.
"""
from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .pcd import align, depth_anything, fusion, io_utils, raycast, vggt_trt


LOGGER = logging.getLogger("pcd")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live point-cloud reconstruction using VGGT + Depth Anything",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="images",
        choices=["images", "videos", "webcams"],
        help="Input modality for live frames.",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Directory of per-camera images (used when --source=images).",
    )
    parser.add_argument(
        "--images_live",
        type=str,
        default=None,
        help="Optional separate directory for live frames in image mode.",
    )
    parser.add_argument(
        "--videos",
        type=str,
        default=None,
        help="Directory containing per-camera videos (used when --source=videos).",
    )
    parser.add_argument(
        "--webcams",
        type=str,
        nargs="*",
        help="List of webcam indices or name=index pairs (used when --source=webcams).",
    )
    parser.add_argument(
        "--num_cams",
        type=int,
        default=8,
        help="Number of camera views to use for VGGT bootstrap and live updates.",
    )
    parser.add_argument(
        "--views",
        type=str,
        default=None,
        help="Comma separated list of camera IDs to use (overrides auto discovery).",
    )
    parser.add_argument(
        "--random_views",
        type=int,
        default=0,
        help="When >0, randomly sample this many cameras from the available set.",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=1,
        help="Process every Nth frame in streaming modes (default: 1).",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional limit on the number of live frame batches to process.",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        required=True,
        help="Camera intrinsics (JSON/YAML/NPZ).",
    )
    parser.add_argument(
        "--poses",
        type=str,
        required=True,
        help="Camera to world poses (JSON/YAML/NPZ).",
    )
    parser.add_argument(
        "--initial_map",
        type=str,
        default=None,
        help="Existing map (.ply or .npz). If not provided and --test_mode=1,"
        " bootstrap from VGGT.",
    )
    parser.add_argument(
        "--vggt_engine",
        type=str,
        default=None,
        help="VGGT TensorRT engine used for bootstrap.",
    )
    parser.add_argument(
        "--depth_engine_dir",
        type=str,
        default="onnx_exports/depth_anything",
        help="Directory containing Depth Anything TensorRT engines.",
    )
    parser.add_argument(
        "--depth_workers",
        type=int,
        default=None,
        help="Number of Depth Anything workers/engines to spawn (default: num_cams).",
    )
    parser.add_argument(
        "--trt_precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16", "fp8", "int8"],
        help="Preferred TensorRT precision.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="surfel",
        choices=["surfel", "tsdf"],
        help="Fusion back-end to use.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel size (m) for TSDF or surfel radius proxy.",
    )
    parser.add_argument(
        "--trunc_margin",
        type=float,
        default=0.04,
        help="TSDF truncation margin (m).",
    )
    parser.add_argument(
        "--align_model",
        type=str,
        default="auto",
        choices=["auto", "scale", "scale_shift", "invdepth_scale_shift"],
        help="Alignment model selection strategy.",
    )
    parser.add_argument(
        "--smooth_scale",
        type=float,
        default=0.0,
        help="EMA factor for per-camera scale smoothing (0 disables).",
    )
    parser.add_argument(
        "--per_camera_calib",
        type=str,
        default=None,
        help="Optional JSON to persist per-camera scale/shift.",
    )
    parser.add_argument(
        "--save_per_frame_depth",
        type=int,
        default=0,
        help="When non-zero, save aligned metric depth per frame.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to store outputs (map, metrics, aligned depth).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--test_mode",
        type=int,
        default=0,
        help="Enable VGGT bootstrap then live simulation when set to 1.",
    )
    return parser


def _parse_view_list(views: Optional[str]) -> Optional[List[str]]:
    if views is None:
        return None
    parsed = [view.strip() for view in views.split(",") if view.strip()]
    return parsed or None


def _build_frame_provider(
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[io_utils.FrameProvider, List[io_utils.FrameData]]:
    requested_views = _parse_view_list(args.views)
    num_cams = max(1, int(args.num_cams))
    if args.source == "images":
        root = args.images_live or args.images
        if root is None:
            raise RuntimeError("--images (or --images_live) required when --source=images")
    else:
        root = None

    if args.source == "images":
        provider = io_utils.ImageSequenceProvider(
            Path(root),
            num_cams=num_cams,
            requested_views=requested_views,
            random_views=args.random_views,
            frame_step=args.frame_step,
            max_batches=args.max_batches,
            rng=rng,
        )
    elif args.source == "videos":
        if args.videos is None:
            raise RuntimeError("--videos is required when --source=videos")
        videos = io_utils.discover_video_files(Path(args.videos))
        provider = io_utils.VideoStreamProvider(
            videos,
            num_cams=num_cams,
            requested_views=requested_views,
            random_views=args.random_views,
            frame_step=args.frame_step,
            max_batches=args.max_batches,
            rng=rng,
        )
    elif args.source == "webcams":
        if not args.webcams:
            raise RuntimeError("--webcams is required when --source=webcams")
        webcams = io_utils.parse_webcam_spec(args.webcams)
        selected = webcams
        if requested_views:
            missing = [cam for cam in requested_views if cam not in webcams]
            if missing:
                raise RuntimeError(f"Requested webcams not available: {missing}")
            selected = {cam: webcams[cam] for cam in requested_views}
        elif args.random_views > 0:
            if args.random_views > len(webcams):
                raise RuntimeError("Cannot sample more webcams than available.")
            chosen = rng.sample(list(webcams.keys()), args.random_views)
            selected = {cam: webcams[cam] for cam in chosen}
        if num_cams < len(selected):
            chosen = list(selected.keys())[:num_cams]
            selected = {cam: selected[cam] for cam in chosen}
        provider = io_utils.WebcamStreamProvider(
            selected,
            frame_step=args.frame_step,
            max_batches=args.max_batches,
        )
    else:  # pragma: no cover - parser guards choices
        raise RuntimeError(f"Unsupported source type: {args.source}")

    bootstrap_frames = provider.bootstrap(num_cams)
    if not bootstrap_frames:
        raise RuntimeError("No frames available for VGGT bootstrap.")
    return provider, bootstrap_frames


def _load_initial_map(
    path: Optional[Path],
    fusion_mode: str,
    voxel_size: float,
    trunc_margin: float,
) -> tuple[Optional[fusion.SurfelMap], Optional[fusion.TSDFVolume]]:
    if path is None:
        return None, None
    path = Path(path)
    if path.suffix.lower() == ".ply":
        points = io_utils.load_point_cloud(path)
        return fusion.SurfelMap(positions=points), None
    if path.suffix.lower() == ".npz":
        coords, values = io_utils.load_tsdf(path)
        origin = coords.min(axis=0) if coords.size else np.zeros(3, dtype=np.float32)
        tsdf = fusion.TSDFVolume(
            origin=origin,
            voxel_size=voxel_size,
            trunc_margin=trunc_margin,
        )
        for coord, value in zip(coords, values):
            idx = tuple(np.floor((coord - tsdf.origin) / tsdf.voxel_size).astype(int))
            tsdf.voxels[idx] = (value, 1.0)
        surfels = fusion.SurfelMap(positions=tsdf.to_point_cloud())
        return surfels, tsdf
    raise ValueError(f"Unknown map extension: {path}")


def bootstrap_initial_map(
    bootstrap_frames: Sequence[io_utils.FrameData],
    intrinsics: Dict[str, np.ndarray],
    poses: Dict[tuple[str, int], np.ndarray],
    vggt_engine: Path,
    fusion_mode: str,
    voxel_size: float,
    trunc_margin: float,
) -> tuple[fusion.SurfelMap, Optional[fusion.TSDFVolume]]:
    LOGGER.info("Bootstrapping initial metric map from VGGT.")
    images = [frame.image for frame in bootstrap_frames]
    if not images:
        raise RuntimeError("Bootstrap requires at least one frame.")
    vggt = vggt_trt.TRTVGGT(engine=vggt_engine)
    start = time.perf_counter()
    result = vggt.run(images)
    duration = (time.perf_counter() - start) * 1000.0
    LOGGER.info("VGGT inference completed in %.1f ms", duration)
    depth_maps = result.depth_maps or [np.ones(images[0].shape[:2], dtype=np.float32)] * len(images)
    Ks = []
    for frame in bootstrap_frames:
        if frame.camera_id not in intrinsics:
            raise RuntimeError(f"Missing intrinsics for camera {frame.camera_id}")
        Ks.append(intrinsics[frame.camera_id])
    poses_wc = []
    for frame in bootstrap_frames:
        pose = poses.get((frame.camera_id, frame.frame_id)) or poses.get((frame.camera_id, 0))
        if pose is None:
            raise RuntimeError(f"Missing pose for bootstrap frame {frame.camera_id}/{frame.frame_id}")
        poses_wc.append(pose)
    surfels = fusion.SurfelMap.from_depth_batch(depth_maps, Ks, poses_wc)
    tsdf_volume: Optional[fusion.TSDFVolume] = None
    if fusion_mode == "tsdf":
        tsdf_volume = fusion.TSDFVolume(
            origin=surfels.positions.min(axis=0) if surfels.positions.size else np.zeros(3, dtype=np.float32),
            voxel_size=voxel_size,
            trunc_margin=trunc_margin,
        )
        for depth, K, pose in zip(depth_maps, Ks, poses_wc):
            tsdf_volume.integrate(depth, K, pose)
        surfels = fusion.SurfelMap(positions=tsdf_volume.to_point_cloud())
    return surfels, tsdf_volume


def run_pipeline(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    intrinsics = io_utils.load_intrinsics(Path(args.intrinsics))
    poses = io_utils.load_poses(Path(args.poses))

    calib_path = Path(args.per_camera_calib) if args.per_camera_calib else None
    scale_cache = io_utils.load_per_camera_calibration(calib_path)
    smoother = align.ScaleShiftEMA(alpha=float(args.smooth_scale))
    for cam_id, (scale, shift) in scale_cache.items():
        smoother.set_state(cam_id, scale, shift)

    rng = random.Random(args.seed)
    provider, bootstrap_frames = _build_frame_provider(args, rng)
    active_cams = [frame.camera_id for frame in bootstrap_frames]
    LOGGER.info("Active cameras: %s", ", ".join(active_cams))

    surfel_map, tsdf_map = _load_initial_map(
        Path(args.initial_map) if args.initial_map else None,
        args.fusion,
        args.voxel_size,
        args.trunc_margin,
    )

    if surfel_map is None and tsdf_map is None:
        if not args.test_mode:
            raise RuntimeError("--initial_map is required unless --test_mode=1")
        if args.vggt_engine is None:
            raise RuntimeError("--vggt_engine must be provided for test mode bootstrap")
        surfel_map, tsdf_map = bootstrap_initial_map(
            bootstrap_frames,
            intrinsics,
            poses,
            Path(args.vggt_engine),
            args.fusion,
            args.voxel_size,
            args.trunc_margin,
        )
        io_utils.save_point_cloud(out_dir / "bootstrap_map.ply", surfel_map.as_point_cloud())

    depth_workers = args.depth_workers or len(active_cams)
    metrics: List[dict] = []
    try:
        with depth_anything.DepthAnythingPool(
            Path(args.depth_engine_dir),
            precision=args.trt_precision,
            num_workers=max(1, int(depth_workers)),
        ) as depth_pool:
            LOGGER.info("Depth Anything engine: %s", depth_pool.engine_path)
            for batch in provider.iter_batches():
                batch_start = time.perf_counter()
                if not batch.frames:
                    continue
                images = {cam_id: frame.image for cam_id, frame in batch.frames.items()}
                depth_start = time.perf_counter()
                rel_depths = depth_pool.infer_batch(images)
                depth_ms = (time.perf_counter() - depth_start) * 1000.0

                for cam_id, frame in batch.frames.items():
                    K = intrinsics.get(cam_id)
                    pose = poses.get((cam_id, frame.frame_id)) or poses.get((cam_id, 0))
                    if K is None or pose is None:
                        LOGGER.warning("Missing calibration for %s frame %d", cam_id, frame.frame_id)
                        continue
                    rel_depth = rel_depths.get(cam_id)
                    if rel_depth is None:
                        LOGGER.warning("Depth Anything produced no output for camera %s", cam_id)
                        continue

                    if args.fusion == "tsdf" and tsdf_map is not None:
                        expected = raycast.expected_depth_from_surfel(
                            tsdf_map.to_point_cloud(),
                            pose,
                            K,
                            frame.image.shape[:2],
                        )
                    else:
                        expected = raycast.expected_depth_from_surfel(
                            surfel_map.positions if surfel_map is not None else np.empty((0, 3), dtype=np.float32),
                            pose,
                            K,
                            frame.image.shape[:2],
                        )

                    try:
                        alignment = align.auto_select_model(
                            rel_depth,
                            expected,
                            mask=np.isfinite(expected),
                            strategy=args.align_model,
                        )
                    except ValueError as exc:
                        LOGGER.warning(
                            "Alignment skipped for batch %d camera %s: %s",
                            batch.index,
                            cam_id,
                            exc,
                        )
                        continue
                    alignment = smoother.update(cam_id, alignment)
                    metric_depth = alignment.apply(rel_depth)

                    if args.fusion == "tsdf":
                        if tsdf_map is None:
                            tsdf_map = fusion.TSDFVolume(
                                origin=np.zeros(3, dtype=np.float32),
                                voxel_size=args.voxel_size,
                                trunc_margin=args.trunc_margin,
                            )
                        tsdf_map.integrate(metric_depth, K, pose)
                        surfel_map = fusion.SurfelMap(positions=tsdf_map.to_point_cloud())
                    else:
                        surfel_map.fuse(metric_depth, K, pose)

                    if args.save_per_frame_depth:
                        depth_dir = out_dir / "depth_metric"
                        depth_dir.mkdir(parents=True, exist_ok=True)
                        io_utils.save_depth(
                            depth_dir / f"camera_{cam_id}_batch_{batch.index:05d}.npz",
                            metric_depth,
                        )

                    metrics.append(
                        {
                            "batch": batch.index,
                            "camera": cam_id,
                            "frame": int(frame.frame_id),
                            "model": alignment.model.value,
                            "scale": alignment.scale,
                            "shift": alignment.shift,
                            "residual": alignment.residual,
                            "depth_ms": depth_ms,
                            "total_ms": (time.perf_counter() - batch_start) * 1000.0,
                        }
                    )
    finally:
        provider.close()

    if surfel_map is None:
        raise RuntimeError("Fusion produced no map output.")

    io_utils.save_point_cloud(out_dir / "map_updated.ply", surfel_map.as_point_cloud())
    if tsdf_map is not None:
        coords, values = tsdf_map.export()
        io_utils.save_tsdf(out_dir / "tsdf_volume.npz", coords, values)
    io_utils.save_metrics(out_dir / "metrics.json", metrics)
    persisted = {
        cam: state
        for cam in intrinsics.keys()
        for state in [smoother.get_state(cam)]
        if state is not None
    }
    io_utils.save_per_camera_calibration(calib_path, persisted)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover
    main()

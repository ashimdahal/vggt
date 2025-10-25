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
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - OpenCV optional
    cv2 = None

from .pcd import align, depth_anything, fusion, io_deepview, io_utils, raycast, vggt_trt


LOGGER = logging.getLogger("pcd")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live point-cloud reconstruction using VGGT + Depth Anything.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="generic",
        choices=["generic", "deepview"],
        help="Dataset adapter to use (generic directory layout or DeepView light-field).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Root directory containing scenes for the selected dataset.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene identifier within --dataset-root (DeepView or generic).",
    )
    parser.add_argument(
        "--n-views",
        type=int,
        default=8,
        help="Number of camera views to use for VGGT bootstrap and live updates.",
    )
    parser.add_argument(
        "--view-select",
        type=str,
        default="max-spread",
        choices=["max-spread", "front-arc", "random"],
        help="Automatic view selection strategy when --views is not provided.",
    )
    parser.add_argument(
        "--views",
        type=str,
        default=None,
        help="Comma separated list of camera IDs to use (overrides auto selection).",
    )
    parser.add_argument(
        "--undistort",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Fisheye undistortion mode (auto enables for fisheye cameras only).",
    )
    parser.add_argument(
        "--rectify-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(518, 518),
        help="Target resolution for rectified images (VGGT expects 518x518).",
    )
    parser.add_argument(
        "--cache-rectify-maps",
        action="store_true",
        help="Persist fisheye rectification maps to disk for reuse.",
    )
    parser.add_argument(
        "--validate-projection",
        action="store_true",
        help="Project the DeepView README reference world point and report pixels.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Initial frame index to bootstrap from (DeepView/video datasets).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame when iterating through sequences.",
    )
    parser.add_argument(
        "--timestamp",
        type=float,
        default=None,
        help="Optional timestamp hint (reserved for future synchronization hooks).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="images",
        choices=["images", "videos", "webcams"],
        help="Legacy generic dataset input modality (used when --dataset=generic).",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Directory of per-camera images (used for generic datasets).",
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
        help="Directory containing per-camera videos (generic mode).",
    )
    parser.add_argument(
        "--webcams",
        type=str,
        nargs="*",
        help="List of webcam indices or name=index pairs (generic live capture).",
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
        default=None,
        help="Camera intrinsics (JSON/YAML/NPZ) for generic datasets.",
    )
    parser.add_argument(
        "--poses",
        type=str,
        default=None,
        help="Camera-to-world poses (JSON/YAML/NPZ) for generic datasets.",
    )
    parser.add_argument(
        "--initial_map",
        type=str,
        default=None,
        help="Existing map (.ply or .npz). If not provided and --test-mode=1 bootstrap from VGGT.",
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
        help="Number of Depth Anything workers/engines to spawn (default: n-views).",
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
        default=42,
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
        "--test-mode",
        type=int,
        default=0,
        help="Enable VGGT bootstrap then simulated live updates when set to 1.",
    )
    # Legacy compatibility flags (suppressed in help but accepted).
    parser.add_argument(
        "--num_cams",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--random_views",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    return parser


def _parse_view_list(views: Optional[str]) -> Optional[List[str]]:
    if views is None:
        return None
    parsed = [view.strip() for view in views.split(",") if view.strip()]
    return parsed or None


def _normalize_args(args: argparse.Namespace) -> None:
    """
    Harmonize legacy aliases and enforce defaults on the parsed arguments.
    """
    if getattr(args, "num_cams", None):
        args.n_views = int(args.num_cams)
    if getattr(args, "frame_step", None):
        args.frame_stride = int(args.frame_step)
    args.n_views = max(1, int(args.n_views))
    args.frame_stride = max(1, int(args.frame_stride))


def _scale_intrinsics_local(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = np.array(K, dtype=np.float32)
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for image resizing.")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


class DeepViewFrameProvider(io_utils.FrameProvider):
    """
    Frame provider that wraps ``DeepViewDataset`` objects and exposes the
    legacy FrameProvider interface expected by the pipeline.
    """

    def __init__(
        self,
        dataset: io_deepview.DeepViewDataset,
        camera_ids: Sequence[str],
        *,
        start_frame: int,
        frame_stride: int,
        max_batches: Optional[int],
    ) -> None:
        self.dataset = dataset
        self.camera_ids = list(camera_ids)
        self.start_frame = max(0, int(start_frame))
        self.frame_stride = max(1, int(frame_stride))
        self.max_batches = max_batches

    def bootstrap(self, num_cams: int) -> List[io_utils.FrameData]:
        frames: List[io_utils.FrameData] = []
        count = min(num_cams, len(self.camera_ids))
        if count == 0:
            raise RuntimeError("DeepView dataset produced no cameras for bootstrap.")
        for cam_id in self.camera_ids[:count]:
            image = self.dataset.get_frame(cam_id, self.start_frame)
            frames.append(
                io_utils.FrameData(
                    camera_id=cam_id,
                    frame_id=self.start_frame,
                    image=image,
                    path=None,
                    timestamp=None,
                )
            )
        return frames

    def iter_batches(self) -> Iterator[io_utils.FrameBatch]:
        batch_idx = 0
        frame_idx = self.start_frame
        while True:
            frames: Dict[str, io_utils.FrameData] = {}
            try:
                for cam_id in self.camera_ids:
                    image = self.dataset.get_frame(cam_id, frame_idx)
                    frames[cam_id] = io_utils.FrameData(
                        camera_id=cam_id,
                        frame_id=frame_idx,
                        image=image,
                        path=None,
                        timestamp=None,
                    )
            except RuntimeError:
                break
            yield io_utils.FrameBatch(index=batch_idx, frames=frames)
            batch_idx += 1
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break
            frame_idx += self.frame_stride

    def close(self) -> None:
        self.dataset.close()


def _save_rectified_intrinsics(out_dir: Path, intrinsics: Dict[str, np.ndarray]) -> None:
    target = out_dir / "calibration_rectified"
    target.mkdir(parents=True, exist_ok=True)
    for cam_id, matrix in intrinsics.items():
        payload = {
            "camera": cam_id,
            "matrix": matrix.tolist(),
        }
        path = target / f"{cam_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


REFERENCE_WORLD_POINT = np.array([0.04770101, 0.04799868, 1.28055], dtype=np.float32)


def _validate_projection(cam_id: str, params: Dict[str, np.ndarray]) -> None:
    """
    Project the DeepView README reference world-space point and log the pixel
    coordinates.  Expected result is approximately [1377.855, 1017.614].
    """
    world = REFERENCE_WORLD_POINT
    R = params["R"]
    t = params["t"]
    point_c = R @ world + t
    x = float(point_c[0])
    y = float(point_c[1])
    z = float(point_c[2])
    r = float(np.hypot(x, y))
    theta = float(np.arctan2(r, z))
    if r < 1e-8:
        theta_over_r = 1.0
    else:
        theta_over_r = theta / r
    r2 = theta * theta
    k1, k2, k3 = params["distortion"]
    distortion = 1.0 + r2 * (float(k1) + r2 * float(k2))
    distortion += (r2 * r2) * float(k3)
    x_norm = theta_over_r * x * distortion
    y_norm = theta_over_r * y * distortion
    K_raw = params["K_raw"]
    u = float(K_raw[0, 0] * x_norm + K_raw[0, 2])
    v = float(K_raw[1, 1] * y_norm + K_raw[1, 2])
    LOGGER.info("Projection check for %s -> [%.3f, %.3f]", cam_id, u, v)


def _build_generic_provider(
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[io_utils.FrameProvider, List[io_utils.FrameData]]:
    requested_views = _parse_view_list(args.views)
    num_cams = max(1, int(args.n_views))
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
            random_views=getattr(args, "random_views", 0),
            frame_step=args.frame_stride,
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
            random_views=getattr(args, "random_views", 0),
            frame_step=args.frame_stride,
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
        elif getattr(args, "random_views", 0) > 0:
            if args.random_views > len(webcams):
                raise RuntimeError("Cannot sample more webcams than available.")
            chosen = rng.sample(list(webcams.keys()), args.random_views)
            selected = {cam: webcams[cam] for cam in chosen}
        if num_cams < len(selected):
            chosen = list(selected.keys())[:num_cams]
            selected = {cam: selected[cam] for cam in chosen}
        provider = io_utils.WebcamStreamProvider(
            selected,
            frame_step=args.frame_stride,
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


def run_pipeline(args: argparse.Namespace) -> int:
    _normalize_args(args)
    logging.basicConfig(level=getattr(logging, args.log_level))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rectify_width, rectify_height = map(int, args.rectify_size)
    rectify_size = (rectify_width, rectify_height)

    calib_path = Path(args.per_camera_calib) if args.per_camera_calib else None
    scale_cache = io_utils.load_per_camera_calibration(calib_path)
    smoother = align.ScaleShiftEMA(alpha=float(args.smooth_scale))
    for cam_id, (scale, shift) in scale_cache.items():
        smoother.set_state(cam_id, scale, shift)

    intrinsics: Dict[str, np.ndarray] = {}
    poses: Dict[tuple[str, int], np.ndarray] = {}
    resize_targets: Dict[str, Tuple[int, int]] = {}
    raw_params: Dict[str, Dict[str, np.ndarray]] = {}

    rng = random.Random(args.seed)
    if args.dataset == "deepview":
        if args.dataset_root is None or args.scene is None:
            raise RuntimeError("--dataset-root and --scene are required for DeepView datasets.")
        dataset = io_deepview.DeepViewDataset(
            root=args.dataset_root,
            scene=args.scene,
            undistort=args.undistort.lower() != "off",
            rectify_to_size=rectify_size,
            cache_maps=args.cache_rectify_maps,
            seed=args.seed,
            frame_stride=args.frame_stride,
            default_frame=args.frame_index,
        )
        available = dataset.list_cameras()
        requested = _parse_view_list(args.views)
        if requested:
            missing = [cam for cam in requested if cam not in available]
            if missing:
                raise RuntimeError(f"Requested DeepView cameras not found: {missing}")
            selected = requested[:]
        else:
            selected = io_deepview.select_views(
                available,
                args.n_views,
                method=args.view_select,
                seed=args.seed,
            )
        provider: io_utils.FrameProvider = DeepViewFrameProvider(
            dataset,
            selected,
            start_frame=args.frame_index,
            frame_stride=args.frame_stride,
            max_batches=args.max_batches,
        )
        bootstrap_frames = provider.bootstrap(len(selected))
        if not bootstrap_frames:
            raise RuntimeError("No frames available for VGGT bootstrap.")
        for cam_id in selected:
            params = dataset.get_camera_params(cam_id)
            raw_params[cam_id] = params
            intrinsics[cam_id] = params["K"]
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = params["R"].T
            pose[:3, 3] = params["center"]
            poses[(cam_id, 0)] = pose
            poses[(cam_id, args.frame_index)] = pose
        active_cams = [frame.camera_id for frame in bootstrap_frames]
        LOGGER.info("Active DeepView cameras: %s", ", ".join(active_cams))
        _save_rectified_intrinsics(out_dir, intrinsics)
        if args.validate_projection and active_cams:
            first_cam = active_cams[0]
            _validate_projection(first_cam, raw_params[first_cam])
    else:
        if args.intrinsics is None or args.poses is None:
            raise RuntimeError("Generic datasets require --intrinsics and --poses.")
        intrinsics_raw = io_utils.load_intrinsics(Path(args.intrinsics))
        poses = io_utils.load_poses(Path(args.poses))
        provider, bootstrap_frames = _build_generic_provider(args, rng)
        if not bootstrap_frames:
            raise RuntimeError("No frames available for VGGT bootstrap.")
        active_cams = [frame.camera_id for frame in bootstrap_frames]
        LOGGER.info("Active generic cameras: %s", ", ".join(active_cams))
        for frame in bootstrap_frames:
            cam_id = frame.camera_id
            image = frame.image
            h, w = image.shape[:2]
            scale_x = rectify_width / float(w)
            scale_y = rectify_height / float(h)
            if (w, h) != (rectify_width, rectify_height):
                resize_targets[cam_id] = (rectify_width, rectify_height)
                frame.image = _resize_image(image, rectify_width, rectify_height)
            K_raw = intrinsics_raw.get(cam_id)
            if K_raw is None:
                raise RuntimeError(f"Missing intrinsics for camera {cam_id}")
            intrinsics[cam_id] = _scale_intrinsics_local(K_raw, scale_x, scale_y)
        _save_rectified_intrinsics(out_dir, intrinsics)

    surfel_map, tsdf_map = _load_initial_map(
        Path(args.initial_map) if args.initial_map else None,
        args.fusion,
        args.voxel_size,
        args.trunc_margin,
    )

    if surfel_map is None and tsdf_map is None:
        if not args.test_mode:
            raise RuntimeError("--initial_map is required unless --test-mode=1")
        if args.vggt_engine is None:
            raise RuntimeError("--vggt_engine must be provided for test mode bootstrap")
        LOGGER.info(
            "VGGT input tensor shape: (1, %d, 3, %d, %d)",
            len(bootstrap_frames),
            rectify_height,
            rectify_width,
        )
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
                for cam_id, frame in batch.frames.items():
                    if cam_id in resize_targets:
                        width, height = resize_targets[cam_id]
                        frame.image = _resize_image(frame.image, width, height)
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
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# VGGT TensorRT Toolkit

This directory bundles everything needed to export `facebook/VGGT-1B` to ONNX/TensorRT and to run live multi-view reconstruction. The goal is to stay within `onnx/` for export, benchmarking, and runtime pipelines, while the rest of the repository focuses on higher-level demos and training.

---

## 1. Directory Map

| Path | Purpose |
| --- | --- |
| `inference_pcd.py` | End-to-end point-cloud reconstruction (bootstrap with VGGT, stream Depth Anything). |
| `pcd/` | Support modules: alignment, fusion, IO helpers, raycasting, TRT wrappers, Depth Anything pool. |
| `tools/` | Standalone CLI utilities for export, inference, benchmarking, and 3DGS experiments. |
| `logs/` | Build & runtime logs (e.g. `logs/out.txt` summarises the latest benchmark sweep). |
| `onnx_exports/` | Populated after exports; holds ONNX graphs, TensorRT engines, tactic caches. |

Top-level helper scripts now live in `scripts/`. They reference the updated tool locations—see §7.

---

## 2. Environment Setup

Minimum stack:

- Python 3.12 or later
- CUDA 12.8+ with TensorRT ≥ 10.10
- PyTorch 2.8+ (CUDA build)
- PyCUDA, Pillow, NumPy

Optional extras:

- OpenCV (`pip install opencv-python`) for video/webcam streaming.
- CuPy / Open3D for GPU preprocessing and visualisation in `tools/inference_depth_anything.py`.
- bitsandbytes or NVIDIA ModelOpt when experimenting with low-precision quant modes.

Quick install example:

```bash
pip install torch torchvision 
pip install onnx onnxsim onnxruntime-gpu pycuda tensorrt pillow numpy tqdm
pip install opencv-python
```

---

## 3. Exporting VGGT to TensorRT

Use `onnx/tools/vggt_to_trt.py` to export ONNX models and build TensorRT engines.

```bash
python onnx/tools/vggt_to_trt.py \
  --export \
  --model-name facebook/VGGT-1B \
  --num-cams 8 --height 518 --width 518 \
  --precision fp16 \
  --pcd-only \
  --output-dir onnx_exports/none
```

Key parameters:

- `--quant-mode`: `none`, `torch-int8-dynamic`, `bitsandbytes-8bit`, `bitsandbytes-nf4`, `bitsandbytes-fp4`, `modelopt-fp8`, `modelopt-nvfp4`.
- `--all-precisions`: build FP16/BF16/FP8/INT8 after a single export.
- `--calib-data` + `--calib-batches`: drive INT8 calibration (directories, globs, or `.npy` tensors).
- `--pcd-only`: strips the classifier heads to keep depth + camera outputs (smaller/faster engines).

Artifacts are placed in `onnx_exports/<quant_mode>/` with filenames like `vggt-8x3x518x518-pcd_fp16.engine`.

---

## 4. Live Point-Cloud Reconstruction

`inference_pcd.py` performs a two-phase loop:

1. **Bootstrap** — runs VGGT on `--num_cams` frames to produce a metric point cloud (surfel or TSDF).
2. **Streaming** — processes subsequent frames with Depth Anything v2, aligns relative depth to the map, and fuses them.

### 4.1 Inputs

- `--source images` *(default)*: PKU-style image folders; use `--images` (and optionally `--images_live`).
- `--source videos`: per-camera video files (flat directory or per-camera subdirectories).
- `--source webcams`: multiple USB/PCIe cameras (`--webcams 0 1 2` or `front=0 left=1` syntax).

### 4.2 Controls

- `--views` or `--random_views`: choose deterministic cameras or sample a subset each run.
- `--frame_step`: process every Nth frame (useful for high-FPS videos/webcams).
- `--depth_workers`: number of Depth Anything TensorRT workers (defaults to `num_cams`).
- `--fusion`: `surfel` *(append surfels)* or `tsdf` *(integrate into TSDF grid)*.
- `--per_camera_calib`: persist scale/shift alignment to JSON for smoother future runs.

### 4.3 Example

```bash
python -m onnx.inference_pcd \
  --source videos \
  --videos data/dome_videos \
  --num_cams 8 --random_views 8 \
  --intrinsics config/dome_intrinsics.npz \
  --poses config/dome_poses.npz \
  --vggt_engine onnx_exports/torch-int8-dynamic/vggt-8x3x518x518-pcd_int8.engine \
  --depth_engine_dir onnx_exports/depth_anything \
  --frame_step 2 \
  --out_dir outputs/dome_session \
  --test_mode 1
```

Outputs include:

- `bootstrap_map.ply` and `map_updated.ply` (metric surfel clouds).
- `tsdf_volume.npz` (when `--fusion tsdf`).
- `metrics.json` (per-camera alignment residuals, scale/shift, timings).
- Optional `depth_metric/*.npz` when `--save_per_frame_depth` is enabled.

---

## 5. Depth Anything TensorRT Pool

`pcd/depth_anything.py` manages a worker pool that:

- Spins up one TensorRT execution context per thread (with dedicated CUDA contexts).
- Accepts RGB images, handles resizing, and returns float32 depth maps.
- Respects the original resolution when emitting depth (automatic resize back).
- Is resilient to FP16/BF16/INT8 engine variants.

`inference_pcd.py` uses it automatically, but you can import `DepthAnythingPool` for custom pipelines if needed.

---

## 6. Additional Tools

| Command | Description |
| --- | --- |
| `python onnx/tools/trt_inference.py --engine <engine>` | Benchmark a single TensorRT engine (real images or synthetic). |
| `python onnx/tools/benchmark_trt_engines.py --root onnx_exports` | Compare FPS/latency across all engines under `onnx_exports/`. |
| `python onnx/tools/inference_depth_anything.py --engine onnx_exports/depth_anything/depth_rtl.engine` | Dual-camera Depth Anything demo with visualization helpers. |
| `python onnx/tools/pcd_inference.py ...` | Legacy reference pipeline (kept for compatibility/testing). |
| `python onnx/tools/pcd_to_3dgs.py ...` | Seeds future 3D Gaussian Splatting experiments using exported PCDs. |

---

## 7. Automation Scripts

Located at the repo root:

- `scripts/run_all_exports.sh` — sweeps every quant mode and builds FP16/BF16/FP8/INT8 engines. It uses `onnx/tools/vggt_to_trt.py` and optionally benchmarks with `onnx/tools/benchmark_trt_engines.py`.
- `scripts/run_all_precisions.sh` — focuses on a single configuration, exporting (optional) and benchmarking via `onnx/tools/vggt_to_trt.py` and `onnx/tools/trt_inference.py`.

Both scripts honour environment variables (`NUM_CAMS`, `CALIB_DATA`, `PCD_ONLY`, etc.) and append consolidated output to `onnx/logs/out.txt`.

---

## 8. Outputs & Logging

- `onnx_exports/<quant_mode>/`: ONNX graphs (`*.onnx`), engines (`*.engine`), INT8 caches, timing caches.
- `logs/build_*.log`, `logs/infer_*.log`: per-run diagnostics.
- `logs/out.txt`: rolling summary of benchmark results (handy for comparing sweeps).
- `outputs/<session>/`: written by `inference_pcd.py` (maps, TSDF volumes, depth archives, metrics).

When engines show unexpected performance, inspect `logs/out.txt` and the per-engine logs; INT8 should currently achieve ~157 ms for VGGT PCD on RTX 5090 with PCIe storage.

---

## 9. Troubleshooting

- **Exporter fails during simplification**: rerun with `--no-simplify`, or set `FORCE_SIMPLIFY=0` when using the helper scripts; the pipeline re-attempts without simplification automatically.
- **Depth alignment unstable**: start with `--test_mode 1` to regenerate a clean bootstrap map and keep `--smooth_scale` near 0.1 for gradual updates. Store calibration JSON via `--per_camera_calib`.
- **Webcam frame drops**: reduce `--depth_workers`, increase `--frame_step`, or lower USB resolution/FPS using `v4l2-ctl` before launching the pipeline.
- **INT8 accuracy off**: ensure calibration data matches VGGT input geometry `(num_cams, 3, 518, 518)` and delete stale caches in `onnx_exports/<quant_mode>/*.cache`.

---

## 10. Next Steps

- Integrate `pcd_to_3dgs.py` once your point clouds look good; the TSDF exports already encode fused geometry.
- When experimenting with new quantisation strategies, wire them through `scripts/run_all_exports.sh` for reproducibility.
- Consider adding regression tests that exercise the new frame providers (`pcd/io_utils.py`) once you have deterministic sample data.

Happy reconstructing!

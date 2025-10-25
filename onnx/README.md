# VGGT → TensorRT Conversion Toolkit

This directory contains a self‑contained pipeline for exporting the [facebookresearch/VGGT](https://huggingface.co/facebook/VGGT-1B) model to ONNX and compiling precision‑tuned TensorRT engines. The focus is on multi‑view inputs (e.g. 8 × 518×518 images) for downstream point‑cloud reconstruction and Gaussian splatting.

The flagship script is `vggt_to_trt_chatgpt.py`, which extends the original workflow with:

- Robust INT8 calibration (image/tensor datasets, GPU/CPU staging, caching)
- Optional pre‑quantisation before ONNX export (PyTorch AoT INT8, bitsandbytes FP4/FP8/NF4, NVIDIA ModelOpt FP8)
- Precision introspection so FP8/INT8 fallback is detected automatically
- Safe opset repair + resilient ONNX simplification
- Configurable PCD‑only export that retains just depth & camera heads for point‑cloud inference

Use this document as a detailed reference for setup, configuration, and troubleshooting.

---

## 1. Repository Layout

| Path | Description |
| --- | --- |
| `vggt_to_trt_chatgpt.py` | Main conversion pipeline (this doc covers it exhaustively) |
| `vggt_to_trt.py` / `vggt_to_trt_faster.py` | Earlier conversion variants (kept for comparison) |
| `pcd_inference.py` | Example point cloud inference harness using compiled engines |
| `pcd_to_3dgs.py` | Early experiments toward 3D Gaussian Splatting (context only) |
| `trt_inference.py` | Generic TensorRT inference utility |
| `onnx_exports/` | Default output directory for exported ONNX models and TensorRT engines |
| `logs/export_logs.txt` | Log from the last conversion run (useful for regression comparisons) |

---

## 2. Prerequisites

### 2.1 Core Requirements

| Component | Minimum Version | Notes |
| --- | --- | --- |
| Python | 3.9+ | Tested with Python 3.10 |
| CUDA Toolkit | 12.2+ | Required for TensorRT 10 and ModelOpt FP8 recipes |
| TensorRT | 10.1+ | Script relies on BuilderFlag changes and new calibration APIs |
| PyTorch | 2.1+ | Must match CUDA version; inference/export uses `torch.onnx.export` |
| ONNX | 1.15+ | Reading/writing ONNX with external data |

Install base packages (modify versions to match your environment):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install onnx onnxruntime-gpu onnxsim pillow tqdm
pip install tensorrt==10.1.0.13
```

> **Tip:** TensorRT wheels often ship separately from PyPI. Use NVIDIA’s `python/tensorrt` wheel matching your CUDA driver.

### 2.2 Optional Quantisation Backends

| Mode | Extra Dependencies | When to Use |
| --- | --- | --- |
| `bitsandbytes-8bit` | `pip install bitsandbytes` | Quantise linear layers to 8‑bit (`Linear8bitLt`) before ONNX export |
| `bitsandbytes-nf4` / `bitsandbytes-fp4` | `bitsandbytes` (GPU or MPS support) | Weight‑only 4‑bit pre‑quantisation for experimentation with ultra‑low precision |
| `modelopt-fp8` | `pip install modelopt` (CUDA 12.2+) | NVIDIA’s FP8 recipe; recommended when TensorRT FP8 fallback is too aggressive |
| `torch-int8-dynamic` | `torch` with quantisation enabled | CPU‑friendly dynamic INT8 path; helpful for export stability |

---

## 3. Quick Start

```bash
python vggt_to_trt_chatgpt.py \
  --export \
  --num-cams 8 \
  --precision fp16
```

This exports `facebook/VGGT-1B` to ONNX (with external weights) and builds a FP16 TensorRT engine under `onnx_exports/`.

### 3.1 Using an Existing ONNX

```bash
python vggt_to_trt_chatgpt.py \
  --onnx-in onnx_exports/vggt-8x3x518x518.onnx \
  --precision int8 \
  --calib-data /path/to/mvs/images \
  --calib-batches 32
```

Set `--export` only when you need to regenerate the ONNX from Hugging Face.

---

## 4. Command Reference

`python vggt_to_trt_chatgpt.py [options]`

### 4.1 Input & Output

| Flag | Description |
| --- | --- |
| `--export` | Export VGGT from Hugging Face first (otherwise reuse `--onnx-in`) |
| `--onnx-in PATH` | Existing ONNX model (skips export) |
| `--output-dir DIR` | Root directory for ONNX/engine/timing cache (default: `onnx_exports`) |

### 4.2 Model Geometry

| Flag | Default | Description |
| --- | --- | --- |
| `--num-cams` | `8` | Number of camera views (batch dimension) |
| `--height` | `518` | Input height (VGGT requires 518×518) |
| `--width` | `518` | Input width |
| `--pcd-only` | off | Keep only depth + camera heads (optimised for point cloud inference) |
| `--model-name` | `facebook/VGGT-1B` | Hugging Face identifier |
| `--opset` | `18` | ONNX opset version |

### 4.3 Precision Options

| Flag | Values | Description |
| --- | --- | --- |
| `--precision` | `fp16` (default), `fp32`, `bf16`, `fp8`, `int8` | Target TensorRT precision |
| `--all-precisions` | (flag) | Build FP16 + BF16 + FP8 + INT8 sequentially |
| `--quant-mode` | see below | Apply pre‑quantisation before ONNX export |

Supported `--quant-mode` values:

- `none`: vanilla export (default)
- `torch-int8-dynamic`: PyTorch dynamic INT8 modules (CPU-centric)
- `bitsandbytes-8bit`: Replace `nn.Linear` with `Linear8bitLt`
- `bitsandbytes-nf4`: Weight‑only NF4 quantisation
- `bitsandbytes-fp4`: Weight‑only FP4 quantisation
- `modelopt-fp8`: NVIDIA ModelOpt FP8 recipe (requires CUDA + `modelopt`)

### 4.4 TensorRT Build Controls

| Flag | Default | Purpose |
| --- | --- | --- |
| `--workspace-gb` | `32` | TensorRT builder workspace (GB) |
| `--opt-level` | `5` | Builder optimisation level (0–5) |
| `--max-aux-streams` | `4` | Auxiliary CUDA streams |
| `--no-simplify` | off | Skip ONNX simplification (fallback path) |

### 4.5 INT8 Calibration

| Flag | Default | Description |
| --- | --- | --- |
| `--calib-data PATH/GLOB/NPY` | `None` | Calibration source (images or tensor file) |
| `--calib-batches` | `32` | Number of batches to feed calibrator |
| `--calib-seed` | `1337` | RNG seed for sample shuffling or synthetic data |
| `--calib-cpu` | off | Force CPU staging (otherwise GPU buffer if available) |

**Accepted sources**

- Directory with images (`.jpg`, `.png`, `.tif`, …) — recursively scanned via `Path.rglob`
- Glob pattern (e.g. `data/mvs360/*.png`)
- NumPy tensor (`.npy` or `.npz`) shaped either `(B, num_cams, 3, H, W)` or `(N, 3, H, W)`
- `None` → synthetic Gaussian noise (fallback, but weaker accuracy)

Calibration batches always match the expected input shape `(num_cams, 3, H, W)`. If the dataset is smaller than required, it is re-sampled with replacement and cached in memory.

---

## 5. Pipeline Stages

1. **Export (optional)** — Loads VGGT, applies pre‑quantisation if requested, exports to ONNX with external weights, and prunes outputs for PCD-only mode.
2. **Sequence removal** — Rewrites `SequenceAt`/`SplitToSequence` constructs into pure tensor ops (TensorRT cannot parse ONNX sequences).
3. **Simplification** — Runs `onnxsim.simplify` with conservative settings (skipped automatically if unavailable or if simplification fails).
4. **Validation** — `onnx.checker.check_model` verifies graph consistency.
5. **TensorRT build** — Parses the ONNX graph, configures builder flags per precision, loads timing cache, sets up INT8 calibration, and serialises the engine.
6. **Precision inspection** — Deserialises the engine in-memory to report actual tensor data types (helps spot FP8/INT8 fallbacks).

Intermediate artefacts:

| File | Purpose |
| --- | --- |
| `<stem>.onnx` | Raw export (kept when `--export` is used) |
| `<stem>.NOSEQ.onnx` | Sequence-free ONNX fed to simplifier |
| `<stem>.simp.onnx` | Final ONNX used for TensorRT compilation |
| `<stem>_<prec>.engine` | Serialized TensorRT engine |
| `trt_timing.cache` | Timing cache reused across builds |
| `calibration-*.cache` | INT8 calibration cache (per shape) |

The stem defaults to `vggt-{N}x3x{H}x{W}` with `-pcd` appended for PCD-only exports.

---

## 6. Working with Quantisation Modes

### 6.1 PyTorch Dynamic INT8 (`torch-int8-dynamic`)
- Runs quantisation on CPU; sets `torch.onnx.export` to CPU mode.
- Uses `torch.ao.quantization.quantize_dynamic` on Linear / LSTM / GRU layers.
- Helpful when TensorRT INT8 calibration cannot be performed (quick CPU experiments).

### 6.2 bitsandbytes Modes
- Require GPU (CUDA) or Apple MPS — enforced by runtime checks.
- Replace `nn.Linear` modules with `Linear8bitLt` or `Linear4bit`.
- Weight initialisation copies state dicts; warnings are printed if bitsandbytes rejects the load and manual weight copy is used.
- ONNX export support for these custom modules is still evolving. Use for experimentation with PyTorch inference or convert with ModelOpt if TensorRT rejects custom ops.

### 6.3 NVIDIA ModelOpt FP8 (`modelopt-fp8`)
- Leverages `modelopt.torch.quantization.quantize_model` with a FP8 recipe.
- Requires CUDA 12.2+, TensorRT 10, and `modelopt>=0.9`.
- Unlocks better FP8 utilisation when direct TensorRT compilation falls back to FP16.

---

## 7. Calibration Data Preparation

1. **Gather frames** — For INT8 accuracy, use representative multi-view image sets (MVS benchmarks, etc.).
2. **Resize** — The pipeline resizes automatically to the declared `--height/--width`, but providing native 518×518 images avoids interpolation artifacts.
3. **Organise** — Point `--calib-data` at the directory root (recursive search) or a glob (`scene*/view*.png`).
4. **Tune batches** — Set `--calib-batches` high enough to cover dataset variance (16–64 typically). Each batch pulls `num_cams` views.
5. **Reuse caches** — `calibration-<shape>.cache` is written under `--output-dir`. Delete it to refresh calibration.

When calibration fails (missing files, bitsandbytes not installed, etc.), the pipeline falls back to synthetic Gaussian noise with a warning. Examine logs carefully because engines built without calibration rarely achieve the expected INT8 speed/accuracy trade-off.

---

## 8. Monitoring Precision & Fallbacks

At the end of each build the script prints an engine precision summary similar to:

```
[INFO] Engine tensor precisions: FP16:812, FP32:6
[WARNING] FP8 requested but engine contains no FP8 tensors; TensorRT likely fell back to FP16/FP32.
```

Interpretation:
- **FP8 build shows no FP8 tensors** → Most layers lacked kernel support; switch to `--quant-mode modelopt-fp8` or try INT8.
- **INT8 build lacks INT8 tensors** → Calibration didn’t run (check for calibration warnings and ensure BuilderFlag.INT8 was set).

This quick diagnostic prevents silent fallbacks where file sizes look similar (e.g. FP8 engine same size as FP16).

---

## 9. Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `Simplification failed: model with IR version >= 3 must specify opset_import` | Opset metadata missing or onnxsim bug | Script now repairs opset, but if it persists run with `--no-simplify` |
| `Unsupported data type FP8` warnings | Expected on consumer GPUs | Ignore, but note engine may revert to FP16 (see precision summary) |
| INT8 calibrator `RuntimeError: size mismatch` | Calibration batches wrong shape | Ensure dataset produces `(num_cams, 3, H, W)` arrays (script logs shapes) |
| bitsandbytes import errors | Package not installed or backend unavailable | Install `bitsandbytes` and ensure CUDA/MPS; otherwise choose another quant mode |
| ModelOpt errors | CUDA < 12.2 or missing `modelopt` | Upgrade CUDA / install `modelopt`, or fall back to pure TensorRT FP8 |
| TensorRT parsing fails | Remaining sequence ops or unsupported node | Inspect `onnx_exports/*.NOSEQ.onnx`; ensure sequence removal succeeded |

Check `logs/export_logs.txt` for historical context when comparing runs.

---

## 10. Integrating with Point-Cloud / 3DGS Pipelines

- **PCD-only mode** (`--pcd-only`) keeps outputs relevant for depth and camera estimation, reducing engine size and speeding up inference by ~30%.
- Use `trt_inference.py` or `pcd_inference.py` as references for feeding batched multi-view tensors into TensorRT engines and constructing point clouds.
- Planned 3D Gaussian Splatting flow (`pcd_to_3dgs.py`) expects consistent depth/camera outputs from the engine; verifying matching shapes after quantisation is recommended.

---

## 11. Live PCD Inference (Videos / Webcams)

`onnx/inference_pcd.py` now supports three capture modes so you can run VGGT + Depth Anything on multi-view datasets or live camera rigs without writing glue code.

- `--source images` *(default)* consumes PKU-style frame directories. Use `--images` to point at the scene root and `--views` (comma separated) or `--random_views 8` to pick the 8 bootstrap cameras. Frames are streamed in lock-step so temporal alignment is preserved.
- `--source videos` expects one video per camera (`camXX.mp4`, or subdirectories). Pass `--videos /path/to/videos` and the script will spawn a TensorRT worker per camera, reading and fusing frames at the requested `--frame_step`.
- `--source webcams` opens multiple USB/PCIe cameras simultaneously. Provide `--webcams 0 1 2 3 4 5 6 7` (or named specs like `front=0`). A per-worker TensorRT engine keeps the Depth Anything inference threads independent.

Shared options:

- `--num_cams` sets how many viewpoints are used for VGGT initialisation and live updates (default: 8).
- `--depth_workers` overrides the number of Depth Anything TensorRT engines (defaults to `num_cams`).
- `--frame_step` skips intermediate frames for high-FPS streams; `--max_batches` bounds the live processing loop.
- `--views`/`--random_views` coexist with all source types so you can preselect a camera subset or sample different domes on each run.

Example invocations:

```bash
# PKU-style dataset with deterministic view selection
python -m onnx.inference_pcd \
  --source images \
  --images data/pku_scene \
  --views cam001,cam005,cam012,cam020,cam024,cam028,cam033,cam040 \
  --intrinsics configs/pku_intrinsics.json \
  --poses configs/pku_poses.json \
  --vggt_engine onnx_exports/none/vggt-8x3x518x518-pcd_fp16.engine \
  --out_dir outputs/pku_scene

# Video capture (picks first eight cameras automatically)
python -m onnx.inference_pcd \
  --source videos \
  --videos data/dome_videos \
  --num_cams 8 \
  --random_views 8 \
  --frame_step 2 \
  --intrinsics configs/dome_intrinsics.npz \
  --poses configs/dome_poses.npz \
  --vggt_engine onnx_exports/torch-int8-dynamic/vggt-8x3x518x518-pcd_int8.engine \
  --depth_engine_dir onnx_exports/depth_anything \
  --out_dir outputs/dome_run

# Eight live webcams on an RTX 5090
python -m onnx.inference_pcd \
  --source webcams \
  --webcams front=0 back=1 left=2 right=3 down=4 up=5 aux1=6 aux2=7 \
  --num_cams 8 \
  --intrinsics configs/webcam_intrinsics.yaml \
  --poses configs/webcam_poses.yaml \
  --initial_map outputs/bootstrap_map.ply \
  --depth_workers 8 \
  --out_dir outputs/webcam_session
```

Notes:

- Video/webcam modes require OpenCV (`pip install opencv-python`). Image mode continues to rely on Pillow.
- The Depth Anything pool spins up one TensorRT context per worker, so budget ~1.5 GB VRAM per engine.
- Per-camera scale/shift smoothing remains compatible; reuse `--per_camera_calib` across runs for faster convergence.

---

## 12. onnx_exports Directory Overview

Exports are organised by quantisation strategy so you can quickly inspect or benchmark specific precisions:

| Directory | Contents |
| --- | --- |
| `onnx_exports/none/` | Baseline VGGT PCD engines (FP32/FP16/BF16/FP8/INT8). |
| `onnx_exports/bitsandbytes-8bit/` | Weight-quantised variants with downstream FP16/BF16/FP8/INT8 TensorRT builds. |
| `onnx_exports/bitsandbytes-fp4/` | FP4 weight-only exports, matching suffix conventions (`_fp16.engine`, etc.). |
| `onnx_exports/bitsandbytes-nf4/` | NF4 weight-only exports. |
| `onnx_exports/modelopt-fp8/` | ModelOpt-assisted FP8 pipelines alongside fallback precisions. |
| `onnx_exports/torch-int8-dynamic/` | Torch dynamic INT8 pre-quantisation; best performing INT8 build (≈6.37 FPS for VGGT PCD). |
| `onnx_exports/depth_anything/` | Depth Anything v2 TensorRT engines (e.g., `depth_rtl.engine`) consumed by the live pipeline. |

Engine filenames follow `vggt-8x3x518x518-pcd_<precision>.engine` which encodes the view count and tensor layout. For quick performance references, `onnx/logs/out.txt` captures the latest benchmark sweep (INT8 currently leads at ~157 ms per VGGT pass).

---

## 13. Example Workflows

### 11.1 Multi-precision Export

```bash
python vggt_to_trt_chatgpt.py \
  --export \
  --num-cams 8 \
  --all-precisions \
  --calib-data data/mvs360 \
  --calib-batches 48
```

Builds FP16, BF16, FP8, and INT8 engines sequentially, reusing the first exported ONNX.

### 11.2 FP8 with ModelOpt + PCD Mode

```bash
python vggt_to_trt_chatgpt.py \
  --export \
  --num-cams 8 \
  --pcd-only \
  --precision fp8 \
  --quant-mode modelopt-fp8
```

Optimised for live point-cloud reconstruction when native TensorRT FP8 support is inadequate.

### 11.3 INT8 Calibration from NumPy Tensor

```bash
python vggt_to_trt_chatgpt.py \
  --onnx-in onnx_exports/vggt-8x3x518x518.onnx \
  --precision int8 \
  --calib-data calibration_batches.npy \
  --calib-batches 64 \
  --calib-cpu
```

`calibration_batches.npy` should contain either `(64, 8, 3, 518, 518)` or `(512, 3, 518, 518)` arrays.

---

## 14. Automation Script (`scripts/run_all_exports.sh`)

A helper shell script (`run_all_exports.sh`) is provided to sweep every supported quantisation mode and TensorRT precision. Move it to your preferred location (for example `scripts/run_all_exports.sh`) and make it executable:

```bash
chmod +x run_all_exports.sh
```

By default it iterates over:

- Quant modes: `none`, `torch-int8-dynamic`, `bitsandbytes-8bit`, `bitsandbytes-nf4`, `bitsandbytes-fp4`, `modelopt-fp8`
- Precisions: FP16, BF16, FP8, INT8 (via `--all-precisions`), plus a separate FP32 build

Environment variables let you customise the run without editing the script:

| Variable | Default | Description |
| --- | --- | --- |
| `PYTHON` | `python` | Python executable |
| `NUM_CAMS` / `HEIGHT` / `WIDTH` | `8 / 518 / 518` | Input shape |
| `MODEL` | `facebook/VGGT-1B` | Hugging Face model name |
| `BASE_OUTPUT` | `onnx_exports` | Root output directory (`$BASE_OUTPUT/<quant_mode>/…`) |
| `CALIB_DATA` | *(empty)* | Calibration dataset (dir/glob/.npy) |
| `CALIB_BATCHES` / `CALIB_SEED` | `32 / 1337` | Calibration controls |
| `CALIB_CPU` | `0` | Set to `1` to stage calibration batches on CPU |
| `PCD_ONLY` | `0` | Set to `1` for depth+camera exports |
| `EXTRA_ARGS` | *(empty)* | Additional CLI flags passed to every invocation |

Example usage:

```bash
CALIB_DATA=data/mvs360 \
CALIB_BATCHES=48 \
BASE_OUTPUT=onnx_sweep \
./run_all_exports.sh
```

The script stops on the first failing configuration. Check its console output and the generated subdirectories for per-mode logs and engines.

---

## 15. Maintenance Notes

- The worktree may already be dirty (tracked in `git status`). Avoid overwriting custom changes unless intentional.
- Delete `onnx_exports/*.cache` if you change calibration data or tactics.
- To re-export ONNX with a different quant mode, pass `--export --quant-mode <mode>`; the script handles device switching automatically.
- Logs are verbose by design; they surface key warnings (fallbacks, missing opset, calibration issues).

---

## 16. References

- [VGGT Paper](https://arxiv.org/abs/2311.01879)
- [facebookresearch/VGGT repository](https://github.com/facebookresearch/VGGT)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [NVIDIA ModelOpt](https://github.com/NVIDIA/modelopt)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

Happy converting! Tune calibration data, experiment with quant modes, and keep an eye on the precision summaries to ensure your engines utilise the expected numeric formats.

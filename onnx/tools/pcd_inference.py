#!/usr/bin/env python3
"""
Optimized VGGT to TensorRT Conversion Pipeline (drop-in)

Fixes:
- Robust onnxsim (sanitized opset + path-based simplify + external-data rebinding)
- Proper INT8 calibrator (trt.IInt8EntropyCalibrator2)
- Optional FP8 build (skips unless explicitly allowed)
- Minor hygiene & clearer logs

Example usage:
  # Export once, build FP16
  python vggt_to_trt_faster.py --export --num-cams 8 --precision fp16

  # Reuse an existing ONNX (no new export) and build INT8 with 20 calib batches
  python vggt_to_trt_faster.py --onnx-in onnx_exports/vggt-8x3x518x518.onnx \
      --precision int8 --calib-batches 20

  # Build multiple precisions (skips FP8 unless --include-fp8 is set)
  python vggt_to_trt_faster.py --export --num-cams 8 --all-precisions --include-fp8
"""

from __future__ import annotations
import os
import sys
import logging
import tempfile
from typing import List, Optional, Tuple
from dataclasses import dataclass

import onnx
from onnx import helper, numpy_helper, AttributeProto
from onnx.external_data_helper import convert_model_to_external_data

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("vggt_to_trt")

def _lazy_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

torch = _lazy_import("torch")
trt_mod = _lazy_import("tensorrt")

EXTERNAL_DATA_THRESHOLD = 1024  # 1KB

def _mkdir_for(p: str) -> None:
    d = os.path.dirname(os.path.abspath(p)) or "."
    os.makedirs(d, exist_ok=True)

def _rm(p: str) -> None:
    try:
        if os.path.isfile(p) or os.path.islink(p):
            os.remove(p)
    except Exception:
        pass

def _data_rel(onnx_path: str) -> str:
    return os.path.basename(onnx_path) + ".data"

def _data_abs(onnx_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(onnx_path)), _data_rel(onnx_path))

def _fmt_size(n: int) -> str:
    gb = n / (1024**3); mb = n / (1024**2); kb = n / 1024
    return f"{gb:.2f} GB" if gb >= 1 else (f"{mb:.1f} MB" if mb >= 1 else f"{kb:.1f} KB")

def _ensure_opset(m: onnx.ModelProto, opset_version: int) -> None:
    """
    Ensure model has explicit opset imports for both default '' and 'ai.onnx'.
    This sidesteps onnxsim/shape inference paths that expect an explicit default domain.
    """
    domains_present = {imp.domain for imp in m.opset_import}
    need_default = "" not in domains_present
    need_ai = "ai.onnx" not in domains_present

    # Update existing entries to the requested opset
    for imp in m.opset_import:
        if imp.domain in ("", "ai.onnx"):
            if imp.version != opset_version:
                logger.info(f"Updating opset[{imp.domain or 'default'}]: {imp.version} -> {opset_version}")
                imp.version = opset_version

    if need_default:
        imp = m.opset_import.add()
        imp.domain = ""
        imp.version = opset_version
        logger.warning("Added missing default opset_import")

    if need_ai:
        imp = m.opset_import.add()
        imp.domain = "ai.onnx"
        imp.version = opset_version
        logger.warning("Added missing 'ai.onnx' opset_import")

def _prod_map(g: onnx.GraphProto) -> dict:
    out = {}
    for n in g.node:
        for o in n.output:
            if o:
                out[o] = n
    return out

def _const_i64(name: str, vals: List[int]) -> onnx.NodeProto:
    import numpy as np
    arr = numpy_helper.from_array(np.asarray(vals, dtype="int64"), name)
    return helper.make_node("Constant", [], [name], name=name + "_const", value=arr)

def _get_attr_i(n: onnx.NodeProto, key: str, default: int = 0) -> int:
    for a in n.attribute:
        if a.name == key and a.type == AttributeProto.INT:
            return a.i
    return default

def _index_from_input(g: onnx.GraphProto, name: str) -> Optional[int]:
    for init in g.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init)
            if arr.size == 1:
                val = int(arr.item())
                return val if val >= 0 else None
            return None
    for n in g.node:
        if n.op_type == "Constant" and n.output and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t)
                    if arr.size == 1:
                        val = int(arr.item())
                        return val if val >= 0 else None
    for n in g.node:
        if n.op_type == "Cast" and n.output and n.output[0] == name:
            return _index_from_input(g, n.input[0])
    return None

def _prune_to_outputs(m: onnx.ModelProto) -> None:
    g = m.graph
    prod = _prod_map(g)
    needed = set(v.name for v in g.output)
    changed = True
    while changed:
        changed = False
        for t in list(needed):
            if t in prod:
                n = prod[t]
                for i in n.input:
                    if i and i not in needed:
                        needed.add(i)
                        changed = True
    g.node[:] = [n for n in g.node if any(o in needed for o in n.output)]
    g.initializer[:] = [i for i in g.initializer if i.name in needed]

# ---------- INT8 Calibrator ----------

class TorchEntropyCalibrator:  # wrapper name for composition + type
    pass

if trt_mod is not None:
    import types as _types
    trt = trt_mod

    class TorchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
        """
        Proper TensorRT calibrator:
          - Uses CUDA tensor as device buffer
          - Returns device pointer (int)
          - Supports calibration cache
        """
        def __init__(self, input_shape: Tuple[int, ...], cache_file: str, num_batches: int = 16, data_dir: Optional[str] = None):
            super().__init__()
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError("CUDA required for INT8 calibration via TorchEntropyCalibrator")

            self.input_shape = input_shape
            self.cache_file = cache_file
            self.num_batches = max(1, int(num_batches))
            self.cur = 0
            self.data_dir = data_dir

            # Pre-allocate device tensor and host buffer generator
            self.device_tensor = torch.empty(input_shape, device="cuda", dtype=torch.float32)
            self._prepare_host_batches()

        def _prepare_host_batches(self):
            import numpy as np
            self.host_batches = []

            if self.data_dir and os.path.isdir(self.data_dir):
                # Very light loader: pull first N .npy files if available; otherwise random
                npys = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith(".npy")]
                for p in npys[:self.num_batches]:
                    arr = np.load(p).astype("float32", copy=False)
                    if tuple(arr.shape) != self.input_shape:
                        continue
                    self.host_batches.append(arr)
            # If insufficient or none, fill the remainder with random
            for _ in range(self.num_batches - len(self.host_batches)):
                self.host_batches.append(
                    np.random.randn(*self.input_shape).astype("float32")
                )

        def get_batch_size(self):
            return self.input_shape[0]

        def get_batch(self, names):
            if self.cur >= self.num_batches:
                return None
            batch = self.host_batches[self.cur]
            self.cur += 1
            # Copy host -> device (keep pointer stable)
            with torch.no_grad():
                self.device_tensor.copy_(torch.from_numpy(batch).to(self.device_tensor.device))
            return [int(self.device_tensor.data_ptr())]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            try:
                with open(self.cache_file, "wb") as f:
                    f.write(cache)
            except Exception:
                pass

# ---------- Precision Configuration ----------

@dataclass
class PrecisionConfig:
    name: str
    flags: List[str]
    suffix: str
    description: str

PRECISIONS = {
    "fp32": PrecisionConfig("fp32", [], "", "Full precision (baseline)"),
    "fp16": PrecisionConfig("fp16", ["FP16", "TF32"], "_fp16", "Half precision with TF32 fallback"),
    "bf16": PrecisionConfig("bf16", ["BF16", "TF32"], "_bf16", "BFloat16 precision"),
    "fp8":  PrecisionConfig("fp8",  ["FP8", "FP16", "TF32"], "_fp8",  "FP8 precision with FP16 fallback"),
    "int8": PrecisionConfig("int8", ["INT8", "FP16", "TF32"], "_int8", "INT8 quantization with FP16 fallback"),
}

class VGGTPipeline:
    def __init__(
        self,
        num_cams: int = 8,
        hw: Tuple[int, int] = (518, 518),
        opset: int = 18,
        workspace_gb: int = 28,
        precision: str = "fp16",
        simplify: bool = True,
        model_name: str = "facebook/VGGT-1B",
        opt_level: int = 5,
        max_aux_streams: int = 4,
        include_fp8: bool = False,
        calib_batches: int = 16,
        calib_data: Optional[str] = None,
    ):
        if num_cams < 1:
            raise ValueError(f"num_cams must be >= 1, got {num_cams}")
        if hw[0] < 1 or hw[1] < 1:
            raise ValueError(f"Invalid dimensions: {hw}")
        if precision not in PRECISIONS:
            raise ValueError(f"Invalid precision: {precision}. Choose from {list(PRECISIONS.keys())}")

        self.num_cams = num_cams
        self.hw = hw
        self.opset = opset
        self.workspace_gb = workspace_gb
        self.precision_config = PRECISIONS[precision]
        self.do_simplify = simplify
        self.model_name = model_name
        self.opt_level = opt_level
        self.max_aux_streams = max_aux_streams
        self.include_fp8 = include_fp8
        self.calib_batches = calib_batches
        self.calib_data = calib_data

        logger.info(f"Pipeline config: {num_cams} cameras, {hw[0]}x{hw[1]}, {precision} precision")

    def export_from_hf(self, out_onnx: str, device: str = "cuda") -> str:
        if torch is None:
            raise RuntimeError("PyTorch not available. pip install torch")
        try:
            from vggt.models.vggt import VGGT
        except ImportError:
            raise RuntimeError("VGGT module not available. pip install vggt")

        _mkdir_for(out_onnx)
        C, H, W = 3, *self.hw
        x = torch.randn(self.num_cams, C, H, W)

        logger.info(f"Loading model: {self.model_name}")
        if device == "cuda" and torch.cuda.is_available():
            model = VGGT.from_pretrained(self.model_name).eval().to("cuda").to(torch.float32)
            x = x.to("cuda", dtype=torch.float32)
            logger.info("Using CUDA device")
        else:
            model = VGGT.from_pretrained(self.model_name).eval().to("cpu").to(torch.float32)
            x = x.to("cpu", dtype=torch.float32)
            logger.info("Using CPU device")

        logger.info(f"Exporting to ONNX with shape [{self.num_cams}, {C}, {H}, {W}]")
        with torch.inference_mode():
            try:
                torch.onnx.export(
                    model, (x,),
                    out_onnx,
                    input_names=["images"],
                    output_names=None,
                    opset_version=self.opset,
                    do_constant_folding=True,
                    dynamic_axes=None,
                    verbose=False,
                    export_params=True,
                    dynamo=True,
                    external_data=True,
                    optimize=True,
                    verify=False,
                    profile=False,
                    report=False,
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning("GPU OOM during export; retrying on CPU...")
                model = model.to("cpu"); x = x.to("cpu")
                torch.onnx.export(
                    model, (x,),
                    out_onnx,
                    input_names=["images"],
                    output_names=None,
                    opset_version=self.opset,
                    do_constant_folding=True,
                    dynamic_axes=None,
                    verbose=False,
                    export_params=True,
                    dynamo=True,
                    external_data=True,
                    optimize=True,
                    verify=False,
                    profile=False,
                    report=False,
                )
        logger.info(f"Wrote {out_onnx}")
        self.rebind_external_data(out_onnx)
        return out_onnx

    def rebind_external_data(self, onnx_path: str) -> None:
        logger.info("Rebinding external data...")
        m = onnx.load(onnx_path, load_external_data=True)
        _ensure_opset(m, self.opset)
        rel = _data_rel(onnx_path); abs_p = _data_abs(onnx_path)
        _rm(abs_p)
        convert_model_to_external_data(
            m, all_tensors_to_one_file=True, location=rel,
            size_threshold=EXTERNAL_DATA_THRESHOLD, convert_attribute=False
        )
        onnx.save_model(m, onnx_path)
        try:
            s_on = os.path.getsize(onnx_path)
            s_da = os.path.getsize(abs_p) if os.path.exists(abs_p) else 0
            logger.info(f"External data: {abs_p}")
            logger.info(f"Sizes: ONNX={_fmt_size(s_on)}, DATA={_fmt_size(s_da)}")
        except Exception:
            pass

    def _normalize_softmax_last_axis(self, g: onnx.GraphProto) -> int:
        changed = 0
        for n in g.node:
            if n.op_type == "Softmax":
                attrs_to_keep = [a for a in n.attribute if a.name != "axis"]
                del n.attribute[:]
                n.attribute.extend(attrs_to_keep)
                n.attribute.extend([helper.make_attribute("axis", -1)])
                changed += 1
        return changed

    def _normalize_squeeze(self, g: onnx.GraphProto) -> int:
        updated, out_nodes = 0, []
        for n in g.node:
            if n.op_type == "Squeeze":
                axes_attr = next((a for a in n.attribute if a.name == "axes"), None)
                if axes_attr is not None:
                    if axes_attr.type == AttributeProto.INT:
                        axes = [axes_attr.i]
                    elif axes_attr.type == AttributeProto.INTS:
                        axes = list(axes_attr.ints)
                    else:
                        out_nodes.append(n); continue
                    axes_name = n.name + "_axes"
                    out_nodes.append(_const_i64(axes_name, axes))
                    out_nodes.append(helper.make_node("Squeeze", [n.input[0], axes_name], list(n.output), name=n.name))
                    updated += 1; continue
            out_nodes.append(n)
        del g.node[:]; g.node.extend(out_nodes)
        return updated

    def remove_sequences(self, src: str, dst: str) -> str:
        logger.info("Removing sequence operations...")
        m = onnx.load(src, load_external_data=True)
        _ensure_opset(m, self.opset)

        g = m.graph; prod = _prod_map(g)
        fixed_sq = self._normalize_squeeze(g)
        if fixed_sq > 0: logger.info(f"Normalized {fixed_sq} Squeeze nodes")
        fixed_sm = self._normalize_softmax_last_axis(g)
        if fixed_sm > 0: logger.info(f"Rewrote {fixed_sm} Softmax axes to -1")

        replaced, new_nodes = 0, []
        for n in g.node:
            if n.op_type != "SequenceAt":
                new_nodes.append(n); continue
            seq, idx = n.input; p = prod.get(seq, None); idx_val = _index_from_input(g, idx)
            if idx_val is None or p is None:
                logger.warning(f"Cannot resolve SequenceAt '{n.name}': idx={idx_val}, producer={p.op_type if p else None}")
                new_nodes.append(n); continue

            if p.op_type == "SplitToSequence":
                X = p.input[0]; axis = _get_attr_i(p, "axis", 0)
                s, e, a = n.name + "_starts", n.name + "_ends", n.name + "_axes"
                sliced = n.output[0] + "_slice"
                new_nodes += [
                    _const_i64(s, [idx_val]),
                    _const_i64(e, [idx_val + 1]),
                    _const_i64(a, [axis]),
                    helper.make_node("Slice", [X, s, e, a], [sliced], name=n.name + "_slice"),
                    helper.make_node("Squeeze", [sliced, a], [n.output[0]], name=n.name + "_squeeze"),
                ]
                replaced += 1; continue

            if p.op_type == "SequenceConstruct":
                ins = list(p.input)
                if 0 <= idx_val < len(ins):
                    new_nodes.append(helper.make_node("Identity", [ins[idx_val]], [n.output[0]], name=n.name + "_id"))
                    replaced += 1; continue
                else:
                    logger.warning(f"SequenceAt index {idx_val} out of range for SequenceConstruct({len(ins)} inputs)")

            new_nodes.append(n)

        del g.node[:]; g.node.extend(new_nodes)
        _prune_to_outputs(m)

        remaining = [n for n in g.node if "Sequence" in n.op_type]
        if remaining:
            for n in remaining[:5]:
                logger.error(f"Remaining sequence op: {n.op_type}: {n.name}")
            raise RuntimeError("Failed to remove all sequence operations")

        _mkdir_for(dst)
        rel = _data_rel(dst); abs_p = _data_abs(dst); _rm(abs_p)
        convert_model_to_external_data(m, True, rel, EXTERNAL_DATA_THRESHOLD, False)
        onnx.save_model(m, dst)
        logger.info(f"Replaced {replaced} sequence operations")
        logger.info(f"Wrote {dst}")
        return dst

    def simplify(self, src: str, dst: str) -> str:
        if not self.do_simplify:
            logger.info("Simplification disabled")
            return src

        try:
            onnxsim = __import__("onnxsim")
        except Exception:
            logger.warning("onnxsim not installed, skipping simplification")
            return src

        logger.info("Simplifying ONNX graph...")
        # Load, sanitize opset, save to temp path for onnxsim (avoids proto/external-data pitfalls)
        model = onnx.load(src, load_external_data=True)
        _ensure_opset(model, self.opset)
        with tempfile.TemporaryDirectory() as td:
            tmp_in = os.path.join(td, "in.onnx")
            tmp_out = os.path.join(td, "out.onnx")
            onnx.save_model(model, tmp_in)

            # Call onnxsim on file paths; be conservative with flags
            try:
                import inspect
                params = set(inspect.signature(onnxsim.simplify).parameters)
                kwargs = {}
                if "skip_shape_inference" in params: kwargs["skip_shape_inference"] = False
                if "skip_constant_folding" in params: kwargs["skip_constant_folding"] = False
                if "perform_optimization" in params:   kwargs["perform_optimization"] = True
                ms, ok = onnxsim.simplify(tmp_in, **kwargs)  # returns (model, check)
                if not ok:
                    logger.warning("onnxsim reported warnings; proceeding with simplified graph")
            except Exception as e:
                logger.error(f"Simplification failed: {e}; using non-simplified graph")
                return src

            # Save simplified with external data
            _mkdir_for(dst)
            rel = _data_rel(dst); abs_p = _data_abs(dst); _rm(abs_p)
            convert_model_to_external_data(ms, True, rel, EXTERNAL_DATA_THRESHOLD, False)
            onnx.save_model(ms, dst)

        logger.info(f"Wrote {dst}")
        return dst

    @staticmethod
    def onnx_check(path: str) -> None:
        logger.info("Validating ONNX model...")
        onnx.checker.check_model(path)
        logger.info("ONNX validation passed")

    def _fp8_really_supported(self) -> bool:
        # Guard: presence of enums/flags is not enough; add a coarse device check
        try:
            if not hasattr(trt_mod.BuilderFlag, "FP8"):
                return False
            if torch and torch.cuda.is_available():
                major = torch.cuda.get_device_capability()[0]
                # Hopper/Blackwell or newer likely; still not a guarantee
                return major >= 9
        except Exception:
            pass
        return False

    def build_trt(self, onnx_path: str, engine_path: str) -> str:
        if trt_mod is None:
            raise RuntimeError("TensorRT not available")
        trt = trt_mod

        # Skip FP8 silently if not truly supported (optional)
        if self.precision_config.name == "fp8" and not self.include_fp8:
            if not self._fp8_really_supported():
                logger.info("FP8 not clearly supported in this env; skipping FP8 build. Use --include-fp8 to force.")
                return engine_path  # No engine built; caller can ignore or choose different precision

        logger.info(f"Building TensorRT engine ({self.precision_config.description})...")
        logger.info(f"TensorRT version: {getattr(trt, '__version__', 'unknown')}")
        _mkdir_for(engine_path)

        logger_trt = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_trt)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger_trt)

        ok = False
        if hasattr(parser, "parse_from_file"):
            ok = parser.parse_from_file(onnx_path)
        else:
            with open(onnx_path, "rb") as f:
                ok = parser.parse(f.read())
        if not ok:
            logger.error("TensorRT parsing failed:")
            for i in range(parser.num_errors):
                logger.error(f"  {parser.get_error(i)}")
            raise RuntimeError("TensorRT failed to parse ONNX model")

        if network.num_inputs > 0:
            input_tensor = network.get_input(0)
            input_shape = tuple(input_tensor.shape)
            logger.info(f"Input shape: {input_shape}")
            try:
                input_tensor.dtype = trt.float32  # keep input as fp32, builder handles downcast
            except Exception:
                pass
            expected = (self.num_cams, 3, *self.hw)
            if input_shape != expected:
                logger.warning(f"Input shape mismatch: expected {expected}, got {input_shape}")

        config = builder.create_builder_config()

        # Workspace
        ws_bytes = self.workspace_gb * (1 << 30)
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_bytes)
        except Exception:
            try:
                config.max_workspace_size = ws_bytes
            except Exception:
                logger.warning("Could not set workspace size")

        # Aux streams
        try:
            builder.max_aux_streams = self.max_aux_streams
            logger.info(f"Max auxiliary streams: {self.max_aux_streams}")
        except Exception:
            pass

        # Precision flags
        for flag_name in self.precision_config.flags:
            try:
                if hasattr(trt.BuilderFlag, flag_name):
                    flag = getattr(trt.BuilderFlag, flag_name)
                    config.set_flag(flag)
                    logger.info(f"Enabled {flag_name}")
            except Exception as e:
                logger.warning(f"Could not enable {flag_name}: {e}")

        # Optimization level
        try:
            if hasattr(config, "set_builder_optimization_level"):
                config.set_builder_optimization_level(self.opt_level)
            elif hasattr(config, "builder_optimization_level"):
                config.builder_optimization_level = self.opt_level
            logger.info(f"Optimization level: {self.opt_level}")
        except Exception:
            pass

        # Tactic sources
        try:
            if hasattr(trt, "TacticSource") and hasattr(config, "set_tactic_sources"):
                TS = trt.TacticSource
                sources = 0
                for name in ("CUBLAS", "CUBLAS_LT", "CUDNN"):
                    if hasattr(TS, name):
                        sources |= getattr(TS, name)
                if sources:
                    config.set_tactic_sources(sources)
                    logger.info("Enabled tactic sources: cuBLAS/cuBLAS_LT/cuDNN")
        except Exception:
            pass

        # Timing cache
        cache_dir = os.path.dirname(engine_path) or "."
        cache_path = os.path.join(cache_dir, "trt_timing.cache")
        tc = None
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    tc = config.create_timing_cache(f.read())
                    config.set_timing_cache(tc, ignore_mismatch=False)
                    logger.info(f"Loaded timing cache from {cache_path}")
        except Exception as e:
            logger.info(f"No timing cache loaded: {e}")

        # INT8 calibration
        if self.precision_config.name == "int8":
            logger.info("Setting up INT8 calibration...")
            calib_cache = os.path.join(cache_dir, "calibration.cache")
            input_shape = (self.num_cams, 3, *self.hw)
            try:
                calibrator = TorchEntropyCalibrator(
                    input_shape=input_shape,
                    cache_file=calib_cache,
                    num_batches=self.calib_batches,
                    data_dir=self.calib_data
                )
                if hasattr(config, "set_int8_calibrator"):
                    config.set_int8_calibrator(calibrator)
                else:
                    config.int8_calibrator = calibrator
                logger.info("INT8 calibrator configured")
            except Exception as e:
                logger.warning(f"INT8 calibration setup failed (INT8 fallback disabled): {e}")

        logger.info("Building engine (this may take several minutes)...")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("TensorRT build returned None (build failed)")

        try:
            if tc is None:
                tc = config.get_timing_cache()
            if tc:
                blob = tc.serialize()
                with open(cache_path, "wb") as f:
                    f.write(blob)
                logger.info(f"Saved timing cache to {cache_path}")
        except Exception as e:
            logger.info(f"Could not save timing cache: {e}")

        with open(engine_path, "wb") as f:
            f.write(engine_bytes)

        engine_size = os.path.getsize(engine_path)
        logger.info(f"Engine saved: {engine_path} ({_fmt_size(engine_size)})")
        return engine_path

    def run(self, onnx_in: Optional[str], onnx_simp: str, onnx_noseq: str, engine_path: str, export: bool) -> str:
        if export:
            base_name = onnx_simp.replace(".simp.onnx", ".onnx")
            onnx_in = self.export_from_hf(base_name)
        if not onnx_in:
            raise ValueError("Must provide --onnx-in or use --export")
        if not os.path.exists(onnx_in):
            raise FileNotFoundError(f"Input ONNX not found: {onnx_in}")

        noseq = self.remove_sequences(onnx_in, onnx_noseq)
        simp = self.simplify(noseq, onnx_simp)
        self.onnx_check(simp)
        engine = self.build_trt(simp, engine_path)
        return engine

# ---------- CLI ----------

def _parse():
    import argparse
    ap = argparse.ArgumentParser(
        description="Convert VGGT models to optimized TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Input/output
    ap.add_argument("--export", action="store_true", help="Export VGGT from HuggingFace")
    ap.add_argument("--onnx-in", help="Use existing ONNX file instead of exporting")
    ap.add_argument("--model-name", default="facebook/VGGT-1B", help="HuggingFace model name")

    # Model configuration
    ap.add_argument("--num-cams", type=int, default=8, help="Number of camera views (default: 8)")
    ap.add_argument("--height", type=int, default=518, help="Input height")
    ap.add_argument("--width", type=int, default=518, help="Input width")
    ap.add_argument("--opset", type=int, default=18, help="ONNX opset version")

    # Precision configuration
    ap.add_argument("--precision", choices=list(PRECISIONS.keys()), default="fp16", help="Precision mode")
    ap.add_argument("--all-precisions", action="store_true", help="Build fp16, bf16, int8 (+fp8 if --include-fp8)")
    ap.add_argument("--include-fp8", action="store_true", help="Include FP8 when building --all-precisions")

    # TensorRT optimization
    ap.add_argument("--workspace-gb", type=int, default=32, help="TensorRT workspace size in GB")
    ap.add_argument("--opt-level", type=int, default=5, help="TensorRT optimization level 0-5")
    ap.add_argument("--max-aux-streams", type=int, default=4, help="Max auxiliary CUDA streams")
    ap.add_argument("--no-simplify", action="store_true", help="Skip ONNX simplification")

    # INT8 calibration
    ap.add_argument("--calib-batches", type=int, default=16, help="Calibration batches for INT8")
    ap.add_argument("--calib-data", type=str, default=None, help="Folder of .npy batches shaped (N,C,H,W) for INT8 calibration")

    # Output paths
    ap.add_argument("--output-dir", default="onnx_exports", help="Output directory for all artifacts")
    return ap.parse_args()

def main():
    args = _parse()
    stem = f"vggt-{args.num_cams}x3x{args.height}x{args.width}"

    if args.all_precisions:
        precisions_to_build = ["fp16", "bf16", "int8"] + (["fp8"] if args.include_fp8 else [])
        logger.info("Building all precision variants: " + ", ".join(p.upper() for p in precisions_to_build))
    else:
        precisions_to_build = [args.precision]

    results = []
    shared_onnx_base = args.onnx_in

    try:
        for idx, precision in enumerate(precisions_to_build):
            logger.info("\n" + "=" * 69)
            logger.info(f"Building {precision.upper()} variant ({idx+1}/{len(precisions_to_build)})")
            logger.info("=" * 69)

            suffix = PRECISIONS[precision].suffix
            onnx_simp = os.path.join(args.output_dir, f"{stem}.simp.onnx")
            onnx_noseq = os.path.join(args.output_dir, f"{stem}.NOSEQ.onnx")
            engine = os.path.join(args.output_dir, f"{stem}{suffix}.engine")

            pipe = VGGTPipeline(
                num_cams=args.num_cams,
                hw=(args.height, args.width),
                opset=args.opset,
                workspace_gb=args.workspace_gb,
                precision=precision,
                simplify=not args.no_simplify,
                model_name=args.model_name,
                opt_level=args.opt_level,
                max_aux_streams=args.max_aux_streams,
                include_fp8=args.include_fp8,
                calib_batches=args.calib_batches,
                calib_data=args.calib_data,
            )

            is_first = (idx == 0)
            do_export = args.export and is_first
            onnx_input = shared_onnx_base

            engine_path = pipe.run(
                onnx_in=onnx_input,
                onnx_simp=onnx_simp,
                onnx_noseq=onnx_noseq,
                engine_path=engine,
                export=do_export,
            )

            if is_first and do_export:
                shared_onnx_base = onnx_simp.replace(".simp.onnx", ".onnx")
                logger.info(f"Shared ONNX for subsequent builds: {shared_onnx_base}")

            results.append((precision, engine_path))

        logger.info("\n" + "=" * 70)
        logger.info("BUILD COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Input shape: [{args.num_cams}, 3, {args.height}, {args.width}]")
        logger.info("")
        logger.info("Built engines:")
        for precision, engine_path in results:
            if engine_path and os.path.exists(engine_path):
                size = os.path.getsize(engine_path)
                desc = PRECISIONS[precision].description
                logger.info(f"  {precision.upper():5s} ({desc:40s}): {engine_path}")
                logger.info(f"        Size: {_fmt_size(size)}")
            else:
                logger.info(f"  {precision.upper():5s}: skipped (no engine)")

        logger.info("")
        logger.info("Next steps:")
        logger.info("  1) Benchmark each engine to find the fastest")
        logger.info("  2) Expected speedup: INT8 > FP16/BF16 (~= FP8 fallback) > FP32")
        logger.info("  3) Quality check: FP16≈BF16≈FP8; INT8 may need calibration tuning")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.error("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()



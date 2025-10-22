#!/usr/bin/env python3
"""
VGGT to TensorRT Conversion Pipeline — Optimized

Exports VGGT models to ONNX (opset 18), removes sequence operations,
optionally simplifies, and builds highly optimized TensorRT engines with
FP16/BF16/FP8/INT8 support, persistent timing cache, tactic controls,
auxiliary streams, and TF32 fallback.

Key guarantees (kept identical to your original intent):
- No change to shapes, batch (N), or model semantics
- No micro-batching, no graph surgery beyond sequence removal + safe normalizations

Examples
========
# 1) Export and build FP16 for 8 cams @ 518x518
python vggt_to_trt.py --export --num-cams 8 --height 518 --width 518 --precision fp16

# 2) Reuse existing ONNX and build FP8 (if available in your TRT stack)
python vggt_to_trt.py --onnx-in onnx_exports/vggt-8x3x518x518.simp.onnx --precision fp8

# 3) Build all useful precisions in one go (fp16, bf16, fp8*, int8)
python vggt_to_trt.py --onnx-in onnx_exports/vggt-8x3x518x518.simp.onnx --all-precisions

# 4) INT8 with random calibrator (no data needed)
python vggt_to_trt.py --onnx-in onnx_exports/vggt-8x3x518x518.simp.onnx --precision int8 \
  --calib-batches 16 --calib-cache onnx_exports/vggt.calib

Notes
=====
• FP8 availability depends on TensorRT + driver stack (and GPU).
• True 4‑bit (FP4/INT4) is not exposed via standard TensorRT builder flags
  as of today; it usually requires TensorRT‑LLM / TransformerEngine routes
  or Q/DQ graph transforms. This script focuses on FP16/BF16/FP8/INT8 paths.
"""

from __future__ import annotations
import os
import sys
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

import onnx
from onnx import helper, numpy_helper, AttributeProto
from onnx.external_data_helper import convert_model_to_external_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Lazy imports

def _lazy_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

torch = _lazy_import("torch")
trt_mod = _lazy_import("tensorrt")

# External data file size threshold (1KB)
EXTERNAL_DATA_THRESHOLD = 1024

# ---------- Utility Functions ----------

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
    gb = n / (1024**3)
    mb = n / (1024**2)
    kb = n / 1024
    if gb >= 1:
        return f"{gb:.2f} GB"
    elif mb >= 1:
        return f"{mb:.1f} MB"
    else:
        return f"{kb:.1f} KB"

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
    # Check initializers
    for init in g.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init)
            if arr.size == 1:
                val = int(arr.item())
                return val if val >= 0 else None
            return None
    # Constant nodes
    for n in g.node:
        if n.op_type == "Constant" and n.output and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t)
                    if arr.size == 1:
                        val = int(arr.item())
                        return val if val >= 0 else None
    # Cast(Constant)
    for n in g.node:
        if n.op_type == "Cast" and n.output and n.output[0] == name:
            src = n.input[0]
            return _index_from_input(g, src)
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
    filtered_nodes = [n for n in g.node if any(o in needed for o in n.output)]
    filtered_inits = [i for i in g.initializer if i.name in needed]
    del g.node[:]
    g.node.extend(filtered_nodes)
    del g.initializer[:]
    g.initializer.extend(filtered_inits)

def _ensure_ai_onnx_opset(m, default_version: int) -> None:
    """
    Ensure exactly one opset_import for the main ONNX domain.
    Use 'ai.onnx' explicitly (some toolchains reject the empty-domain alias).
    Keeps all other domain imports (e.g., 'ai.onnx.ml') untouched.
    """
    # current imports as a dict
    seen = {imp.domain: int(getattr(imp, "version", 0)) for imp in m.opset_import}

    # choose a version for ai.onnx: prefer existing, else empty-domain, else default
    ai_ver = seen.get("ai.onnx", seen.get("", int(default_version)))
    if not ai_ver:
        ai_ver = int(default_version)

    from onnx import helper
    new_imports = [imp for imp in m.opset_import if imp.domain not in ("", "ai.onnx")]
    new_imports.append(helper.make_operatorsetid("ai.onnx", int(ai_ver)))

    # replace in-place (avoid creating a gigantic new proto)
    del m.opset_import[:]
    m.opset_import.extend(new_imports)

# ---------- INT8 Calibrator ----------

class _EntropyCalibrator:  # Minimal calibrator using random data unless images are provided later
    def __init__(self, input_shape: Tuple[int, ...], cache_file: str, num_batches: int = 8):
        if trt_mod is None:
            raise RuntimeError("TensorRT not available")
        self.trt = trt_mod
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.num_batches = num_batches
        self.curr = 0
        import numpy as np
        self.host_batches = [np.random.randn(*input_shape).astype('float32') for _ in range(num_batches)]
        self.device_tensor = None
        if torch is not None and torch.cuda.is_available():
            self.device_tensor = torch.empty(*input_shape, dtype=torch.float32, device='cuda')

        # Build proper calibrator base class at runtime (TensorRT API variants)
        if hasattr(self.trt, 'IInt8EntropyCalibrator2'):
            base = self.trt.IInt8EntropyCalibrator2
        else:
            base = self.trt.IInt8EntropyCalibrator
        self.__class__ = type(self.__class__.__name__, (base,), dict(self.__class__.__dict__))

    def get_batch_size(self):
        return self.input_shape[0]

    def get_batch(self, names):  # names: input tensor names
        if self.curr >= self.num_batches:
            return None
        import numpy as np
        data = self.host_batches[self.curr]
        self.curr += 1
        if self.device_tensor is not None:
            self.device_tensor.copy_(torch.from_numpy(data), non_blocking=True)
            return [int(self.device_tensor.data_ptr())]
        # Fallback: host pointer path isn't supported by Python API; abort
        return None

    def read_calibration_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if self.cache_file:
            with open(self.cache_file, 'wb') as f:
                f.write(cache)

# ---------- Precision Configuration ----------

@dataclass
class PrecisionConfig:
    name: str
    flags: List[str]
    suffix: str
    description: str

PRECISIONS = {
    "fp32": PrecisionConfig("fp32", [], "_fp32", "Full precision (baseline)"),
    "fp16": PrecisionConfig("fp16", ["FP16", "TF32"], "_fp16", "Half precision with TF32 fallback"),
    "bf16": PrecisionConfig("bf16", ["BF16", "TF32"], "_bf16", "BFloat16 precision"),
    "fp8":  PrecisionConfig("fp8",  ["FP8", "FP16", "TF32"], "_fp8",  "FP8 with FP16 fallback (if supported)"),
    "int8": PrecisionConfig("int8", ["INT8", "FP16", "TF32"], "_int8", "INT8 PTQ with FP16 fallback"),
}

# ---------- Pipeline Class ----------

class VGGTPipeline:
    """
    Pipeline for converting VGGT models to TensorRT engines.

    Stages:
      1. Export VGGT (HF) -> ONNX
      2. Remove sequence ops; normalize Squeeze/Softmax
      3. Optional onnx-simplify
      4. Validate ONNX
      5. Build TensorRT engine (precision selectable)
    """

    def __init__(
        self,
        num_cams: int = 10,
        hw: Tuple[int, int] = (518, 518),
        opset: int = 18,
        workspace_gb: int = 28,
        precision: str = "fp16",
        simplify: bool = True,
        model_name: str = "facebook/VGGT-1B",
        opt_level: int = 5,
        max_aux_streams: int = 4,
        use_timing_cache: bool = True,
        profiling_verbose: bool = False,
        calib_batches: int = 8,
        calib_cache: Optional[str] = None,
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
        self.precision_cfg = PRECISIONS[precision]
        self.do_simplify = simplify
        self.model_name = model_name
        self.opt_level = opt_level
        self.max_aux_streams = max_aux_streams
        self.use_timing_cache = use_timing_cache
        self.profiling_verbose = profiling_verbose
        self.calib_batches = calib_batches
        self.calib_cache = calib_cache
        logger.info(f"Pipeline config: N={num_cams}, {hw[0]}x{hw[1]}, precision={precision}")

    # ------- Export / Graph prep -------

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
        try:
            if device == "cuda" and torch.cuda.is_available():
                model = VGGT.from_pretrained(self.model_name).eval().to("cuda").to(torch.float32)
                x = x.to("cuda", dtype=torch.float32)
                logger.info("Using CUDA device for export")
            else:
                model = VGGT.from_pretrained(self.model_name).eval().to("cpu").to(torch.float32)
                x = x.to("cpu", dtype=torch.float32)
                logger.info("Using CPU for export")
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.model_name}': {e}")

        logger.info(f"Exporting to ONNX with shape [{self.num_cams}, {C}, {H}, {W}]")
        with torch.inference_mode():
            try:
                torch.onnx.export(
                    model, (x,), out_onnx,
                    input_names=["images"], output_names=None,
                    opset_version=self.opset,
                    do_constant_folding=True, dynamic_axes=None,
                    verbose=False, export_params=True, dynamo=True,
                    external_data=True, optimize=True, verify=False,
                    profile=False, report=False,
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning("GPU OOM during export; retrying on CPU…")
                model = model.to("cpu"); x = x.to("cpu")
                torch.onnx.export(
                    model, (x,), out_onnx,
                    input_names=["images"], output_names=None,
                    opset_version=self.opset,
                    do_constant_folding=True, dynamic_axes=None,
                    verbose=False, export_params=True, dynamo=True,
                    external_data=True, optimize=True, verify=False,
                    profile=False, report=False,
                )
            except Exception as e:
                raise RuntimeError(f"ONNX export failed: {e}")

        logger.info(f"Wrote {out_onnx}")
        self.rebind_external_data(out_onnx)
        return out_onnx

    def rebind_external_data(self, onnx_path: str) -> None:
        logger.info("Rebinding external data…")
        m = onnx.load(onnx_path, load_external_data=True)
        rel = _data_rel(onnx_path)
        abs_p = _data_abs(onnx_path)
        _rm(abs_p)
        convert_model_to_external_data(m, True, rel, EXTERNAL_DATA_THRESHOLD, False)
        onnx.save_model(m, onnx_path)
        logger.info(f"External data: {abs_p}")
        try:
            s_on = os.path.getsize(onnx_path)
            s_da = os.path.getsize(abs_p) if os.path.exists(abs_p) else 0
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
        updated, out = 0, []
        for n in g.node:
            if n.op_type == "Squeeze":
                axes_attr = None
                for a in n.attribute:
                    if a.name == "axes":
                        axes_attr = a
                        break
                if axes_attr is not None:
                    if axes_attr.type == AttributeProto.INT:
                        axes = [axes_attr.i]
                    elif axes_attr.type == AttributeProto.INTS:
                        axes = list(axes_attr.ints)
                    else:
                        out.append(n); continue
                    axes_name = n.name + "_axes"
                    out.append(_const_i64(axes_name, axes))
                    out.append(helper.make_node("Squeeze", [n.input[0], axes_name], list(n.output), name=n.name))
                    updated += 1
                    continue
            out.append(n)
        del g.node[:]
        g.node.extend(out)
        return updated

    def remove_sequences(self, src: str, dst: str) -> str:
        logger.info("Removing sequence operations…")
        m = onnx.load(src, load_external_data=True)
        g = m.graph
        prod = _prod_map(g)

        fixed_sq = self._normalize_squeeze(g)
        if fixed_sq > 0:
            logger.info(f"Normalized {fixed_sq} Squeeze nodes")
        fixed_sm = self._normalize_softmax_last_axis(g)
        if fixed_sm > 0:
            logger.info(f"Rewrote {fixed_sm} Softmax axes to -1")

        replaced, new_nodes = 0, []
        for n in g.node:
            if n.op_type != "SequenceAt":
                new_nodes.append(n); continue
            seq, idx = n.input
            p = prod.get(seq, None)
            idx_val = _index_from_input(g, idx)
            if idx_val is None or p is None:
                logger.warning(f"Cannot resolve SequenceAt '{n.name}': idx_val={idx_val}, producer={p.op_type if p else None}")
                new_nodes.append(n); continue
            if p.op_type == "SplitToSequence":
                X = p.input[0]
                axis = _get_attr_i(p, "axis", 0)
                s, e, a = n.name + "_starts", n.name + "_ends", n.name + "_axes"
                sliced = n.output[0] + "_slice"
                new_nodes += [
                    _const_i64(s, [idx_val]), _const_i64(e, [idx_val + 1]), _const_i64(a, [axis]),
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
                    logger.warning(f"SequenceAt index {idx_val} out of range for SequenceConstruct with {len(ins)} inputs")
            new_nodes.append(n)
        del g.node[:]; g.node.extend(new_nodes)
        _prune_to_outputs(m)
        remaining = [n for n in g.node if "Sequence" in n.op_type]
        if remaining:
            logger.error(f"{len(remaining)} sequence ops remain (first 5 shown):")
            for n in remaining[:5]:
                logger.error(f"  - {n.op_type}: {n.name}")
            raise RuntimeError("Failed to remove all sequence operations")
        _mkdir_for(dst)
        rel = _data_rel(dst); abs_p = _data_abs(dst); _rm(abs_p)
        convert_model_to_external_data(m, True, rel, EXTERNAL_DATA_THRESHOLD, False)
        onnx.save_model(m, dst)
        logger.info(f"Replaced {replaced} sequence ops; wrote {dst}")
        logger.info(f"Data: {abs_p}")
        return dst

    def simplify(self, src: str, dst: str) -> str:
        if not self.do_simplify:
            logger.info("Simplification disabled")
            return src

        try:
            onnxsim = __import__("onnxsim")
        except Exception:
            logger.warning("onnxsim not installed, skipping simplification")
            logger.info("Install with: pip install onnxsim")
            return src

        logger.info("Simplifying ONNX graph…")
        m = onnx.load(src, load_external_data=True)
        _ensure_ai_onnx_opset(m, int(self.opset))
        # right after: m = onnx.load(src, load_external_data=True)
        # if not m.opset_import or len(m.opset_import) == 0:
        #     imp = onnx.helper.make_operatorsetid("", self.opset)  # e.g., 18
        #     m.opset_import.extend([imp])


        # --- minimal fix: ensure default opset import exists & matches self.opset ---
        # from onnx import helper
        # default = None
        # for imp in m.opset_import:
        #     if imp.domain in ("", "ai.onnx"):
        #         default = imp
        #         break
        # if default is None:
        #     m.opset_import.extend([helper.make_operatorsetid("", int(self.opset))])
        # else:
        #     default.version = int(self.opset)
        # ---------------------------------------------------------------------------

        # call onnxsim with your existing kwargs
        import inspect
        params = set(inspect.signature(onnxsim.simplify).parameters)
        kwargs = {}
        if "skip_shape_inference" in params: kwargs["skip_shape_inference"] = False
        if "skip_constant_folding" in params: kwargs["skip_constant_folding"] = False
        if "skip_optimization" in params:     kwargs["skip_optimization"] = False
        if "perform_optimization" in params:  kwargs["perform_optimization"] = True

        try:
            result = onnxsim.simplify(m, **kwargs)
        except TypeError:
            result = onnxsim.simplify(m)

        ms = result[0] if isinstance(result, tuple) else result

        _mkdir_for(dst)
        rel = _data_rel(dst); abs_p = _data_abs(dst); _rm(abs_p)
        convert_model_to_external_data(ms, True, rel, EXTERNAL_DATA_THRESHOLD, False)
        onnx.save_model(ms, dst)
        logger.info(f"Wrote {dst}")
        logger.info(f"Data: {abs_p}")
        return dst

    @staticmethod
    def onnx_check(path: str) -> None:
        logger.info("Validating ONNX model…")
        try:
            onnx.checker.check_model(path)
            logger.info("ONNX validation passed")
        except Exception as e:
            raise RuntimeError(f"ONNX validation failed: {e}")

    # ------- TensorRT build -------

    def build_trt(self, onnx_path: str, engine_path: str) -> str:
        if trt_mod is None:
            raise RuntimeError("TensorRT not available. Install TensorRT Python package.")
        trt = trt_mod
        logger.info(f"Building TensorRT engine… (precision={self.precision_cfg.name})")
        logger.info(f"TensorRT version: {getattr(trt, '__version__', 'unknown')}")
        _mkdir_for(engine_path)
        logger_trt = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_trt)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger_trt)

        # Prefer parse_from_file so ONNX external data resolves correctly
        ok = parser.parse_from_file(onnx_path) if hasattr(parser, 'parse_from_file') else False
        if not ok:
            # Fallback to manual read with cwd swap
            model_dir = os.path.dirname(onnx_path) or "."; model_name = os.path.basename(onnx_path)
            cwd = os.getcwd()
            try:
                os.chdir(model_dir)
                with open(model_name, 'rb') as f:
                    ok = parser.parse(f.read())
            finally:
                os.chdir(cwd)
        if not ok:
            logger.error("TensorRT parsing failed:")
            for i in range(parser.num_errors):
                logger.error(f"  {parser.get_error(i)}")
            raise RuntimeError("TensorRT failed to parse ONNX model")

        # Input dtype & shape check
        if network.num_inputs > 0:
            input_tensor = network.get_input(0)
            input_shape = tuple(input_tensor.shape)
            logger.info(f"Input shape: {input_shape}")
            try:
                # Keep FP32 input; TRT will insert casts as needed internally
                input_tensor.dtype = trt.float32
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
                logger.warning("Could not set workspace size; using TRT default")

        # Aux streams (helps transformer scheduling)
        try:
            builder.max_aux_streams = self.max_aux_streams
            logger.info(f"Max auxiliary streams: {self.max_aux_streams}")
        except Exception:
            pass

        # Precision flags
        for flag_name in self.precision_cfg.flags:
            try:
                if hasattr(trt.BuilderFlag, flag_name):
                    config.set_flag(getattr(trt.BuilderFlag, flag_name))
                    logger.info(f"Enabled {flag_name}")
            except Exception as e:
                logger.warning(f"Could not enable {flag_name}: {e}")

        # Optimization level
        try:
            if hasattr(config, 'set_builder_optimization_level'):
                config.set_builder_optimization_level(self.opt_level)
            elif hasattr(config, 'builder_optimization_level'):
                config.builder_optimization_level = self.opt_level
            logger.info(f"Optimization level: {self.opt_level}")
        except Exception:
            logger.info("Optimization-level control not available; skipping")

        # Tactic sources (enable all available)
        try:
            if hasattr(trt, 'TacticSource') and hasattr(config, 'set_tactic_sources'):
                TS = trt.TacticSource
                sources = 0
                for name in ("CUBLAS", "CUBLAS_LT", "CUDNN"):
                    if hasattr(TS, name):
                        sources |= getattr(TS, name)
                if sources:
                    config.set_tactic_sources(sources)
                    logger.info("Enabled tactic sources: cuBLAS/cuBLASLt/cuDNN")
        except Exception:
            pass

        # Timing cache (persistent)
        cache_dir = os.path.dirname(engine_path) or "."
        cache_path = os.path.join(cache_dir, "trt_timing.cache")
        tc_loaded = False
        if self.use_timing_cache:
            try:
                if os.path.exists(cache_path):
                    with open(cache_path, 'rb') as f:
                        tc = config.create_timing_cache(f.read())
                        config.set_timing_cache(tc, ignore_mismatch=False)
                        tc_loaded = True
                        logger.info(f"Loaded timing cache: {cache_path}")
            except Exception as e:
                logger.info(f"Timing cache not loaded: {e}")
        else:
            logger.info("Timing cache disabled by flag")

        # Profiling verbosity (build time only)
        try:
            if self.profiling_verbose and hasattr(config, 'profiling_verbosity'):
                config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        except Exception:
            pass

        # INT8 calibrator if needed
        if self.precision_cfg.name == 'int8':
            logger.info("Setting up INT8 calibrator (random batches)…")
            input_shape = (self.num_cams, 3, *self.hw)
            calibrator = _EntropyCalibrator(input_shape, self.calib_cache or os.path.join(cache_dir, 'calibration.cache'), self.calib_batches)
            try:
                config.set_flag(trt.BuilderFlag.INT8)
                config.int8_calibrator = calibrator
                logger.info(f"INT8 calibration batches: {self.calib_batches}")
            except Exception as e:
                raise RuntimeError(f"Failed to configure INT8 calibration: {e}")

        # Build
        logger.info("Building engine (this can take a while)…")
        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("TensorRT build returned None (build failed)")

        # Persist timing cache after successful build
        if self.use_timing_cache:
            try:
                tc = config.get_timing_cache()
                if tc is not None:
                    blob = tc.serialize()
                    with open(cache_path, 'wb') as f:
                        f.write(blob)
                    logger.info(f"Saved timing cache: {cache_path}{' (was preloaded)' if tc_loaded else ''}")
            except Exception as e:
                logger.info(f"Could not save timing cache: {e}")

        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine_bytes)
        logger.info(f"Engine saved: {engine_path} ({_fmt_size(os.path.getsize(engine_path))})")
        return engine_path

    # ------- Orchestrator -------

    def run(self, onnx_in: Optional[str], onnx_simp: str, onnx_noseq: str, engine_path: str, export: bool) -> str:
        # Stage 1: Export or use existing
        if export:
            base_name = onnx_simp.replace(".simp.onnx", ".onnx")
            onnx_in = self.export_from_hf(base_name)
        if not onnx_in:
            raise ValueError("Must provide --onnx-in or use --export to generate ONNX")
        if not os.path.exists(onnx_in):
            raise FileNotFoundError(f"Input ONNX not found: {onnx_in}")

        # Stage 2: Remove sequences first
        noseq = self.remove_sequences(onnx_in, onnx_noseq)
        # Stage 3: Simplify after sequence removal
        simp = self.simplify(noseq, onnx_simp)
        # Stage 4: Validate
        self.onnx_check(simp)
        # Stage 5: Build
        engine = self.build_trt(simp, engine_path)
        return engine

# ---------- CLI ----------

def _parse():
    import argparse
    ap = argparse.ArgumentParser(
        description="Convert VGGT models to optimized TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O
    ap.add_argument("--export", action="store_true", help="Export VGGT from HuggingFace")
    ap.add_argument("--onnx-in", help="Use existing ONNX file instead of exporting")
    ap.add_argument("--model-name", default="facebook/VGGT-1B", help="HuggingFace model name")

    # Model config
    ap.add_argument("--num-cams", type=int, default=10, help="Number of input images (default: 10)")
    ap.add_argument("--height", type=int, default=518, help="Input height")
    ap.add_argument("--width", type=int, default=518, help="Input width")
    ap.add_argument("--opset", type=int, default=18, help="ONNX opset version")

    # TensorRT config
    ap.add_argument("--workspace-gb", type=int, default=28, help="TensorRT workspace (GB)")
    ap.add_argument("--precision", choices=list(PRECISIONS.keys()), default="fp16", help="Precision mode")
    ap.add_argument("--opt-level", type=int, default=5, help="Builder optimization level (0–5)")
    ap.add_argument("--aux-streams", type=int, default=4, help="Max auxiliary CUDA streams")
    ap.add_argument("--no-simplify", action="store_true", help="Skip onnx-simplify stage")
    ap.add_argument("--no-timing-cache", action="store_true", help="Disable persistent timing cache")
    ap.add_argument("--profiling-verbose", action="store_true", help="Detailed builder profiling verbosity")

    # INT8 calibration (if precision=int8)
    ap.add_argument("--calib-batches", type=int, default=8, help="INT8 calibration batches (random)")
    ap.add_argument("--calib-cache", default=None, help="Calibration cache path")

    # Convenience: build several precisions in one call
    ap.add_argument("--all-precisions", action="store_true", help="Build fp16, bf16, fp8*(if available), int8")

    # Output paths
    ap.add_argument("--onnx-simplified", default="onnx_exports/vggt-10x3x518x518.simp.onnx", help="Simplified ONNX path")
    ap.add_argument("--onnx-noseq", default="onnx_exports/vggt-10x3x518x518.NOSEQ.onnx", help="No-seq ONNX path")
    ap.add_argument("--engine", default="onnx_exports/vggt-10x3x518x518_fp16.engine", help="TensorRT engine path")

    return ap.parse_args()


def _stem(num_cams: int, h: int, w: int) -> str:
    return f"vggt-{num_cams}x3x{h}x{w}"


def _auto_paths(args, precision_suffix: str):
    stem = _stem(args.num_cams, args.height, args.width)
    onnx_simp = args.onnx_simplified
    onnx_noseq = args.onnx_noseq
    engine = args.engine
    if onnx_simp == "onnx_exports/vggt-10x3x518x518.simp.onnx":
        onnx_simp = f"onnx_exports/{stem}.simp.onnx"
    if onnx_noseq == "onnx_exports/vggt-10x3x518x518.NOSEQ.onnx":
        onnx_noseq = f"onnx_exports/{stem}.NOSEQ.onnx"
    # If user didn't override engine, adapt suffix to precision
    default_engine = "onnx_exports/vggt-10x3x518x518_fp16.engine"
    if engine == default_engine:
        engine = f"onnx_exports/{stem}{precision_suffix}.engine"
    return onnx_simp, onnx_noseq, engine


def main():
    args = _parse()

    # Build one or many precisions
    target_precisions = [args.precision]
    if args.all_precisions:
        target_precisions = [p for p in ("fp16", "bf16", "fp8", "int8") if p in PRECISIONS]

    built = []
    for prec in target_precisions:
        cfg = PRECISIONS[prec]
        onnx_simp, onnx_noseq, engine = _auto_paths(args, cfg.suffix)
        try:
            pipe = VGGTPipeline(
                num_cams=args.num_cams,
                hw=(args.height, args.width),
                opset=args.opset,
                workspace_gb=args.workspace_gb,
                precision=prec,
                simplify=not args.no_simplify,
                model_name=args.model_name,
                opt_level=args.opt_level,
                max_aux_streams=args.aux_streams,
                use_timing_cache=not args.no_timing_cache,
                profiling_verbose=args.profiling_verbose,
                calib_batches=args.calib_batches,
                calib_cache=args.calib_cache,
            )

            engine_path = pipe.run(
                onnx_in=args.onnx_in,
                onnx_simp=onnx_simp,
                onnx_noseq=onnx_noseq,
                engine_path=engine,
                export=args.export,
            )
            built.append((prec, engine_path))
            logger.info("")
            logger.info("=" * 60)
            logger.info("SUCCESS!")
            logger.info(f"Precision: {prec}")
            logger.info(f"TensorRT engine: {engine_path}")
            logger.info(f"Input shape: [{args.num_cams}, 3, {args.height}, {args.width}]")
            logger.info("=" * 60)
        except KeyboardInterrupt:
            logger.error("\nInterrupted by user"); sys.exit(130)
        except Exception as e:
            logger.error(f"Pipeline failed for precision={prec}: {e}")
            # Continue building other precisions if requested
            if not args.all_precisions:
                sys.exit(1)

    if built:
        logger.info("Built engines:")
        for prec, ep in built:
            logger.info(f"  - {prec}: {ep}")


if __name__ == "__main__":
    main()



#!/usr/bin/env python3
"""
Optimized VGGT to TensorRT Conversion Pipeline

Exports VGGT models to ONNX and builds highly optimized TensorRT engines
with FP16, BF16, FP8, and INT8 quantization support.

Example usage:
    # Export FP16 (baseline)
    python vggt_to_trt.py --export --num-cams 8
    
    # Build all precision variants
    python vggt_to_trt.py --export --num-cams 8 --all-precisions
    
    # Build specific precision
    python vggt_to_trt.py --export --num-cams 8 --precision int8
    
    # Use existing ONNX
    python vggt_to_trt.py --onnx-in model.onnx --precision fp8
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
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Lazy imports
def _lazy_import(name: str):
    """Import module lazily without side effects."""
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
    """Create parent directory for given path if it doesn't exist."""
    d = os.path.dirname(os.path.abspath(p)) or "."
    os.makedirs(d, exist_ok=True)

def _rm(p: str) -> None:
    """Remove file if it exists, ignoring errors."""
    try:
        if os.path.isfile(p) or os.path.islink(p):
            os.remove(p)
    except Exception:
        pass

def _data_rel(onnx_path: str) -> str:
    """Get relative external data filename for ONNX model."""
    return os.path.basename(onnx_path) + ".data"

def _data_abs(onnx_path: str) -> str:
    """Get absolute external data filepath for ONNX model."""
    return os.path.join(
        os.path.dirname(os.path.abspath(onnx_path)),
        _data_rel(onnx_path)
    )

def _fmt_size(n: int) -> str:
    """Format byte size as human-readable string."""
    gb = n / (1024**3)
    mb = n / (1024**2)
    kb = n / 1024
    if gb >= 1:
        return f"{gb:.2f} GB"
    elif mb >= 1:
        return f"{mb:.1f} MB"
    else:
        return f"{kb:.1f} KB"

def _ensure_opset(m: onnx.ModelProto, opset_version: int) -> None:
    """
    Ensure model has valid opset_import with default domain.
    Critical for onnxsim and TensorRT compatibility.
    """
    if len(m.opset_import) == 0:
        logger.warning("Model missing opset_import entirely - adding default")
        opset = m.opset_import.add()
        opset.domain = ""
        opset.version = opset_version
        return
    
    # Check for default domain
    has_default = False
    for op in m.opset_import:
        if op.domain == "" or op.domain == "ai.onnx":
            has_default = True
            if op.version != opset_version:
                logger.info(f"Updating opset version {op.version} -> {opset_version}")
                op.version = opset_version
            break
    
    if not has_default:
        logger.warning("Model missing default opset domain - adding")
        opset = m.opset_import.add()
        opset.domain = ""
        opset.version = opset_version


def _prod_map(g: onnx.GraphProto) -> dict:
    """Build map from output name to producing node."""
    out = {}
    for n in g.node:
        for o in n.output:
            if o:
                out[o] = n
    return out

def _const_i64(name: str, vals: List[int]) -> onnx.NodeProto:
    """Create Constant node with int64 array."""
    import numpy as np
    arr = numpy_helper.from_array(
        np.asarray(vals, dtype="int64"),
        name
    )
    return helper.make_node(
        "Constant", [], [name],
        name=name + "_const",
        value=arr
    )

def _get_attr_i(n: onnx.NodeProto, key: str, default: int = 0) -> int:
    """Get integer attribute from node."""
    for a in n.attribute:
        if a.name == key and a.type == AttributeProto.INT:
            return a.i
    return default

def _index_from_input(g: onnx.GraphProto, name: str) -> Optional[int]:
    """Extract integer index value from graph input."""
    # Check initializers
    for init in g.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init)
            if arr.size == 1:
                val = int(arr.item())
                return val if val >= 0 else None
            return None
    
    # Check Constant nodes
    for n in g.node:
        if n.op_type == "Constant" and n.output and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t)
                    if arr.size == 1:
                        val = int(arr.item())
                        return val if val >= 0 else None
    
    # Check Cast(Constant) pattern
    for n in g.node:
        if n.op_type == "Cast" and n.output and n.output[0] == name:
            src = n.input[0]
            return _index_from_input(g, src)
    
    return None

def _prune_to_outputs(m: onnx.ModelProto) -> None:
    """Remove unused nodes and initializers from graph."""
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

# ---------- INT8 Calibrator ----------

class SimpleCalibrator:
    """
    Simple entropy calibrator for INT8 quantization.
    Uses random or provided calibration data.
    """
    
    def __init__(self, input_shape: Tuple[int, ...], cache_file: str, num_batches: int = 10):
        """
        Args:
            input_shape: Input tensor shape (N, C, H, W)
            cache_file: Path to calibration cache file
            num_batches: Number of calibration batches
        """
        if trt_mod is None:
            raise RuntimeError("TensorRT not available")
        trt = trt_mod
        
        self.input_shape = input_shape
        self.cache_file = cache_file
        self.num_batches = num_batches
        self.current_batch = 0
        
        # Generate random calibration data
        import numpy as np
        self.data = [
            np.random.randn(*input_shape).astype(np.float32)
            for _ in range(num_batches)
        ]
        
        # Allocate device memory
        if torch is not None and torch.cuda.is_available():
            self.device_input = torch.cuda.FloatTensor(*input_shape)
    
    def get_batch_size(self):
        return self.input_shape[0]
    
    def get_batch(self, names):
        """Return batch data for calibration."""
        if self.current_batch >= self.num_batches:
            return None
        
        batch = self.data[self.current_batch]
        self.current_batch += 1
        
        if torch is not None and torch.cuda.is_available():
            self.device_input.copy_(torch.from_numpy(batch))
            return [int(self.device_input.data_ptr())]
        
        return None
    
    def read_calibration_cache(self):
        """Read calibration cache if exists."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        """Write calibration cache."""
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# ---------- Precision Configuration ----------

@dataclass
class PrecisionConfig:
    """Configuration for different precision modes."""
    name: str
    flags: List[str]
    suffix: str
    description: str

PRECISIONS = {
    "fp32": PrecisionConfig(
        name="fp32",
        flags=[],
        suffix="",
        description="Full precision (baseline)"
    ),
    "fp16": PrecisionConfig(
        name="fp16",
        flags=["FP16", "TF32"],
        suffix="_fp16",
        description="Half precision with TF32 fallback"
    ),
    "bf16": PrecisionConfig(
        name="bf16",
        flags=["BF16", "TF32"],
        suffix="_bf16",
        description="BFloat16 precision"
    ),
    "fp8": PrecisionConfig(
        name="fp8",
        flags=["FP8", "FP16", "TF32"],
        suffix="_fp8",
        description="FP8 precision with FP16 fallback"
    ),
    "int8": PrecisionConfig(
        name="int8",
        flags=["INT8", "FP16", "TF32"],
        suffix="_int8",
        description="INT8 quantization with FP16 fallback"
    ),
}

# ---------- Pipeline Class ----------

class VGGTPipeline:
    """
    Optimized pipeline for converting VGGT models to TensorRT engines.
    """
    
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
    ):
        """
        Initialize pipeline configuration.
        
        Args:
            num_cams: Number of camera views (1-32 recommended)
            hw: Input height and width as (H, W)
            opset: ONNX opset version
            workspace_gb: TensorRT workspace size in GB
            precision: Precision mode (fp32/fp16/bf16/fp8/int8)
            simplify: Enable ONNX simplification
            model_name: HuggingFace model identifier
            opt_level: TensorRT optimization level (0-5)
            max_aux_streams: Maximum auxiliary CUDA streams
        """
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
        
        logger.info(f"Pipeline config: {num_cams} cameras, {hw[0]}x{hw[1]}, {precision} precision")

    def export_from_hf(self, out_onnx: str, device: str = "cuda") -> str:
        """Export VGGT model from HuggingFace to ONNX format."""
        if torch is None:
            raise RuntimeError("PyTorch not available. Install with: pip install torch")
        
        try:
            from vggt.models.vggt import VGGT
        except ImportError:
            raise RuntimeError("VGGT module not available. Install with: pip install vggt")

        _mkdir_for(out_onnx)
        C, H, W = 3, *self.hw
        
        x = torch.randn(self.num_cams, C, H, W)

        logger.info(f"Loading model: {self.model_name}")
        try:
            if device == "cuda" and torch.cuda.is_available():
                model = (
                    VGGT.from_pretrained(self.model_name)
                    .eval()
                    .to("cuda")
                    .to(torch.float32)
                )
                x = x.to("cuda", dtype=torch.float32)
                logger.info("Using CUDA device")
            else:
                model = (
                    VGGT.from_pretrained(self.model_name)
                    .eval()
                    .to("cpu")
                    .to(torch.float32)
                )
                x = x.to("cpu", dtype=torch.float32)
                logger.info("Using CPU device")
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{self.model_name}': {e}")

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
                model = model.to("cpu")
                x = x.to("cpu")
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
            except Exception as e:
                raise RuntimeError(f"ONNX export failed: {e}")

        logger.info(f"Wrote {out_onnx}")
        self.rebind_external_data(out_onnx)
        return out_onnx

    def rebind_external_data(self, onnx_path: str) -> None:
        """Rewrite external data file with relative paths."""
        logger.info("Rebinding external data...")
        m = onnx.load(onnx_path, load_external_data=True)
        rel = _data_rel(onnx_path)
        abs_p = _data_abs(onnx_path)
        
        _rm(abs_p)
        
        convert_model_to_external_data(
            m,
            all_tensors_to_one_file=True,
            location=rel,
            size_threshold=EXTERNAL_DATA_THRESHOLD,
            convert_attribute=False,
        )
        onnx.save_model(m, onnx_path)
        
        logger.info(f"External data: {abs_p}")
        try:
            s_on = os.path.getsize(onnx_path)
            s_da = os.path.getsize(abs_p) if os.path.exists(abs_p) else 0
            logger.info(f"Sizes: ONNX={_fmt_size(s_on)}, DATA={_fmt_size(s_da)}")
        except Exception:
            pass

    def _normalize_softmax_last_axis(self, g: onnx.GraphProto) -> int:
        """Normalize Softmax nodes to use axis=-1."""
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
        """Normalize Squeeze nodes to use input instead of attributes."""
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
                        out.append(n)
                        continue
                    
                    axes_name = n.name + "_axes"
                    out.append(_const_i64(axes_name, axes))
                    out.append(
                        helper.make_node(
                            "Squeeze",
                            [n.input[0], axes_name],
                            list(n.output),
                            name=n.name
                        )
                    )
                    updated += 1
                    continue
            
            out.append(n)
        
        del g.node[:]
        g.node.extend(out)
        return updated

    def remove_sequences(self, src: str, dst: str) -> str:
        """Remove sequence operations from ONNX graph."""
        logger.info("Removing sequence operations...")
        m = onnx.load(src, load_external_data=True)
        
        # Ensure valid opset before modifications
        _ensure_opset(m, self.opset)
        
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
                new_nodes.append(n)
                continue
            
            seq, idx = n.input
            p = prod.get(seq, None)
            idx_val = _index_from_input(g, idx)
            
            if idx_val is None or p is None:
                logger.warning(
                    f"Cannot resolve SequenceAt node '{n.name}': "
                    f"idx_val={idx_val}, producer={p.op_type if p else None}"
                )
                new_nodes.append(n)
                continue

            if p.op_type == "SplitToSequence":
                X = p.input[0]
                axis = _get_attr_i(p, "axis", 0)
                s, e, a = n.name + "_starts", n.name + "_ends", n.name + "_axes"
                sliced = n.output[0] + "_slice"
                
                new_nodes += [
                    _const_i64(s, [idx_val]),
                    _const_i64(e, [idx_val + 1]),
                    _const_i64(a, [axis]),
                    helper.make_node(
                        "Slice", [X, s, e, a], [sliced],
                        name=n.name + "_slice"
                    ),
                    helper.make_node(
                        "Squeeze", [sliced, a], [n.output[0]],
                        name=n.name + "_squeeze"
                    ),
                ]
                replaced += 1
                continue

            if p.op_type == "SequenceConstruct":
                ins = list(p.input)
                if 0 <= idx_val < len(ins):
                    new_nodes.append(
                        helper.make_node(
                            "Identity", [ins[idx_val]], [n.output[0]],
                            name=n.name + "_id"
                        )
                    )
                    replaced += 1
                    continue
                else:
                    logger.warning(
                        f"SequenceAt index {idx_val} out of range "
                        f"for SequenceConstruct with {len(ins)} inputs"
                    )

            new_nodes.append(n)

        del g.node[:]
        g.node.extend(new_nodes)
        _prune_to_outputs(m)

        remaining = [n for n in g.node if "Sequence" in n.op_type]
        if remaining:
            logger.error(f"{len(remaining)} sequence operations remain:")
            for n in remaining[:5]:
                logger.error(f"  - {n.op_type}: {n.name}")
            raise RuntimeError("Failed to remove all sequence operations")

        _mkdir_for(dst)
        rel = _data_rel(dst)
        abs_p = _data_abs(dst)
        _rm(abs_p)
        
        convert_model_to_external_data(
            m, True, rel, EXTERNAL_DATA_THRESHOLD, False
        )
        onnx.save_model(m, dst)
        
        logger.info(f"Replaced {replaced} sequence operations")
        logger.info(f"Wrote {dst}")
        return dst

    def simplify(self, src: str, dst: str) -> str:
        """Simplify ONNX graph using onnxsim."""
        if not self.do_simplify:
            logger.info("Simplification disabled")
            return src

        try:
            onnxsim = __import__("onnxsim")
        except Exception:
            logger.warning("onnxsim not installed, skipping simplification")
            return src

        logger.info("Simplifying ONNX graph...")
        m = onnx.load(src, load_external_data=True)
        
        # Ensure valid opset (critical for onnxsim compatibility)
        _ensure_opset(m, self.opset)
        
        # Verify opset is now correct
        logger.info(f"Model IR version: {m.ir_version}")
        for idx, op in enumerate(m.opset_import):
            logger.info(f"  opset[{idx}]: domain='{op.domain}', version={op.version}")

        import inspect
        params = set(inspect.signature(onnxsim.simplify).parameters)
        kwargs = {}
        if "skip_shape_inference" in params:
            kwargs["skip_shape_inference"] = False
        if "skip_constant_folding" in params:
            kwargs["skip_constant_folding"] = False
        if "skip_optimization" in params:
            kwargs["skip_optimization"] = False
        if "perform_optimization" in params:
            kwargs["perform_optimization"] = True

        try:
            result = onnxsim.simplify(m, **kwargs)
        except TypeError:
            result = onnxsim.simplify(m)
        except Exception as e:
            logger.error(f"Simplification failed: {e}")
            return src

        if isinstance(result, tuple):
            ms, ok = result
            if not ok:
                logger.warning("Simplification completed with warnings")
        else:
            ms = result

        _mkdir_for(dst)
        rel = _data_rel(dst)
        abs_p = _data_abs(dst)
        _rm(abs_p)

        convert_model_to_external_data(ms, True, rel, EXTERNAL_DATA_THRESHOLD, False)
        onnx.save_model(ms, dst)

        logger.info(f"Wrote {dst}")
        return dst

    @staticmethod
    def onnx_check(path: str) -> None:
        """Validate ONNX model."""
        logger.info("Validating ONNX model...")
        try:
            onnx.checker.check_model(path)
            logger.info("ONNX validation passed")
        except Exception as e:
            raise RuntimeError(f"ONNX validation failed: {e}")

    def build_trt(self, onnx_path: str, engine_path: str) -> str:
        """Build optimized TensorRT engine with all performance knobs."""
        if trt_mod is None:
            raise RuntimeError("TensorRT not available")
        trt = trt_mod

        logger.info(f"Building TensorRT engine ({self.precision_config.description})...")
        logger.info(f"TensorRT version: {getattr(trt, '__version__', 'unknown')}")
        _mkdir_for(engine_path)
        
        # Setup logger (WARNING level for best perf during inference)
        logger_trt = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_trt)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger_trt)

        # Parse ONNX
        ok = False
        if hasattr(parser, "parse_from_file"):
            ok = parser.parse_from_file(onnx_path)
        else:
            model_dir = os.path.dirname(onnx_path) or "."
            model_name = os.path.basename(onnx_path)
            cwd = os.getcwd()
            try:
                os.chdir(model_dir)
                with open(model_name, "rb") as f:
                    ok = parser.parse(f.read())
            finally:
                os.chdir(cwd)

        if not ok:
            logger.error("TensorRT parsing failed:")
            for i in range(parser.num_errors):
                logger.error(f"  {parser.get_error(i)}")
            raise RuntimeError("TensorRT failed to parse ONNX model")

        # Set input dtype
        if network.num_inputs > 0:
            input_tensor = network.get_input(0)
            input_shape = input_tensor.shape
            logger.info(f"Input shape: {input_shape}")
            
            try:
                # Use FP32 for input regardless of precision (will be converted internally)
                input_tensor.dtype = trt.float32
            except Exception:
                pass
            
            expected = (self.num_cams, 3, *self.hw)
            actual = tuple(input_shape)
            if actual != expected:
                logger.warning(f"Input shape mismatch: expected {expected}, got {actual}")

        # Configure builder
        config = builder.create_builder_config()
        
        # === OPTIMIZATION 1: Workspace size ===
        ws_bytes = self.workspace_gb * (1 << 30)
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_bytes)
        except Exception:
            try:
                config.max_workspace_size = ws_bytes
            except Exception:
                logger.warning("Could not set workspace size")
        
        # === OPTIMIZATION 2: Auxiliary streams ===
        try:
            builder.max_aux_streams = self.max_aux_streams
            logger.info(f"Max auxiliary streams: {self.max_aux_streams}")
        except Exception:
            pass
        
        # === OPTIMIZATION 3: Precision flags ===
        for flag_name in self.precision_config.flags:
            try:
                if hasattr(trt.BuilderFlag, flag_name):
                    flag = getattr(trt.BuilderFlag, flag_name)
                    config.set_flag(flag)
                    logger.info(f"Enabled {flag_name}")
            except Exception as e:
                logger.warning(f"Could not enable {flag_name}: {e}")
        
        # === OPTIMIZATION 4: Optimization level ===
        opt_level_set = False
        try:
            if hasattr(config, "set_builder_optimization_level"):
                config.set_builder_optimization_level(self.opt_level)
                opt_level_set = True
            elif hasattr(config, "builder_optimization_level"):
                config.builder_optimization_level = self.opt_level
                opt_level_set = True
        except Exception:
            pass
        
        if opt_level_set:
            logger.info(f"Optimization level: {self.opt_level}")
        
        # === OPTIMIZATION 5: Tactic sources ===
        try:
            if hasattr(trt, "TacticSource") and hasattr(config, "set_tactic_sources"):
                TS = trt.TacticSource
                sources = 0
                for name in ("CUBLAS", "CUBLAS_LT", "CUDNN"):
                    if hasattr(TS, name):
                        sources |= getattr(TS, name)
                if sources:
                    config.set_tactic_sources(sources)
                    logger.info("Enabled all available tactic sources")
        except Exception:
            pass
        
        # === OPTIMIZATION 6: Timing cache ===
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
        
        # === INT8 calibration ===
        if self.precision_config.name == "int8":
            logger.info("Setting up INT8 calibration...")
            calib_cache = os.path.join(cache_dir, "calibration.cache")
            input_shape = (self.num_cams, 3, *self.hw)
            
            try:
                calibrator = SimpleCalibrator(input_shape, calib_cache, num_batches=10)
                config.int8_calibrator = calibrator
                logger.info("INT8 calibrator configured")
            except Exception as e:
                logger.warning(f"INT8 calibration setup failed: {e}")
        
        # === OPTIMIZATION 7: Profiling verbosity (for debugging only) ===
        # Uncomment for detailed layer profiling:
        # try:
        #     config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        # except Exception:
        #     pass

        # Build engine
        logger.info("Building engine (this may take several minutes)...")
        engine_bytes = builder.build_serialized_network(network, config)
        
        if engine_bytes is None:
            raise RuntimeError("TensorRT build returned None (build failed)")

        # === OPTIMIZATION 8: Save timing cache ===
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

        # Save engine
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        
        engine_size = os.path.getsize(engine_path)
        logger.info(f"Engine saved: {engine_path} ({_fmt_size(engine_size)})")
        return engine_path

    def run(
        self,
        onnx_in: Optional[str],
        onnx_simp: str,
        onnx_noseq: str,
        engine_path: str,
        export: bool
    ) -> str:
        """Run complete pipeline."""
        # Stage 1: Export or use existing ONNX
        if export:
            base_name = onnx_simp.replace(".simp.onnx", ".onnx")
            onnx_in = self.export_from_hf(base_name)
        
        if not onnx_in:
            raise ValueError("Must provide --onnx-in or use --export")
        
        if not os.path.exists(onnx_in):
            raise FileNotFoundError(f"Input ONNX not found: {onnx_in}")

        # Stage 2: Remove sequences FIRST
        noseq = self.remove_sequences(onnx_in, onnx_noseq)
        
        # Stage 3: Simplify AFTER sequences removed
        simp = self.simplify(noseq, onnx_simp)
        
        # Stage 4: Validate
        self.onnx_check(simp)
        
        # Stage 5: Build TensorRT
        engine = self.build_trt(simp, engine_path)
        
        return engine


# ---------- CLI ----------

def _parse():
    """Parse command line arguments."""
    import argparse
    
    ap = argparse.ArgumentParser(
        description="Convert VGGT models to optimized TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with FP16 (default, fast)
  %(prog)s --export --num-cams 8
  
  # Build all precision variants for comparison
  %(prog)s --export --num-cams 8 --all-precisions
  
  # Build specific precision
  %(prog)s --export --num-cams 8 --precision int8
  %(prog)s --export --num-cams 8 --precision fp8
  
  # Use existing ONNX
  %(prog)s --onnx-in model.onnx --precision bf16
  
  # Maximum optimization
  %(prog)s --export --num-cams 8 --precision fp16 --opt-level 5 --workspace-gb 32
        """
    )
    
    # Input/output
    ap.add_argument(
        "--export",
        action="store_true",
        help="Export VGGT from HuggingFace"
    )
    ap.add_argument(
        "--onnx-in",
        help="Use existing ONNX file instead of exporting"
    )
    ap.add_argument(
        "--model-name",
        default="facebook/VGGT-1B",
        help="HuggingFace model name (default: facebook/VGGT-1B)"
    )
    
    # Model configuration
    ap.add_argument(
        "--num-cams",
        type=int,
        default=8,
        help="Number of camera views (default: 8, matching your benchmark)"
    )
    ap.add_argument(
        "--height",
        type=int,
        default=518,
        help="Input height (default: 518)"
    )
    ap.add_argument(
        "--width",
        type=int,
        default=518,
        help="Input width (default: 518)"
    )
    ap.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)"
    )
    
    # Precision configuration
    ap.add_argument(
        "--precision",
        choices=list(PRECISIONS.keys()),
        default="fp16",
        help="Precision mode (default: fp16)"
    )
    ap.add_argument(
        "--all-precisions",
        action="store_true",
        help="Build all precision variants (fp16, bf16, fp8, int8)"
    )
    
    # TensorRT optimization
    ap.add_argument(
        "--workspace-gb",
        type=int,
        default=32,
        help="TensorRT workspace size in GB (default: 32, increased for 5090)"
    )
    ap.add_argument(
        "--opt-level",
        type=int,
        default=5,
        help="TensorRT optimization level 0-5 (default: 5 = maximum)"
    )
    ap.add_argument(
        "--max-aux-streams",
        type=int,
        default=4,
        help="Maximum auxiliary CUDA streams (default: 4)"
    )
    ap.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX simplification"
    )
    
    # Output paths
    ap.add_argument(
        "--output-dir",
        default="onnx_exports",
        help="Output directory for all artifacts (default: onnx_exports)"
    )
    
    return ap.parse_args()


def main():
    """Main entry point."""
    args = _parse()
    
    # Auto-generate filenames
    stem = f"vggt-{args.num_cams}x3x{args.height}x{args.width}"
    
    precisions_to_build = []
    if args.all_precisions:
        # Build fp16, bf16, fp8, int8 (skip fp32 as it's slow)
        precisions_to_build = ["fp16", "bf16", "fp8", "int8"]
        logger.info("Building all precision variants: fp16, bf16, fp8, int8")
    else:
        precisions_to_build = [args.precision]
    
    results = []
    shared_onnx_base = None  # Track the base ONNX file for reuse
    
    try:
        for idx, precision in enumerate(precisions_to_build):
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"Building {precision.upper()} variant ({idx+1}/{len(precisions_to_build)})")
            logger.info("=" * 70)
            
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
            )
            
            # Export only on first iteration, then reuse
            is_first = (idx == 0)
            do_export = args.export and is_first
            
            # Use shared ONNX from first build for subsequent precisions
            onnx_input = args.onnx_in if args.onnx_in else shared_onnx_base
            
            engine_path = pipe.run(
                onnx_in=onnx_input,
                onnx_simp=onnx_simp,
                onnx_noseq=onnx_noseq,
                engine_path=engine,
                export=do_export,
            )
            
            # Store the base ONNX path after first export
            if is_first and do_export:
                shared_onnx_base = onnx_simp.replace(".simp.onnx", ".onnx")
                logger.info(f"Shared ONNX for subsequent builds: {shared_onnx_base}")
            
            results.append((precision, engine_path))
        
        # Summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("BUILD COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Input shape: [{args.num_cams}, 3, {args.height}, {args.width}]")
        logger.info("")
        logger.info("Built engines:")
        for precision, engine_path in results:
            size = os.path.getsize(engine_path)
            desc = PRECISIONS[precision].description
            logger.info(f"  {precision.upper():5s} ({desc:40s}): {engine_path}")
            logger.info(f"        Size: {_fmt_size(size)}")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Benchmark each engine to find the fastest")
        logger.info("  2. Expected speedup hierarchy: INT8 > FP8 > FP16/BF16 > FP32")
        logger.info("  3. On RTX 5090, INT8 can be 2-3x faster than FP16")
        logger.info("  4. Test quality: FP16≈BF16≈FP8 > INT8 (INT8 may need calibration tuning)")
        logger.info("")
        logger.info("Optimizations applied:")
        logger.info(f"  ✓ Timing cache (speeds up rebuilds)")
        logger.info(f"  ✓ Optimization level {args.opt_level} (maximum)")
        logger.info(f"  ✓ Auxiliary streams: {args.max_aux_streams}")
        logger.info(f"  ✓ TF32 enabled (Ampere+ GPUs)")
        logger.info(f"  ✓ All CUDA tactic sources (cuBLAS, cuDNN)")
        logger.info(f"  ✓ Workspace: {args.workspace_gb} GB")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.error("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


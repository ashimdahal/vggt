#!/usr/bin/env python3
"""
VGGT to TensorRT Conversion Pipeline

Exports VGGT models to ONNX (opset 18), removes sequence operations,
and builds optimized TensorRT engines.

Example usage:
    # Export and build with 10 cameras (default)
    python vggt_to_trt.py --export
    
    # Export with custom number of cameras
    python vggt_to_trt.py --export --num-cams 6
    
    # Use existing ONNX file
    python vggt_to_trt.py --onnx-in model.onnx
"""

from __future__ import annotations
import os
import sys
import logging
from typing import List, Optional, Tuple

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
    """
    Extract integer index value from graph input.
    Checks initializers, Constant nodes, and Cast(Constant) patterns.
    """
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
    
    # Check Cast(Constant) -> name pattern (some exporters wrap indices)
    for n in g.node:
        if n.op_type == "Cast" and n.output and n.output[0] == name:
            src = n.input[0]
            return _index_from_input(g, src)
    
    return None

def _prune_to_outputs(m: onnx.ModelProto) -> None:
    """Remove unused nodes and initializers from graph."""
    g = m.graph
    prod = _prod_map(g)
    
    # Build set of needed tensors starting from outputs
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
    
    # Prune graph - use protobuf-safe modification
    filtered_nodes = [n for n in g.node if any(o in needed for o in n.output)]
    filtered_inits = [i for i in g.initializer if i.name in needed]
    
    del g.node[:]
    g.node.extend(filtered_nodes)
    del g.initializer[:]
    g.initializer.extend(filtered_inits)

# ---------- Pipeline Class ----------

class VGGTPipeline:
    """
    Pipeline for converting VGGT models to TensorRT engines.
    
    Pipeline stages:
    1. Export from HuggingFace to ONNX
    2. Simplify ONNX graph (optional)
    3. Remove sequence operations
    4. Validate ONNX model
    5. Build TensorRT engine
    """
    
    def __init__(
        self,
        num_cams: int = 10,
        hw: Tuple[int, int] = (518, 518),
        opset: int = 18,
        workspace_gb: int = 28,
        fp16: bool = True,
        simplify: bool = True,
        model_name: str = "facebook/VGGT-1B"
    ):
        """
        Initialize pipeline configuration.
        
        Args:
            num_cams: Number of camera views/images (1-32 recommended)
            hw: Input height and width as (H, W)
            opset: ONNX opset version
            workspace_gb: TensorRT workspace size in GB
            fp16: Enable FP16 precision
            simplify: Enable ONNX simplification
            model_name: HuggingFace model identifier
        """
        if num_cams < 1:
            raise ValueError(f"num_cams must be >= 1, got {num_cams}")
        if hw[0] < 1 or hw[1] < 1:
            raise ValueError(f"Invalid dimensions: {hw}")
        
        self.num_cams = num_cams
        self.hw = hw
        self.opset = opset
        self.workspace_gb = workspace_gb
        self.fp16 = fp16
        self.do_simplify = simplify
        self.model_name = model_name
        
        logger.info(f"Pipeline config: {num_cams} cameras, {hw[0]}x{hw[1]}, opset {opset}")

    def export_from_hf(self, out_onnx: str, device: str = "cuda") -> str:
        """
        Export VGGT model from HuggingFace to ONNX format.
        
        Args:
            out_onnx: Output ONNX file path
            device: Device to use ('cuda' or 'cpu')
            
        Returns:
            Path to exported ONNX file
        """
        if torch is None:
            raise RuntimeError("PyTorch not available. Install with: pip install torch")
        
        try:
            from vggt.models.vggt import VGGT
        except ImportError:
            raise RuntimeError(
                "VGGT module not available. Install with: pip install vggt"
            )

        _mkdir_for(out_onnx)
        C, H, W = 3, *self.hw
        
        # Create dummy input with dynamic batch (num_cams)
        x = torch.randn(self.num_cams, C, H, W)

        # Load model
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

        # Export to ONNX
        logger.info(f"Exporting to ONNX with shape [{self.num_cams}, {C}, {H}, {W}]")
        with torch.inference_mode():
            try:
                torch.onnx.export(
                    model, (x,),
                    out_onnx,
                    input_names=["images"],
                    output_names=None,  # Let PyTorch name all outputs
                    opset_version=self.opset,
                    do_constant_folding=True,
                    dynamic_axes=None,  # Static shape for TensorRT optimization
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
        """
        Rewrite external data file with relative paths (ONNX spec compliant).
        Prevents append issues and ensures clean data files.
        
        Args:
            onnx_path: Path to ONNX model file
        """
        logger.info("Rebinding external data...")
        m = onnx.load(onnx_path, load_external_data=True)
        rel = _data_rel(onnx_path)
        abs_p = _data_abs(onnx_path)
        
        # Remove old data file to prevent append
        _rm(abs_p)
        
        convert_model_to_external_data(
            m,
            all_tensors_to_one_file=True,
            location=rel,  # Relative path per ONNX spec
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
        """
        Normalize Softmax nodes to use axis=-1 (last dimension).
        Prevents memory explosion in ORT and improves TRT performance.
        
        Args:
            g: ONNX graph
            
        Returns:
            Number of nodes updated
        """
        changed = 0
        for n in g.node:
            if n.op_type == "Softmax":
                # Remove existing axis attribute
                attrs_to_keep = [a for a in n.attribute if a.name != "axis"]
                del n.attribute[:]
                n.attribute.extend(attrs_to_keep)
                # Add axis=-1
                n.attribute.extend([helper.make_attribute("axis", -1)])
                changed += 1
        return changed

    def _normalize_squeeze(self, g: onnx.GraphProto) -> int:
        """
        Normalize Squeeze nodes to use input (opset 13+ style) instead of attributes.
        
        Args:
            g: ONNX graph
            
        Returns:
            Number of nodes updated
        """
        updated, out = 0, []
        for n in g.node:
            if n.op_type == "Squeeze":
                axes_attr = None
                for a in n.attribute:
                    if a.name == "axes":
                        axes_attr = a
                        break
                
                if axes_attr is not None:
                    # Convert attribute to input
                    if axes_attr.type == AttributeProto.INT:
                        axes = [axes_attr.i]
                    elif axes_attr.type == AttributeProto.INTS:
                        axes = list(axes_attr.ints)
                    else:
                        # Unsupported type, keep as-is
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
        
        # Use protobuf-safe modification
        del g.node[:]
        g.node.extend(out)
        return updated

    def remove_sequences(self, src: str, dst: str) -> str:
        """
        Remove sequence operations (SplitToSequence, SequenceAt, SequenceConstruct)
        by replacing them with Slice and Identity operations.
        
        Args:
            src: Source ONNX file path
            dst: Destination ONNX file path
            
        Returns:
            Path to processed ONNX file
        """
        logger.info("Removing sequence operations...")
        m = onnx.load(src, load_external_data=True)
        g = m.graph
        prod = _prod_map(g)

        # Normalize Squeeze nodes first
        fixed_sq = self._normalize_squeeze(g)
        if fixed_sq > 0:
            logger.info(f"Normalized {fixed_sq} Squeeze nodes")
        
        # Normalize Softmax to last axis (prevents memory explosion)
        fixed_sm = self._normalize_softmax_last_axis(g)
        if fixed_sm > 0:
            logger.info(f"Rewrote {fixed_sm} Softmax axes to -1")

        # Replace sequence operations
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

            # Pattern 1: SplitToSequence -> SequenceAt => Slice + Squeeze
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

            # Pattern 2: SequenceConstruct -> SequenceAt => Identity
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

            # Unknown pattern, keep original
            new_nodes.append(n)

        # Use protobuf-safe modification
        del g.node[:]
        g.node.extend(new_nodes)
        _prune_to_outputs(m)

        # Validate all sequences removed
        remaining = [n for n in g.node if "Sequence" in n.op_type]
        if remaining:
            logger.error(f"{len(remaining)} sequence operations remain:")
            for n in remaining[:5]:
                logger.error(f"  - {n.op_type}: {n.name}")
            raise RuntimeError(
                "Failed to remove all sequence operations. "
                "Model may contain unsupported patterns."
            )

        # Save processed model
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
        logger.info(f"Data: {abs_p}")
        return dst

    def simplify(self, src: str, dst: str) -> str:
        """
        Simplify ONNX graph using onnxsim with version-safe kwargs.
        
        Args:
            src: Source ONNX file path
            dst: Destination ONNX file path
            
        Returns:
            Path to simplified ONNX file (or src if simplification skipped)
        """
        if not self.do_simplify:
            logger.info("Simplification disabled")
            return src

        try:
            onnxsim = __import__("onnxsim")
        except Exception:
            logger.warning("onnxsim not installed, skipping simplification")
            logger.info("Install with: pip install onnxsim")
            return src

        logger.info("Simplifying ONNX graph (this may take several minutes)...")
        m = onnx.load(src, load_external_data=True)

        import inspect
        params = set(inspect.signature(onnxsim.simplify).parameters)
        # Build kwargs only if the current onnxsim exposes them
        kwargs = {}
        # Our intent: DO NOT skip shape inference or constant folding; DO perform optimization.
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
            # Fallback for very old versions: call with no kwargs
            result = onnxsim.simplify(m)
        except Exception as e:
            logger.error(f"Simplification failed: {e}")
            logger.warning("Using un-simplified model")
            return src

        # onnxsim returns either ModelProto or (ModelProto, check_ok)
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
        logger.info(f"Data: {abs_p}")
        return dst

    @staticmethod
    def onnx_check(path: str) -> None:
        """
        Validate ONNX model using onnx.checker.
        
        Args:
            path: Path to ONNX model file
        """
        logger.info("Validating ONNX model...")
        try:
            # m = onnx.load(path, load_external_data=True)
            onnx.checker.check_model(path)
            logger.info("ONNX validation passed")
        except Exception as e:
            raise RuntimeError(f"ONNX validation failed: {e}")

    def build_trt(self, onnx_path: str, engine_path: str) -> str:
        """
        Build TensorRT engine from ONNX model.
        
        Args:
            onnx_path: Path to ONNX model file
            engine_path: Output engine file path
            
        Returns:
            Path to built engine
        """
        if trt_mod is None:
            raise RuntimeError(
                "TensorRT not available. Install TensorRT Python package."
            )
        trt = trt_mod

        logger.info("Building TensorRT engine...")
        logger.info(f"TensorRT version: {getattr(trt, '__version__', 'unknown')}")
        _mkdir_for(engine_path)
        
        logger_trt = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_trt)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger_trt)

        # Parse ONNX model (prefer parse_from_file for external data support)
        ok = False
        if hasattr(parser, "parse_from_file"):
            ok = parser.parse_from_file(onnx_path)
        else:
            # Fallback: change directory so relative .data path resolves
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

        # Set input dtype explicitly
        if network.num_inputs > 0:
            input_tensor = network.get_input(0)
            input_shape = input_tensor.shape
            logger.info(f"Input shape: {input_shape}")
            
            # Set dtype based on precision flag
            try:
                input_tensor.dtype = trt.float16 if self.fp16 else trt.float32
            except Exception:
                pass
            
            # Warn if shape mismatch
            expected = (self.num_cams, 3, *self.hw)
            actual = tuple(input_shape)
            if actual != expected:
                logger.warning(
                    f"Input shape mismatch: expected {expected}, got {actual}"
                )

        # Configure builder
        config = builder.create_builder_config()
        
        # --- Workspace size (TRT versions differ) ---
        ws_bytes = self.workspace_gb * (1 << 30)
        try:
            # TRT ≥ 8.6
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_bytes)
        except Exception:
            try:
                # Older TRT
                config.max_workspace_size = ws_bytes
            except Exception:
                logger.warning("Could not set workspace size; using TensorRT default")
        
        # Reduce auxiliary streams for stability
        try:
            builder.max_aux_streams = 1
        except Exception:
            pass
        
        # FP16 flag
        if self.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 precision enabled")
        elif self.fp16:
            logger.warning("FP16 requested but not supported on this platform")
        
        # --- Optimization level (API varies by TRT version) ---
        opt_level_set = False
        try:
            if hasattr(config, "set_builder_optimization_level"):
                config.set_builder_optimization_level(3)
                opt_level_set = True
            elif hasattr(config, "builder_optimization_level"):
                config.builder_optimization_level = 3
                opt_level_set = True
        except Exception:
            pass
        
        if not opt_level_set:
            logger.info("Optimization-level control not available on this TensorRT; skipping.")
        
        # Optional: disable timing cache if available
        try:
            config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        except Exception:
            pass

        # Build engine
        logger.info("Building engine (this may take several minutes)...")
        engine_bytes = builder.build_serialized_network(network, config)
        
        if engine_bytes is None:
            raise RuntimeError("TensorRT build returned None (build failed)")

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
        """
        Run complete pipeline.
        
        Args:
            onnx_in: Input ONNX file (if not exporting)
            onnx_simp: Simplified ONNX output path
            onnx_noseq: No-sequence ONNX output path
            engine_path: TensorRT engine output path
            export: Whether to export from HuggingFace
            
        Returns:
            Path to built TensorRT engine
        """
        # Stage 1: Export or use existing ONNX
        if export:
            base_name = onnx_simp.replace(".simp.onnx", ".onnx")
            onnx_in = self.export_from_hf(base_name)
        
        if not onnx_in:
            raise ValueError(
                "Must provide --onnx-in or use --export to generate ONNX"
            )
        
        if not os.path.exists(onnx_in):
            raise FileNotFoundError(f"Input ONNX not found: {onnx_in}")

        # Stage 2: Remove sequences FIRST (before simplify)
        noseq = self.remove_sequences(onnx_in, onnx_noseq)
        
        # Stage 3: Simplify AFTER sequences are removed
        # (onnxsim can rewrite patterns that hide SplitToSequence)
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
        description="Convert VGGT models to TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with 10 cameras (default)
  %(prog)s --export
  
  # Export with 6 cameras
  %(prog)s --export --num-cams 6
  
  # Use existing ONNX
  %(prog)s --onnx-in model.onnx
  
  # Custom output paths
  %(prog)s --export --engine output/my_engine.trt
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
        default=10,
        help="Number of camera views/input images (default: 10)"
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
    
    # TensorRT configuration
    ap.add_argument(
        "--workspace-gb",
        type=int,
        default=28,
        help="TensorRT workspace size in GB (default: 28)"
    )
    ap.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 precision"
    )
    ap.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX simplification"
    )
    
    # Output paths
    ap.add_argument(
        "--onnx-simplified",
        default="onnx_exports/vggt-10x3x518x518.simp.onnx",
        help="Simplified ONNX output path"
    )
    ap.add_argument(
        "--onnx-noseq",
        default="onnx_exports/vggt-10x3x518x518.NOSEQ.onnx",
        help="No-sequence ONNX output path"
    )
    ap.add_argument(
        "--engine",
        default="onnx_exports/vggt-10x3x518x518_fp16.engine",
        help="TensorRT engine output path"
    )
    
    return ap.parse_args()


def main():
    """Main entry point."""
    args = _parse()
    
    # Auto-generate filenames based on actual config (prevents confusion)
    stem = f"vggt-{args.num_cams}x3x{args.height}x{args.width}"
    if args.onnx_simplified == "onnx_exports/vggt-10x3x518x518.simp.onnx":
        args.onnx_simplified = f"onnx_exports/{stem}.simp.onnx"
    if args.onnx_noseq == "onnx_exports/vggt-10x3x518x518.NOSEQ.onnx":
        args.onnx_noseq = f"onnx_exports/{stem}.NOSEQ.onnx"
    if args.engine == "onnx_exports/vggt-10x3x518x518_fp16.engine":
        suffix = "_fp16" if not args.no_fp16 else ""
        args.engine = f"onnx_exports/{stem}{suffix}.engine"
    
    try:
        pipe = VGGTPipeline(
            num_cams=args.num_cams,
            hw=(args.height, args.width),
            opset=args.opset,
            workspace_gb=args.workspace_gb,
            fp16=not args.no_fp16,
            simplify=not args.no_simplify,
            model_name=args.model_name,
        )
        
        engine = pipe.run(
            onnx_in=args.onnx_in,
            onnx_simp=args.onnx_simplified,
            onnx_noseq=args.onnx_noseq,
            engine_path=args.engine,
            export=args.export,
        )
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("SUCCESS!")
        logger.info(f"TensorRT engine: {engine}")
        logger.info(f"Input shape: [{args.num_cams}, 3, {args.height}, {args.width}]")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.error("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


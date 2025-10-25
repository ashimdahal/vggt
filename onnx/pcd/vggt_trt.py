"""TensorRT wrapper for VGGT engines (PCD package)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from . import trt_utils

try:
    from ..tools.trt_inference import SimpleTrtRunner  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("onnx.tools.trt_inference.SimpleTrtRunner is required") from exc


LOGGER = logging.getLogger(__name__)


def _prepare_batch(images: Sequence[np.ndarray]) -> np.ndarray:
    if len(images) == 0:
        raise ValueError("At least one image is required")
    shapes = {img.shape for img in images}
    if len(shapes) != 1:
        raise ValueError("All input images must share the same shape")
    batch = np.stack(images, axis=0)
    batch = np.transpose(batch, (0, 3, 1, 2)).astype(np.float32)
    return batch


@dataclass
class VGGTOutput:
    depth_maps: List[np.ndarray]
    raw_outputs: List[np.ndarray]


class TRTVGGT:
    def __init__(self, engine: Path, verbose: bool = False) -> None:
        self.engine_path = Path(engine)
        if not self.engine_path.exists():
            raise FileNotFoundError(self.engine_path)
        self.runner = SimpleTrtRunner(str(self.engine_path), verbose=verbose, force_sync=False)
        LOGGER.info("Loaded VGGT engine %s", self.engine_path)

    @classmethod
    def from_directory(cls, engine_dir: Path, precision: str = "auto", base_name: str = "vggt", **kwargs) -> "TRTVGGT":
        engines = trt_utils.discover_engines(engine_dir, None, base_name)
        engine = trt_utils.select_engine(engines, precision)
        return cls(engine=engine, **kwargs)

    def run(self, images: Sequence[np.ndarray]) -> VGGTOutput:
        batch = _prepare_batch(images)
        outputs = self.runner.infer(batch, copy_outputs=True)
        depth_maps: List[np.ndarray] = []
        for meta, array in zip(self.runner.output_meta, outputs):
            name = meta["name"]
            if "view" in name and array.ndim >= 4:
                depth_maps.extend(array.reshape(array.shape[0], *array.shape[2:]).astype(np.float32))
        return VGGTOutput(depth_maps=depth_maps, raw_outputs=outputs)

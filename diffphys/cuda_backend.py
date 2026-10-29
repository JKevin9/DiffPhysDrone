"""Build isolated local CUDA extensions without replacing installed modules."""

import torch
from functools import lru_cache
from pathlib import Path


@lru_cache(None)
def scene_backend():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1]
    return load(
        name="diffphys_scene_cuda",
        sources=[str(root / "src/scene_kernel.cu")],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


@lru_cache(None)
def dynamics_backend():
    try:
        import quadsim_cuda

        return quadsim_cuda
    except ModuleNotFoundError as exc:
        if exc.name != "quadsim_cuda":
            raise
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[1] / "src"
    return load(
        name="diffphys_dynamics_cuda",
        sources=[str(root / f) for f in ("quadsim.cpp", "quadsim_kernel.cu", "dynamics_kernel.cu")],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )

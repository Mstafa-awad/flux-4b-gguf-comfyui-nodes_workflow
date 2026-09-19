"""Memory-staged FLUX.2 Klein 4B GGUF nodes for ComfyUI.

This pack deliberately delegates model parsing and quantized operations to
ComfyUI-GGUF, and delegates FLUX.2 sampling math to current ComfyUI core nodes.
It only owns orchestration, validation, presets, caching, cleanup policy,
performance modes, and CPU-mode compatibility.
"""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mostafa Awad

from __future__ import annotations

import gc
import hashlib
import inspect
import logging
import os
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

import comfy.model_management as model_management
import comfy.samplers
import comfy.utils
import folder_paths
import nodes as comfy_nodes

LOG = logging.getLogger("Flux2KleinGGUFStaged")
CATEGORY = "Flux2 Klein GGUF Staged"

# -----------------------------------------------------------------------------
# Modes and policies
# -----------------------------------------------------------------------------

MODE_BALANCED = "balanced"
MODE_LOW_VRAM = "low_vram"
MODE_HIGH_SPEED = "high_speed"
MODE_CPU_ONLY = "cpu_only"

PERFORMANCE_MODES = [
    MODE_BALANCED,
    MODE_LOW_VRAM,
    MODE_HIGH_SPEED,
    MODE_CPU_ONLY,
]

CLEANUP_AUTO = "auto"
CLEANUP_ALWAYS = "always_release"
CLEANUP_KEEP = "keep_loaded"

CLEANUP_POLICIES = [
    CLEANUP_AUTO,
    CLEANUP_ALWAYS,
    CLEANUP_KEEP,
]

MEDIA_AUTO = "auto_by_mode"
MEDIA_PREFER_NORMAL = "prefer_normal"
MEDIA_ALWAYS_TILED = "always_tiled"
MEDIA_NORMAL_ONLY = "normal_only"

MEDIA_POLICIES = [
    MEDIA_AUTO,
    MEDIA_PREFER_NORMAL,
    MEDIA_ALWAYS_TILED,
    MEDIA_NORMAL_ONLY,
]

PATCH_DEVICE_POLICIES = ["auto", "true", "false"]

OFFICIAL_PRESETS: Dict[str, Tuple[int, int]] = {
    "square 1024x1024": (1024, 1024),
    "portrait 944x1104": (944, 1104),
    "portrait 880x1184": (880, 1184),
    "portrait 832x1248": (832, 1248),
    "portrait 800x1328": (800, 1328),
    "portrait 752x1392": (752, 1392),
    "portrait 720x1456": (720, 1456),
    "portrait 688x1504": (688, 1504),
    "portrait 672x1568": (672, 1568),
    "landscape 1104x944": (1104, 944),
    "landscape 1184x880": (1184, 880),
    "landscape 1248x832": (1248, 832),
    "landscape 1328x800": (1328, 800),
    "landscape 1392x752": (1392, 752),
    "landscape 1456x720": (1456, 720),
    "landscape 1504x688": (1504, 688),
    "landscape 1568x672": (1568, 672),
}


class _RuntimeState:
    def __init__(self) -> None:
        self.device_mode_applied: Optional[bool] = None
        self.cpu_threads_configured: bool = False
        self.cpu_warning_shown: bool = False
        self.gguf_fingerprint: Optional[str] = None
        self.timings: Dict[str, float] = {}


RUNTIME = _RuntimeState()


_PROMPT_CACHE_MAX_ENTRIES = 4
_REFERENCE_CACHE_MAX_ENTRIES = 8
_SCHEDULER_CACHE_MAX_ENTRIES = 64


class LRUCache:
    def __init__(self, name: str, max_entries: int):
        self.name = name
        self.max_entries = max(1, int(max_entries))
        self._data: "OrderedDict[Any, Any]" = OrderedDict()

    def get(self, key: Any) -> Any:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def set(self, key: Any, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()

    def stats(self) -> str:
        return f"{self.name}: {len(self._data)}/{self.max_entries}"


_PROMPT_CACHE = LRUCache("prompt_cache", _PROMPT_CACHE_MAX_ENTRIES)
_REFERENCE_CACHE = LRUCache("reference_cache", _REFERENCE_CACHE_MAX_ENTRIES)
_SCHEDULER_CACHE = LRUCache("scheduler_cache", _SCHEDULER_CACHE_MAX_ENTRIES)

_ACTIVE_PATCHERS: Dict[str, List[Any]] = {
    "clip": [],
    "unet": [],
    "vae": [],
}


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _unique(items: Iterable[str]) -> List[str]:
    return sorted(set(x for x in items if isinstance(x, str)))


def _filenames(keys: Sequence[str], extension: Optional[str] = None) -> List[str]:
    result: List[str] = []
    for key in keys:
        try:
            result.extend(folder_paths.get_filename_list(key))
        except Exception:
            continue

    result = _unique(result)
    if extension:
        result = [x for x in result if x.lower().endswith(extension.lower())]
    return result


def _gguf_diffusion_names() -> List[str]:
    files = _filenames(("unet_gguf", "diffusion_models", "unet"), ".gguf")
    return files or ["FLUX.2-klein-4B-Q8_0.gguf"]


def _gguf_qwen_names() -> List[str]:
    files = _filenames(("clip_gguf", "text_encoders", "clip"))
    qwen = [x for x in files if "qwen" in x.lower()]
    return qwen or files or ["Qwen3-4B-Q8_0.gguf"]


def _vae_names() -> List[str]:
    files = _filenames(("vae",))
    flux = [
        x
        for x in files
        if "flux2" in x.lower() or "flux_2" in x.lower() or "flux-2" in x.lower()
    ]
    return flux or files or ["flux2-vae.safetensors"]


def _sampler_names() -> List[str]:
    try:
        names = list(comfy.samplers.KSampler.SAMPLERS)
    except Exception:
        names = ["euler"]

    if "euler" in names:
        names.remove("euler")
    names.insert(0, "euler")
    return names


_SIGNATURE_CACHE: Dict[str, Optional[inspect.Signature]] = {}


def _signature_for(name: str, function: Any) -> Optional[inspect.Signature]:
    if name in _SIGNATURE_CACHE:
        return _SIGNATURE_CACHE[name]

    try:
        signature = inspect.signature(function)
    except Exception:
        signature = None

    _SIGNATURE_CACHE[name] = signature
    return signature


def _require_node(name: str):
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        if "GGUF" in name:
            raise RuntimeError(
                f"Required node '{name}' is missing. Install/update ComfyUI-GGUF, "
                "update ComfyUI, and restart ComfyUI."
            )
        raise RuntimeError(
            f"Required ComfyUI core node '{name}' is missing. Update ComfyUI and restart it."
        )
    return cls


def _normalize_node_output(result: Any) -> Tuple[Any, ...]:
    if hasattr(result, "args"):
        return tuple(result.args)

    if isinstance(result, dict) and "result" in result:
        wrapped = result["result"]
        return tuple(wrapped) if isinstance(wrapped, (list, tuple)) else (wrapped,)

    if isinstance(result, (list, tuple)):
        return tuple(result)

    return (result,)


def _invoke(name: str, **kwargs: Any) -> Tuple[Any, ...]:
    """Invoke a registered node and normalize legacy/V3 output containers."""
    cls = _require_node(name)
    instance = cls()
    function_name = getattr(cls, "FUNCTION")
    function = getattr(instance, function_name)

    signature = _signature_for(name, function)
    if signature is not None:
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        if not accepts_var_kwargs:
            allowed = set(signature.parameters.keys())
            kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    return _normalize_node_output(function(**kwargs))


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    return value


def _clone_conditioning(value: Any) -> Any:
    """Deep-clone tensors inside conditioning/latent containers.

    This protects cached prompt and reference data from accidental in-place
    modification by downstream nodes.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone_conditioning(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_conditioning(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_conditioning(v) for v in value)
    return value


def _round16(value: int) -> int:
    return max(64, int(round(int(value) / 16.0) * 16))


def _fit_megapixels(width: int, height: int, max_megapixels: float) -> Tuple[int, int]:
    """Preserve aspect ratio while fitting dimensions into a megapixel budget."""
    width, height = int(width), int(height)
    max_pixels = max(64 * 64, float(max_megapixels) * 1_000_000)
    scale = min(1.0, (max_pixels / max(1, width * height)) ** 0.5)

    if scale < 1.0:
        width = max(64, int(width * scale) // 16 * 16)
        height = max(64, int(height * scale) // 16 * 16)
    else:
        width, height = _round16(width), _round16(height)

    return width, height


def _resize_reference(image: Any, max_megapixels: float, method: str):
    """Downscale a BHWC Comfy image before VAE reference encoding."""
    source_width = int(image.shape[2])
    source_height = int(image.shape[1])
    width, height = _fit_megapixels(source_width, source_height, max_megapixels)

    if width == source_width and height == source_height:
        return image, width, height

    channels_first = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(channels_first, width, height, method, "disabled")
    return resized.movedim(1, -1), width, height


def _record_timing(key: str, elapsed: float) -> None:
    RUNTIME.timings[key] = float(elapsed)


# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------

def _file_path(keys: Sequence[str], name: str) -> Optional[str]:
    for key in keys:
        try:
            path = folder_paths.get_full_path(key, name)
            if path and os.path.isfile(path):
                return path
        except Exception:
            continue
    return None


def _file_size_gb(keys: Sequence[str], name: str) -> Optional[float]:
    path = _file_path(keys, name)
    if not path:
        return None
    try:
        return os.path.getsize(path) / 1_000_000_000
    except Exception:
        return None


def _file_fingerprint(keys: Sequence[str], name: str) -> str:
    path = _file_path(keys, name)
    if not path:
        return f"missing:{name}"

    try:
        stat = os.stat(path)
        return f"{os.path.basename(path)}:{stat.st_size}:{int(stat.st_mtime_ns)}"
    except Exception:
        return f"{name}:unknown"


def _gguf_runtime_fingerprint() -> str:
    if RUNTIME.gguf_fingerprint:
        return RUNTIME.gguf_fingerprint

    fingerprint = "gguf:unknown"
    for module_name in ("ComfyUI_GGUF", "comfyui_gguf"):
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", None)
            if isinstance(version, str) and version:
                fingerprint = f"{module_name}:{version}"
                break
        except Exception:
            continue

    RUNTIME.gguf_fingerprint = fingerprint
    return fingerprint


def _vae_fingerprint(vae: Any) -> str:
    for attr in ("vae_name", "loaded_ckpt_name", "ckpt_name", "filename", "file_name"):
        name = getattr(vae, attr, None)
        if isinstance(name, str) and name:
            return _file_fingerprint(("vae",), name)

    first_stage = getattr(vae, "first_stage", None)
    if first_stage is not None:
        try:
            config = getattr(first_stage, "config", None)
            if config is not None:
                return f"vae-config:{hash(str(config))}"
        except Exception:
            pass

    return "vae-unknown"


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    hasher = hashlib.sha1()
    t = tensor.detach()

    hasher.update(str(tuple(t.shape)).encode("utf-8"))
    hasher.update(str(t.dtype).encode("utf-8"))

    if t.device.type != "cpu":
        t = t.to("cpu")

    t = t.contiguous()

    try:
        arr = t.numpy()
        hasher.update(memoryview(arr))
    except Exception:
        hasher.update(t.to(torch.float32).cpu().numpy().tobytes())

    return hasher.hexdigest()


# -----------------------------------------------------------------------------
# Device / memory helpers
# -----------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _safe_empty_cache() -> None:
    try:
        if hasattr(model_management, "soft_empty_cache"):
            model_management.soft_empty_cache()
    except Exception:
        LOG.debug("soft_empty_cache failed", exc_info=True)


def _configure_cpu_threads() -> None:
    if RUNTIME.cpu_threads_configured:
        return

    RUNTIME.cpu_threads_configured = True

    cores: Optional[int] = None
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
    except Exception:
        cores = None

    if not cores:
        cores = os.cpu_count()

    threads = max(1, int(cores or 1))

    try:
        torch.set_num_threads(threads)
    except Exception:
        LOG.debug("torch.set_num_threads failed", exc_info=True)

    try:
        interop = max(1, min(4, threads // 2 or 1))
        torch.set_num_interop_threads(interop)
    except Exception:
        # PyTorch may refuse interop-thread changes after parallel work started.
        pass


def _set_comfy_cpu_state(enabled: bool) -> bool:
    cpu_enum = getattr(model_management, "CPUState", None)
    current = getattr(model_management, "cpu_state", None)

    if cpu_enum is None or current is None:
        return not enabled

    target = cpu_enum.CPU if enabled else cpu_enum.GPU
    if current == target:
        return True

    if not enabled and not _cuda_available():
        return False

    try:
        model_management.unload_all_models()
        model_management.cpu_state = target
        _safe_empty_cache()
        return True
    except Exception:
        LOG.debug("Could not switch ComfyUI CPUState", exc_info=True)
        return False


def _apply_device_mode(mode: str) -> None:
    want_cpu = mode == MODE_CPU_ONLY

    if RUNTIME.device_mode_applied == want_cpu:
        return

    if want_cpu:
        success = _set_comfy_cpu_state(True)
        _configure_cpu_threads()

        if not success and not RUNTIME.cpu_warning_shown:
            LOG.warning(
                "Could not force ComfyUI into CPU state at runtime. "
                "For reliable CPU-only execution, restart ComfyUI with --cpu."
            )
            RUNTIME.cpu_warning_shown = True

        RUNTIME.device_mode_applied = True
    else:
        if _cuda_available():
            _set_comfy_cpu_state(False)
        RUNTIME.device_mode_applied = False


def _resolve_mode(mode: str) -> str:
    resolved = str(mode or MODE_BALANCED).strip().lower()
    if resolved not in PERFORMANCE_MODES:
        resolved = MODE_BALANCED

    if resolved != MODE_CPU_ONLY and not _cuda_available():
        resolved = MODE_CPU_ONLY

    _apply_device_mode(resolved)
    return resolved


def _is_comfy_cpu_device() -> bool:
    try:
        device = model_management.get_torch_device()
        return getattr(device, "type", str(device)) == "cpu"
    except Exception:
        return False


def _gpu_total_gb() -> float:
    if not _cuda_available():
        return 0.0

    try:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
    except Exception:
        return 0.0


def _gpu_free_gb() -> float:
    if not _cuda_available():
        return 0.0

    try:
        return torch.cuda.mem_get_info()[0] / 2**30
    except Exception:
        return 0.0


def _ram_total_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().total / 2**30
    except Exception:
        return 0.0


def _ram_free_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:
        return 0.0


def _available_memory_gb(mode: str) -> float:
    if mode == MODE_CPU_ONLY or _is_comfy_cpu_device():
        return _ram_free_gb()

    if _cuda_available():
        return _gpu_free_gb()

    return _ram_free_gb()


# -----------------------------------------------------------------------------
# Patcher tracking and cleanup
# -----------------------------------------------------------------------------

def _get_patcher(obj: Any) -> Any:
    if obj is None:
        return None

    patcher = getattr(obj, "patcher", None)
    if patcher is not None:
        return patcher

    if hasattr(obj, "model") and hasattr(obj, "clone"):
        return obj

    return None


def _register_patcher(kind: str, obj: Any) -> None:
    patcher = _get_patcher(obj)
    if patcher is None:
        return

    registry = _ACTIVE_PATCHERS.setdefault(kind, [])

    for ref in registry:
        try:
            if ref() is patcher:
                return
        except Exception:
            continue

    try:
        registry.append(weakref.ref(patcher))
    except TypeError:
        # Object is not weak-referenceable; emergency cleanup can still unload all.
        pass


def _release_patcher_object(patcher: Any, empty_cache: bool = False) -> None:
    if patcher is None:
        return

    try:
        if hasattr(model_management, "unload_model_and_clones"):
            model_management.unload_model_and_clones(patcher)
        elif hasattr(model_management, "unload_model_clones"):
            model_management.unload_model_clones(patcher)
        else:
            model_management.unload_all_models()
    except Exception:
        LOG.debug("Targeted patcher release failed", exc_info=True)

    if empty_cache:
        _safe_empty_cache()


def _release_kind(kind: str, empty_cache: bool = True) -> None:
    registry = _ACTIVE_PATCHERS.get(kind)
    if not registry:
        return

    for ref in list(registry):
        try:
            patcher = ref()
        except Exception:
            patcher = None

        if patcher is not None:
            _release_patcher_object(patcher, empty_cache=False)

    registry.clear()

    if empty_cache:
        _safe_empty_cache()


def _release_all_kinds(empty_cache: bool = True) -> None:
    for kind in list(_ACTIVE_PATCHERS.keys()):
        _release_kind(kind, empty_cache=False)

    if empty_cache:
        _safe_empty_cache()


def _emergency_cleanup() -> None:
    _release_all_kinds(empty_cache=False)

    try:
        model_management.unload_all_models()
    except Exception:
        LOG.debug("unload_all_models failed during emergency cleanup", exc_info=True)

    _safe_empty_cache()
    gc.collect()


def _is_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True

    text = str(exc).lower()
    markers = (
        "out of memory",
        "oom",
        "bad_alloc",
        "cannot allocate",
        "can't allocate",
        "cuda memory",
        "cpu memory",
        "not enough memory",
    )
    return any(marker in text for marker in markers)


def _should_release_clip(mode: str, policy: str) -> bool:
    if policy == CLEANUP_ALWAYS:
        return True
    if policy == CLEANUP_KEEP:
        return False
    return mode != MODE_HIGH_SPEED


def _pre_sampling_cleanup(mode: str, policy: str) -> None:
    if policy == CLEANUP_KEEP:
        return

    _release_kind("clip", empty_cache=False)

    if mode != MODE_HIGH_SPEED or policy == CLEANUP_ALWAYS:
        _release_kind("vae", empty_cache=False)

    _safe_empty_cache()

    if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY) or policy == CLEANUP_ALWAYS:
        gc.collect()


def _pre_reference_cleanup(mode: str, policy: str) -> None:
    if policy == CLEANUP_KEEP:
        return

    _release_kind("clip", empty_cache=False)

    if mode != MODE_HIGH_SPEED or policy == CLEANUP_ALWAYS:
        _release_kind("unet", empty_cache=False)

    _safe_empty_cache()

    if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY) or policy == CLEANUP_ALWAYS:
        gc.collect()


def _post_reference_cleanup(mode: str, policy: str) -> None:
    if policy == CLEANUP_KEEP:
        return

    _release_kind("vae", empty_cache=False)

    if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY) or policy == CLEANUP_ALWAYS:
        _release_kind("clip", empty_cache=False)
        _release_kind("unet", empty_cache=False)
    elif mode == MODE_BALANCED:
        _release_kind("clip", empty_cache=False)

    _safe_empty_cache()

    if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY) or policy == CLEANUP_ALWAYS:
        gc.collect()


def _keep_diffusion_after_decode(mode: str, policy: str) -> bool:
    if policy == CLEANUP_KEEP:
        return True
    if policy == CLEANUP_ALWAYS:
        return False
    return mode == MODE_HIGH_SPEED


def _pre_decode_cleanup(mode: str, policy: str, keep_diffusion: bool) -> None:
    if policy == CLEANUP_KEEP:
        return

    _release_kind("clip", empty_cache=False)

    if not keep_diffusion:
        _release_kind("unet", empty_cache=False)

    _safe_empty_cache()

    if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY) or policy == CLEANUP_ALWAYS:
        gc.collect()


def _post_decode_cleanup(mode: str, policy: str) -> None:
    if policy == CLEANUP_KEEP:
        return

    if mode == MODE_HIGH_SPEED and policy == CLEANUP_AUTO:
        _release_kind("clip", empty_cache=False)
    else:
        _release_kind("unet", empty_cache=False)
        _release_kind("vae", empty_cache=False)
        _release_kind("clip", empty_cache=False)

    _safe_empty_cache()

    if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY) or policy == CLEANUP_ALWAYS:
        gc.collect()


# -----------------------------------------------------------------------------
# VAE strategy helpers
# -----------------------------------------------------------------------------

def _mode_tile_size(mode: str, kind: str) -> int:
    if mode in (MODE_LOW_VRAM, MODE_CPU_ONLY):
        return 512
    if mode == MODE_BALANCED:
        return 768
    if mode == MODE_HIGH_SPEED:
        return 1024
    return 512


def _sanitize_tile(tile_size: int, overlap: int) -> Tuple[int, int]:
    tile = int(tile_size) if tile_size and int(tile_size) > 0 else 512
    tile = max(256, min(2048, tile))

    ov = int(overlap) if overlap is not None else 0
    ov = max(0, min(512, ov))

    max_overlap = max(0, tile // 2)
    if ov > max_overlap:
        ov = max_overlap

    return tile, ov


def _normal_decode_ok(mode: str, mp: float, free_gb: float, tile_trigger: float) -> bool:
    if free_gb <= 0.0:
        return False

    trigger = max(0.01, float(tile_trigger))

    if mode == MODE_CPU_ONLY:
        return mp <= min(0.5, trigger) and free_gb >= 8.0 + mp * 8.0

    if mode == MODE_LOW_VRAM:
        return False

    if mode == MODE_BALANCED:
        return mp < trigger and free_gb >= 5.0 + mp * 2.5

    if mode == MODE_HIGH_SPEED:
        return free_gb >= 3.0 + mp * 2.0

    return False


def _normal_encode_ok(mode: str, mp: float, free_gb: float) -> bool:
    if free_gb <= 0.0:
        return False

    if mode == MODE_CPU_ONLY:
        return mp <= 0.5 and free_gb >= 6.0 + mp * 6.0

    if mode == MODE_LOW_VRAM:
        return False

    if mode == MODE_BALANCED:
        return free_gb >= 4.0 + mp * 2.0

    if mode == MODE_HIGH_SPEED:
        return free_gb >= 2.5 + mp * 1.5

    return False


def _plan_media(
    media_policy: str,
    mode: str,
    mp: float,
    free_gb: float,
    tile_size: int,
    overlap: int,
    kind: str,
    tile_trigger: float = 1.0,
) -> Tuple[str, int, int]:
    policy = str(media_policy or MEDIA_AUTO)

    if policy == MEDIA_NORMAL_ONLY:
        strategy = "normal"
    elif policy == MEDIA_ALWAYS_TILED:
        strategy = "tiled"
    elif policy == MEDIA_PREFER_NORMAL:
        strategy = "normal"
    else:
        if kind == "decode":
            strategy = "normal" if _normal_decode_ok(mode, mp, free_gb, tile_trigger) else "tiled"
        else:
            strategy = "normal" if _normal_encode_ok(mode, mp, free_gb) else "tiled"

    resolved_tile = int(tile_size) if tile_size and int(tile_size) > 0 else _mode_tile_size(mode, kind)
    tile, ov = _sanitize_tile(resolved_tile, overlap)
    return strategy, tile, ov


def _dedupe_attempts(attempts: Sequence[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    seen = set()
    result: List[Tuple[str, int, int]] = []

    for attempt in attempts:
        if attempt not in seen:
            seen.add(attempt)
            result.append(attempt)

    return result


def _build_media_attempts(
    strategy: str,
    tile: int,
    overlap: int,
    fallback: bool,
) -> List[Tuple[str, int, int]]:
    attempts: List[Tuple[str, int, int]] = []

    if strategy == "normal":
        attempts.append(("normal", tile, overlap))

        if fallback:
            attempts.append(("tiled", tile, overlap))

            if tile > 512:
                half = max(256, tile // 2)
                attempts.append(("tiled", half, min(overlap, half // 2)))

            if tile > 256:
                attempts.append(("tiled", 256, min(overlap, 128)))
    else:
        attempts.append(("tiled", tile, overlap))

        if fallback:
            if tile > 512:
                half = max(256, tile // 2)
                attempts.append(("tiled", half, min(overlap, half // 2)))

            if tile > 256:
                attempts.append(("tiled", 256, min(overlap, 128)))

    return _dedupe_attempts(attempts)


def _latent_estimate(samples: Any, vae: Any) -> Tuple[Any, int, int, float]:
    latent = samples["samples"] if isinstance(samples, dict) and "samples" in samples else samples

    compression = 8
    try:
        if hasattr(vae, "spacial_compression_decode"):
            compression = int(vae.spacial_compression_decode())
    except Exception:
        compression = 8

    width = 0
    height = 0

    if isinstance(latent, torch.Tensor):
        width = int(latent.shape[-1]) * compression
        height = int(latent.shape[-2]) * compression

    mp = (width * height) / 1_000_000 if width and height else 0.0
    return latent, width, height, mp


def _plan_encode(
    image: Any,
    vae: Any,
    mode: str,
    media_policy: str,
    tile_size: int,
    overlap: int,
) -> Tuple[str, int, int, float, float]:
    height = int(image.shape[1])
    width = int(image.shape[2])
    mp = (width * height) / 1_000_000
    free_gb = _available_memory_gb(mode)

    strategy, tile, ov = _plan_media(
        media_policy,
        mode,
        mp,
        free_gb,
        tile_size,
        overlap,
        "encode",
    )

    return strategy, tile, ov, mp, free_gb


def _execute_vae_encode(
    image: Any,
    vae: Any,
    strategy: str,
    tile_size: int,
    overlap: int,
    media_policy: str = MEDIA_AUTO,
    use_fallback: bool = True,
) -> Tuple[Any, str, int, int, bool]:
    _register_patcher("vae", vae)

    fallback = bool(use_fallback) and media_policy != MEDIA_NORMAL_ONLY
    attempts = _build_media_attempts(strategy, tile_size, overlap, fallback)
    last_exc: Optional[BaseException] = None

    for index, (attempt_strategy, tile, ov) in enumerate(attempts):
        try:
            if attempt_strategy == "normal":
                latent = _invoke("VAEEncode", pixels=image, vae=vae)[0]
            else:
                latent = _invoke(
                    "VAEEncodeTiled",
                    pixels=image,
                    vae=vae,
                    tile_size=tile,
                    overlap=ov,
                    temporal_size=64,
                    temporal_overlap=8,
                )[0]

            planned_ok = attempt_strategy == strategy and (
                attempt_strategy == "normal" or (tile == tile_size and ov == overlap)
            )

            return latent, attempt_strategy, tile, ov, planned_ok

        except Exception as exc:
            last_exc = exc
            has_next_attempt = index < len(attempts) - 1

            if has_next_attempt and _is_memory_error(exc):
                _emergency_cleanup()
                continue

            raise

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("VAE encode failed.")


def _execute_vae_decode(
    samples: Any,
    vae: Any,
    mode: str,
    media_policy: str,
    tile_size: int,
    overlap: int,
    tile_trigger: float,
    use_fallback: bool,
    width: int,
    height: int,
    mp: float,
) -> Tuple[Any, str, bool]:
    _register_patcher("vae", vae)

    free_gb = _available_memory_gb(mode)

    strategy, tile, ov = _plan_media(
        media_policy,
        mode,
        mp,
        free_gb,
        tile_size,
        overlap,
        "decode",
        tile_trigger=tile_trigger,
    )

    fallback = bool(use_fallback) and media_policy != MEDIA_NORMAL_ONLY
    attempts = _build_media_attempts(strategy, tile, ov, fallback)
    last_exc: Optional[BaseException] = None

    for index, (attempt_strategy, attempt_tile, attempt_overlap) in enumerate(attempts):
        try:
            if attempt_strategy == "normal":
                image = _invoke("VAEDecode", samples=samples, vae=vae)[0]
                method = "normal"
            else:
                image = _invoke(
                    "VAEDecodeTiled",
                    samples=samples,
                    vae=vae,
                    tile_size=attempt_tile,
                    overlap=attempt_overlap,
                    temporal_size=64,
                    temporal_overlap=8,
                )[0]
                method = f"tiled {attempt_tile}px / {attempt_overlap}px overlap"

            planned_ok = attempt_strategy == strategy and (
                attempt_strategy == "normal"
                or (attempt_tile == tile and attempt_overlap == ov)
            )

            if not planned_ok:
                method += " (memory fallback)"

            return image, method, planned_ok

        except Exception as exc:
            last_exc = exc
            has_next_attempt = index < len(attempts) - 1

            if has_next_attempt and _is_memory_error(exc):
                _emergency_cleanup()
                continue

            raise

    if last_exc is not None:
        raise last_exc

    raise RuntimeError("VAE decode failed.")


def _encode_one_reference(
    source_image: Any,
    vae: Any,
    mode: str,
    per_reference_budget: float,
    resize_method: str,
    media_policy: str,
    tile_size: int,
    overlap: int,
    use_cache: bool,
    use_fallback: bool,
) -> Tuple[Any, int, int, str, bool]:
    if mode == MODE_CPU_ONLY:
        source_image = _to_cpu(source_image)

    resized, encoded_width, encoded_height = _resize_reference(
        source_image,
        per_reference_budget,
        resize_method,
    )

    strategy, tile, ov, _mp, _free = _plan_encode(
        resized,
        vae,
        mode,
        media_policy,
        tile_size,
        overlap,
    )

    cache_key: Optional[Tuple[Any, ...]] = None
    vae_ref: Optional[Any] = None

    if use_cache:
        try:
            vae_ref = weakref.ref(vae)
        except TypeError:
            vae_ref = None

        if vae_ref is not None:
            source_fingerprint = _tensor_fingerprint(source_image)

            cache_key = (
                source_fingerprint,
                id(vae),
                _vae_fingerprint(vae),
                round(float(per_reference_budget), 5),
                str(resize_method),
                mode,
                media_policy,
                strategy,
                int(tile),
                int(ov),
                _gguf_runtime_fingerprint(),
            )

            entry = _REFERENCE_CACHE.get(cache_key)
            if entry is not None:
                entry_vae_ref, cached_latent, cached_width, cached_height, cached_strategy = entry

                try:
                    vae_matches = entry_vae_ref() is vae
                except Exception:
                    vae_matches = False

                if vae_matches and cached_strategy == strategy:
                    return (
                        _clone_conditioning(cached_latent),
                        int(cached_width),
                        int(cached_height),
                        cached_strategy,
                        True,
                    )

    latent, actual_strategy, actual_tile, actual_overlap, planned_ok = _execute_vae_encode(
        resized,
        vae,
        strategy,
        tile,
        ov,
        media_policy=media_policy,
        use_fallback=use_fallback,
    )

    latent_cpu = _to_cpu(latent)

    if use_cache and cache_key is not None and vae_ref is not None and planned_ok:
        _REFERENCE_CACHE.set(
            cache_key,
            (
                vae_ref,
                _clone_conditioning(latent_cpu),
                encoded_width,
                encoded_height,
                actual_strategy,
            ),
        )

    return latent_cpu, encoded_width, encoded_height, actual_strategy, False


def _get_sigmas(steps: int, width: int, height: int) -> Any:
    key = (int(steps), int(width), int(height))

    cached = _SCHEDULER_CACHE.get(key)
    if cached is not None:
        return cached.clone() if isinstance(cached, torch.Tensor) else cached

    sigmas = _invoke(
        "Flux2Scheduler",
        steps=int(steps),
        width=int(width),
        height=int(height),
    )[0]

    if isinstance(sigmas, torch.Tensor):
        _SCHEDULER_CACHE.set(key, sigmas.detach().clone())

    return sigmas


def _resolve_patch_on_device(policy: str, mode: str, file_size_gb: Optional[float]) -> bool:
    if mode == MODE_CPU_ONLY:
        return False

    if policy == "true":
        return True

    if policy == "false":
        return False

    if mode == MODE_HIGH_SPEED and _cuda_available():
        total_gb = _gpu_total_gb()
        if total_gb >= 12.0:
            if file_size_gb is None or total_gb >= (float(file_size_gb) * 2.0 + 4.0):
                return True

    return False


# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------

class Flux2KleinGGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "performance_mode": (PERFORMANCE_MODES, {"default": MODE_BALANCED}),
                "flux_gguf": (_gguf_diffusion_names(),),
                "dequant_dtype": (
                    ["target", "default", "float16", "bfloat16", "float32"],
                    {"default": "target"},
                ),
                "patch_dtype": (
                    ["default", "target", "float16", "bfloat16", "float32"],
                    {"default": "default"},
                ),
                "patch_on_device_policy": (PATCH_DEVICE_POLICIES, {"default": "auto"}),
                "cpu_force_float32": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "load_info")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Loads FLUX.2 Klein 4B GGUF through ComfyUI-GGUF. Performance mode controls "
        "cleanup and patch placement. Sampling quality settings are unchanged."
    )

    def load(
        self,
        performance_mode=MODE_BALANCED,
        flux_gguf="",
        dequant_dtype="target",
        patch_dtype="default",
        patch_on_device_policy="auto",
        cpu_force_float32=True,
    ):
        mode = _resolve_mode(performance_mode)

        if not flux_gguf:
            raise ValueError("No FLUX GGUF file selected or found.")

        low = flux_gguf.lower()

        if "base" in low:
            raise ValueError(
                "This pack is tuned for the distilled 4-step Klein checkpoint, not the Base checkpoint."
            )

        if any(token in low for token in ("9b", "12b", "14b", "32b")):
            raise ValueError(
                "Select FLUX.2 Klein 4B. Larger FLUX.2 checkpoints do not fit this low-VRAM workflow."
            )

        keys = ("unet_gguf", "diffusion_models", "unet")
        size = _file_size_gb(keys, flux_gguf)

        selected_dequant = dequant_dtype
        selected_patch = patch_dtype

        if mode == MODE_CPU_ONLY and cpu_force_float32:
            selected_dequant = "float32"
            selected_patch = "float32"

        patch_on_device = _resolve_patch_on_device(patch_on_device_policy, mode, size)

        started = time.perf_counter()

        model = _invoke(
            "UnetLoaderGGUFAdvanced",
            unet_name=flux_gguf,
            dequant_dtype=selected_dequant,
            patch_dtype=selected_patch,
            patch_on_device=patch_on_device,
        )[0]

        _register_patcher("unet", model)

        elapsed = time.perf_counter() - started
        _record_timing("flux_load", elapsed)

        size_text = f"{size:.2f} GB" if size is not None else "size unavailable"
        info = (
            f"FLUX stage ready: {flux_gguf} ({size_text}); mode={mode}; "
            f"dequant={selected_dequant}; patch={selected_patch}; "
            f"patch_on_device={patch_on_device}; load={elapsed:.2f}s."
        )

        return model, info


class Flux2StagedQwenEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "performance_mode": (PERFORMANCE_MODES, {"default": MODE_BALANCED}),
                "qwen_gguf": (_gguf_qwen_names(),),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "default": "A cinematic photograph with natural light and fine detail",
                    },
                ),
                "use_prompt_cache": ("BOOLEAN", {"default": True}),
                "force_gpu_stage": ("BOOLEAN", {"default": True}),
                "cleanup_policy": (CLEANUP_POLICIES, {"default": CLEANUP_AUTO}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative_zeroed", "stage_info")
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Encodes Qwen3 4B, moves conditioning to CPU, and releases the encoder according "
        "to the selected performance mode. Exact prompt caching avoids recomputation."
    )

    def encode(
        self,
        performance_mode=MODE_BALANCED,
        qwen_gguf="",
        prompt="",
        use_prompt_cache=True,
        force_gpu_stage=True,
        cleanup_policy=CLEANUP_AUTO,
    ):
        mode = _resolve_mode(performance_mode)

        if not qwen_gguf:
            raise ValueError("No Qwen GGUF text encoder selected or found.")

        if not prompt.strip():
            raise ValueError("The prompt is empty.")

        low = qwen_gguf.lower()

        if "qwen" not in low:
            LOG.warning("Selected text encoder does not include 'Qwen' in its filename: %s", qwen_gguf)

        if any(token in low for token in ("8b", "14b", "30b", "32b")):
            raise ValueError("FLUX.2 Klein 4B requires Qwen3 4B. Do not use the 8B or larger encoder.")

        keys = ("clip_gguf", "text_encoders", "clip")
        qwen_fingerprint = _file_fingerprint(keys, qwen_gguf)
        size = _file_size_gb(keys, qwen_gguf)
        size_text = f"{size:.2f} GB" if size is not None else "size unavailable"

        resolved_force_gpu = bool(force_gpu_stage) and mode != MODE_CPU_ONLY and _cuda_available()

        cache_key = None
        if use_prompt_cache:
            cache_key = (
                prompt,
                qwen_gguf,
                qwen_fingerprint,
                mode,
                resolved_force_gpu,
                _gguf_runtime_fingerprint(),
            )

            cached = _PROMPT_CACHE.get(cache_key)
            if cached is not None:
                cached_positive, cached_negative, cached_elapsed = cached
                _record_timing("qwen_encode", 0.0)

                info = (
                    f"Qwen exact cache hit: {qwen_gguf} ({size_text}); mode={mode}; "
                    f"original encode={cached_elapsed:.2f}s. No recomputation and no quality change."
                )

                return (
                    _clone_conditioning(cached_positive),
                    _clone_conditioning(cached_negative),
                    info,
                )

        started = time.perf_counter()

        clip = _invoke("CLIPLoaderGGUF", clip_name=qwen_gguf, type="flux2")[0]
        _register_patcher("clip", clip)
        patcher = _get_patcher(clip)

        try:
            if resolved_force_gpu and patcher is not None:
                try:
                    model_management.load_models_gpu([patcher])
                except Exception as exc:
                    LOG.warning(
                        "Qwen GPU staging failed; continuing with ComfyUI default placement: %s",
                        exc,
                    )

            positive = _invoke("CLIPTextEncode", clip=clip, text=prompt)[0]
            positive = _to_cpu(positive)

            negative = _invoke("ConditioningZeroOut", conditioning=positive)[0]
            negative = _to_cpu(negative)

        finally:
            if _should_release_clip(mode, cleanup_policy):
                _release_patcher_object(patcher, empty_cache=False)
                _release_kind("clip", empty_cache=False)
                _safe_empty_cache()
                gc.collect()

            del clip

        elapsed = time.perf_counter() - started
        _record_timing("qwen_encode", elapsed)

        if cache_key is not None:
            _PROMPT_CACHE.set(
                cache_key,
                (
                    _clone_conditioning(positive),
                    _clone_conditioning(negative),
                    elapsed,
                ),
            )

        kept = not _should_release_clip(mode, cleanup_policy)
        placement = "CPU" if mode == MODE_CPU_ONLY else "GPU staged" if resolved_force_gpu else "ComfyUI default"

        info = (
            f"Qwen encoded: {qwen_gguf} ({size_text}); mode={mode}; placement={placement}; "
            f"kept_loaded={kept}; encode={elapsed:.2f}s. Conditioning is stored on CPU."
        )

        return positive, negative, info


class Flux2KleinCanvas:
    PRESETS = list(OFFICIAL_PRESETS.keys()) + ["custom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "performance_mode": (PERFORMANCE_MODES, {"default": MODE_BALANCED}),
                "preset": (cls.PRESETS, {"default": "square 1024x1024"}),
                "custom_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "custom_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8}),
            },
            "optional": {
                "width_override": ("INT", {"forceInput": True}),
                "height_override": ("INT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING")
    RETURN_NAMES = ("latent", "width", "height", "size_info")
    FUNCTION = "make"
    CATEGORY = CATEGORY
    DESCRIPTION = "Creates the official FLUX.2 Klein canvas latent and warns when the selected mode is under memory pressure."

    def make(
        self,
        performance_mode=MODE_BALANCED,
        preset="square 1024x1024",
        custom_width=1024,
        custom_height=1024,
        batch_size=1,
        width_override=None,
        height_override=None,
    ):
        mode = _resolve_mode(performance_mode)

        if width_override is not None and height_override is not None:
            width, height = _round16(width_override), _round16(height_override)
        elif preset == "custom":
            width, height = _round16(custom_width), _round16(custom_height)
        else:
            width, height = OFFICIAL_PRESETS.get(preset, (1024, 1024))

        latent = _invoke(
            "EmptyFlux2LatentImage",
            width=width,
            height=height,
            batch_size=batch_size,
        )[0]

        if mode == MODE_CPU_ONLY:
            latent = _to_cpu(latent)

        mp = (width * height) / 1_000_000
        warning = ""

        if mode == MODE_CPU_ONLY:
            if mp > 0.25:
                warning = " CPU mode: this is large and may be extremely slow or memory-heavy."
        elif mode == MODE_LOW_VRAM:
            if mp > 1.0:
                warning = " Low-VRAM mode: tiled VAE and aggressive cleanup will be used."
        elif mode == MODE_BALANCED:
            if mp > 1.15:
                warning = " Balanced mode: above the recommended 8GB starting area; memory fallback may be used."
        elif mode == MODE_HIGH_SPEED:
            if mp > 2.0:
                warning = " High-speed mode: large area while keeping models loaded may OOM; use balanced if it fails."

        if batch_size > 1:
            warning += f" Batch size {batch_size} multiplies memory and time."

        info = f"{width}x{height}, batch {batch_size}, {mp:.2f} MP, mode={mode}.{warning}"
        return latent, width, height, info


class Flux2KleinDistilledSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "performance_mode": (PERFORMANCE_MODES, {"default": MODE_BALANCED}),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative_zeroed": ("CONDITIONING",),
                "latent": ("LATENT",),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "seed": (
                    "INT",
                    {
                        "default": 43,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "steps": ("INT", {"default": 4, "min": 1, "max": 12}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "sampler_name": (_sampler_names(), {"default": "euler"}),
                "cleanup_policy": (CLEANUP_POLICIES, {"default": CLEANUP_AUTO}),
                "retry_after_oom": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("samples", "denoised_samples", "sample_info")
    FUNCTION = "sample"
    CATEGORY = CATEGORY
    DESCRIPTION = "Official FLUX.2 Klein distilled path: Euler + Flux2Scheduler, 4 steps, CFG 1."

    def sample(
        self,
        performance_mode=MODE_BALANCED,
        model=None,
        positive=None,
        negative_zeroed=None,
        latent=None,
        width=1024,
        height=1024,
        seed=43,
        steps=4,
        cfg=1.0,
        sampler_name="euler",
        cleanup_policy=CLEANUP_AUTO,
        retry_after_oom=True,
    ):
        mode = _resolve_mode(performance_mode)

        width = _round16(int(width))
        height = _round16(int(height))

        if mode == MODE_CPU_ONLY:
            positive = _to_cpu(positive)
            negative_zeroed = _to_cpu(negative_zeroed)
            latent = _to_cpu(latent)

        started = time.perf_counter()

        _pre_sampling_cleanup(mode, cleanup_policy)
        _register_patcher("unet", model)

        if steps != 4:
            LOG.warning("FLUX.2 Klein Distilled is validated at 4 steps; selected %s.", steps)

        if abs(float(cfg) - 1.0) > 0.001:
            LOG.warning("FLUX.2 Klein Distilled is validated at CFG 1.0; selected %s.", cfg)

        if sampler_name != "euler":
            LOG.warning("Euler is the validated Klein sampler; selected %s.", sampler_name)

        sigmas = _get_sigmas(steps, width, height)
        if mode == MODE_CPU_ONLY:
            sigmas = _to_cpu(sigmas)

        noise = _invoke("RandomNoise", noise_seed=seed)[0]
        sampler = _invoke("KSamplerSelect", sampler_name=sampler_name)[0]

        used_memory_retry = False

        guider = _invoke(
            "CFGGuider",
            model=model,
            positive=positive,
            negative=negative_zeroed,
            cfg=float(cfg),
        )[0]

        try:
            result = _invoke(
                "SamplerCustomAdvanced",
                noise=noise,
                guider=guider,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent,
            )
        except Exception as exc:
            if retry_after_oom and _is_memory_error(exc):
                _emergency_cleanup()
                _register_patcher("unet", model)
                used_memory_retry = True

                guider = _invoke(
                    "CFGGuider",
                    model=model,
                    positive=positive,
                    negative=negative_zeroed,
                    cfg=float(cfg),
                )[0]

                result = _invoke(
                    "SamplerCustomAdvanced",
                    noise=noise,
                    guider=guider,
                    sampler=sampler,
                    sigmas=sigmas,
                    latent_image=latent,
                )
            else:
                raise

        if len(result) < 1:
            raise RuntimeError("SamplerCustomAdvanced returned no output.")

        output = result[0]
        denoised = result[1] if len(result) > 1 else output

        elapsed = time.perf_counter() - started
        _record_timing("sampling", elapsed)

        info = (
            f"{width}x{height}; {steps} steps; CFG {float(cfg):g}; {sampler_name}; seed {seed}; "
            f"mode={mode}; sample={elapsed:.2f}s."
        )

        if used_memory_retry:
            info += " Memory fallback retry was used after an OOM error."

        return output, denoised, info


class Flux2ReferenceStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "performance_mode": (PERFORMANCE_MODES, {"default": MODE_BALANCED}),
                "positive": ("CONDITIONING",),
                "negative_zeroed": ("CONDITIONING",),
                "vae": ("VAE",),
                "reference_1": ("IMAGE",),
                "output_size": (["first reference", "custom"], {"default": "first reference"}),
                "custom_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "custom_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "memory_budget_megapixels": ("FLOAT", {"default": 1.05, "min": 0.25, "max": 4.0, "step": 0.05}),
                "enforce_8gb_limit": ("BOOLEAN", {"default": True}),
                "resize_method": (["area", "lanczos", "bicubic", "bilinear"], {"default": "area"}),
                "encode_policy": (MEDIA_POLICIES, {"default": MEDIA_AUTO}),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 32}),
                "use_reference_cache": ("BOOLEAN", {"default": True}),
                "use_oom_fallback": ("BOOLEAN", {"default": True}),
                "cleanup_policy": (CLEANUP_POLICIES, {"default": CLEANUP_AUTO}),
            },
            "optional": {
                "reference_2": ("IMAGE",),
                "reference_3": ("IMAGE",),
                "reference_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "INT", "INT", "STRING")
    RETURN_NAMES = ("positive", "negative_zeroed", "width", "height", "reference_info")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Supports 1-4 reference images, fits them into a total pixel budget, VAE-encodes them, "
        "applies ReferenceLatent to both positive and negative conditioning, then cleans up by mode."
    )

    def apply(
        self,
        performance_mode=MODE_BALANCED,
        positive=None,
        negative_zeroed=None,
        vae=None,
        reference_1=None,
        output_size="first reference",
        custom_width=1024,
        custom_height=1024,
        memory_budget_megapixels=1.05,
        enforce_8gb_limit=True,
        resize_method="area",
        encode_policy=MEDIA_AUTO,
        tile_size=0,
        overlap=64,
        use_reference_cache=True,
        use_oom_fallback=True,
        cleanup_policy=CLEANUP_AUTO,
        reference_2=None,
        reference_3=None,
        reference_4=None,
    ):
        mode = _resolve_mode(performance_mode)

        refs = [x for x in (reference_1, reference_2, reference_3, reference_4) if x is not None]
        if not refs:
            raise ValueError("At least one reference image is required.")

        if mode == MODE_CPU_ONLY:
            refs = [_to_cpu(x) for x in refs]
            positive = _to_cpu(positive)
            negative_zeroed = _to_cpu(negative_zeroed)

        source_sizes = [(int(image.shape[2]), int(image.shape[1])) for image in refs]

        if enforce_8gb_limit:
            per_reference_budget = float(memory_budget_megapixels) / len(refs)
        else:
            per_reference_budget = 1_000_000.0

        started = time.perf_counter()

        _pre_reference_cleanup(mode, cleanup_policy)
        _register_patcher("vae", vae)

        pos = _clone_conditioning(positive)
        neg = _clone_conditioning(negative_zeroed)

        encoded_sizes: List[Tuple[int, int]] = []
        strategies: List[str] = []
        cache_hits = 0

        try:
            for image in refs:
                latent, encoded_width, encoded_height, strategy, cache_hit = _encode_one_reference(
                    image,
                    vae,
                    mode,
                    per_reference_budget,
                    resize_method,
                    encode_policy,
                    int(tile_size),
                    int(overlap),
                    bool(use_reference_cache),
                    bool(use_oom_fallback),
                )

                encoded_sizes.append((encoded_width, encoded_height))
                strategies.append(strategy)
                cache_hits += 1 if cache_hit else 0

                pos = _invoke("ReferenceLatent", conditioning=pos, latent=latent)[0]
                neg = _invoke("ReferenceLatent", conditioning=neg, latent=latent)[0]

        finally:
            _post_reference_cleanup(mode, cleanup_policy)

        if output_size == "first reference":
            width, height = source_sizes[0]
        else:
            width, height = int(custom_width), int(custom_height)

        if enforce_8gb_limit:
            width, height = _fit_megapixels(width, height, float(memory_budget_megapixels))
        else:
            width, height = _round16(width), _round16(height)

        elapsed = time.perf_counter() - started
        _record_timing("reference_stack", elapsed)

        encoded_text = ", ".join(
            f"{sw}x{sh}->{ew}x{eh}({strategy})"
            for (sw, sh), (ew, eh), strategy in zip(source_sizes, encoded_sizes, strategies)
        )

        info = (
            f"Applied {len(refs)} reference image(s); mode={mode}; cache_hits={cache_hits}; "
            f"reference resize/encode: {encoded_text}; output {width}x{height}; "
            f"total budget {float(memory_budget_megapixels):.2f} MP; time={elapsed:.2f}s."
        )

        return pos, neg, width, height, info


class Flux2AutoVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "performance_mode": (PERFORMANCE_MODES, {"default": MODE_BALANCED}),
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "decode_policy": (MEDIA_POLICIES, {"default": MEDIA_AUTO}),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 32}),
                "tile_trigger_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 8.0, "step": 0.05}),
                "use_oom_fallback": ("BOOLEAN", {"default": True}),
                "cleanup_policy": (CLEANUP_POLICIES, {"default": CLEANUP_AUTO}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "decode_info")
    FUNCTION = "decode"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Decodes the final latent. Balanced and high-speed modes prefer normal decode when memory allows, "
        "then fall back safely to tiled decode if needed."
    )

    def decode(
        self,
        performance_mode=MODE_BALANCED,
        samples=None,
        vae=None,
        decode_policy=MEDIA_AUTO,
        tile_size=0,
        overlap=64,
        tile_trigger_megapixels=1.0,
        use_oom_fallback=True,
        cleanup_policy=CLEANUP_AUTO,
    ):
        mode = _resolve_mode(performance_mode)

        latent, width, height, mp = _latent_estimate(samples, vae)

        if mode == MODE_CPU_ONLY:
            samples = _to_cpu(samples)

        keep_diffusion = _keep_diffusion_after_decode(mode, cleanup_policy)

        _pre_decode_cleanup(mode, cleanup_policy, keep_diffusion)
        _register_patcher("vae", vae)

        started = time.perf_counter()

        image, method, planned_ok = _execute_vae_decode(
            samples=samples,
            vae=vae,
            mode=mode,
            media_policy=decode_policy,
            tile_size=int(tile_size),
            overlap=int(overlap),
            tile_trigger=float(tile_trigger_megapixels),
            use_fallback=bool(use_oom_fallback),
            width=width,
            height=height,
            mp=mp,
        )

        elapsed = time.perf_counter() - started
        _record_timing("vae_decode", elapsed)

        _post_decode_cleanup(mode, cleanup_policy)

        info = (
            f"Decoded {width}x{height} using {method}; mode={mode}; "
            f"keep_diffusion={keep_diffusion}; decode={elapsed:.2f}s."
        )

        if not planned_ok:
            info += " A memory-safe fallback path was used."

        return image, info


class Flux2MemoryReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "performance_mode": (PERFORMANCE_MODES, {"default": MODE_BALANCED}),
                "qwen_gguf": (_gguf_qwen_names(),),
                "flux_gguf": (_gguf_diffusion_names(),),
                "vae_name": (_vae_names(),),
                "clear_caches": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "report"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True
    DESCRIPTION = "Reports resolved runtime mode, memory, model file sizes, active tracked models, cache usage, and stage timings."

    def report(
        self,
        performance_mode=MODE_BALANCED,
        qwen_gguf="",
        flux_gguf="",
        vae_name="",
        clear_caches=False,
    ):
        selected_mode = str(performance_mode or MODE_BALANCED)
        mode = _resolve_mode(selected_mode)

        cleared_text = "no"
        if clear_caches:
            _PROMPT_CACHE.clear()
            _REFERENCE_CACHE.clear()
            _SCHEDULER_CACHE.clear()
            cleared_text = "yes"

        qwen = _file_size_gb(("clip_gguf", "text_encoders", "clip"), qwen_gguf)
        flux = _file_size_gb(("unet_gguf", "diffusion_models", "unet"), flux_gguf)
        vae = _file_size_gb(("vae",), vae_name)

        gpu_total = _gpu_total_gb()
        gpu_free = _gpu_free_gb()
        ram_total = _ram_total_gb()
        ram_free = _ram_free_gb()

        def fmt(value: Optional[float]) -> str:
            return f"{value:.2f} GB" if value is not None else "not found"

        active_parts: List[str] = []
        for kind in ("clip", "unet", "vae"):
            alive = 0
            for ref in _ACTIVE_PATCHERS.get(kind, []):
                try:
                    if ref() is not None:
                        alive += 1
                except Exception:
                    continue
            active_parts.append(f"{kind}={alive}")

        active_text = ", ".join(active_parts)
        cache_text = ", ".join(
            cache.stats() for cache in (_PROMPT_CACHE, _REFERENCE_CACHE, _SCHEDULER_CACHE)
        )

        if RUNTIME.timings:
            timing_text = ", ".join(
                f"{key}={value:.2f}s"
                for key, value in sorted(RUNTIME.timings.items())
            )
        else:
            timing_text = "No stage timings recorded yet."

        lines = [
            "FLUX.2 Klein staged memory report",
            f"Selected mode: {selected_mode}",
            f"Resolved mode: {mode}",
            f"Runtime CPU state forced: {bool(RUNTIME.device_mode_applied)}",
            f"CUDA visible: {yes_no(_cuda_available())}",
            f"Comfy device is CPU: {yes_no(_is_comfy_cpu_device())}",
            f"GPU: {gpu_free:.2f}/{gpu_total:.2f} GiB free/total",
            f"RAM: {ram_free:.2f}/{ram_total:.2f} GiB available/total",
            f"Torch threads: {torch.get_num_threads()}",
            f"Qwen: {qwen_gguf} — {fmt(qwen)}",
            f"FLUX: {flux_gguf} — {fmt(flux)}",
            f"VAE: {vae_name} — {fmt(vae)}",
            f"Active tracked models: {active_text}",
            f"Caches: {cache_text}",
            f"Caches cleared by this report: {cleared_text}",
            f"Last timings: {timing_text}",
        ]

        if mode == MODE_CPU_ONLY:
            lines.append(
                "CPU mode: this node attempts to force ComfyUI into CPU state. "
                "For maximum reliability, close ComfyUI and start it with --cpu."
            )

            if _cuda_available():
                lines.append(
                    "Warning: CUDA is still visible to PyTorch. Some custom GGUF operations "
                    "may still prefer GPU unless ComfyUI was launched with --cpu."
                )

        report_text = "\n".join(lines)
        return {"ui": {"text": [report_text]}, "result": (report_text,)}


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


NODE_CLASS_MAPPINGS = {
    "Flux2KleinGGUFLoader": Flux2KleinGGUFLoader,
    "Flux2StagedQwenEncoder": Flux2StagedQwenEncoder,
    "Flux2KleinCanvas": Flux2KleinCanvas,
    "Flux2KleinDistilledSampler": Flux2KleinDistilledSampler,
    "Flux2ReferenceStack": Flux2ReferenceStack,
    "Flux2AutoVAEDecode": Flux2AutoVAEDecode,
    "Flux2MemoryReport": Flux2MemoryReport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinGGUFLoader": "FLUX2 Klein GGUF Loader (Staged)",
    "Flux2StagedQwenEncoder": "Qwen3 4B Encode + Release (Staged)",
    "Flux2KleinCanvas": "FLUX2 Klein Canvas Presets",
    "Flux2KleinDistilledSampler": "FLUX2 Klein 4-Step Sampler",
    "Flux2ReferenceStack": "FLUX2 Reference Stack (1-4 Images)",
    "Flux2AutoVAEDecode": "FLUX2 VAE Decode (Auto / Quality Safe)",
    "Flux2MemoryReport": "FLUX2 Staged Memory Report",
}

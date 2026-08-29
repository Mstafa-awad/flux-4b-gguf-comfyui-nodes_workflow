"""Memory-staged FLUX.2 Klein 4B GGUF nodes for ComfyUI.

The pack deliberately delegates model parsing and quantized operations to
ComfyUI-GGUF, and delegates FLUX.2 sampling math to current ComfyUI core nodes.
It only owns orchestration, validation, presets, and explicit stage cleanup.
"""

# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Mostafa Awad

from __future__ import annotations

import gc
import inspect
import logging
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

import comfy.model_management as model_management
import comfy.samplers
import comfy.utils
import folder_paths
import nodes as comfy_nodes


LOG = logging.getLogger("Flux2KleinGGUFStaged")
CATEGORY = "Flux2 Klein GGUF Staged"


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


def _unique(items: Iterable[str]) -> List[str]:
    return sorted(set(x for x in items if isinstance(x, str)))


def _filenames(keys: Sequence[str], extension: str | None = None) -> List[str]:
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
    files = _filenames(("clip_gguf", "text_encoders", "clip"), ".gguf")
    qwen = [x for x in files if "qwen" in x.lower()]
    return qwen or files or ["Qwen3-4B-Q8_0.gguf"]


def _vae_names() -> List[str]:
    files = _filenames(("vae",))
    flux = [x for x in files if "flux2" in x.lower() or "flux_2" in x.lower()]
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


def _invoke(name: str, **kwargs):
    """Invoke a registered node and normalize legacy/V3 output containers.

    Current ComfyUI V3 core nodes return ``io.NodeOutput`` with outputs stored
    in ``args``. Older ComfyUI and most custom nodes return ordinary tuples.
    Keeping this conversion here makes every wrapper below work with both APIs.
    """
    cls = _require_node(name)
    instance = cls()
    function_name = getattr(cls, "FUNCTION")
    function = getattr(instance, function_name)
    signature = inspect.signature(function)
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
    result = function(**kwargs)
    if hasattr(result, "args"):
        return tuple(result.args)
    if isinstance(result, dict) and "result" in result:
        wrapped = result["result"]
        return tuple(wrapped) if isinstance(wrapped, (list, tuple)) else (wrapped,)
    return result


def _release_patcher(patcher) -> None:
    if patcher is None:
        return
    try:
        if hasattr(model_management, "unload_model_and_clones"):
            model_management.unload_model_and_clones(patcher)
        elif hasattr(model_management, "unload_model_clones"):
            model_management.unload_model_clones(patcher)
        else:
            model_management.unload_all_models()
    finally:
        model_management.soft_empty_cache()


def _release_everything() -> None:
    model_management.unload_all_models()
    model_management.soft_empty_cache()


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


def _round16(value: int) -> int:
    return max(64, int(round(int(value) / 16.0) * 16))


def _fit_megapixels(width: int, height: int, max_megapixels: float) -> Tuple[int, int]:
    """Preserve aspect ratio while fitting dimensions into a megapixel budget."""
    width, height = int(width), int(height)
    max_pixels = max(64 * 64, float(max_megapixels) * 1_000_000)
    scale = min(1.0, (max_pixels / max(1, width * height)) ** 0.5)
    if scale < 1.0:
        # Round down so a safety limit never becomes larger after alignment.
        width = max(64, int(width * scale) // 16 * 16)
        height = max(64, int(height * scale) // 16 * 16)
    else:
        width, height = _round16(width), _round16(height)
    return width, height


def _resize_reference(image, max_megapixels: float, method: str):
    """Downscale a BHWC Comfy image before VAE reference encoding."""
    source_width = int(image.shape[2])
    source_height = int(image.shape[1])
    width, height = _fit_megapixels(source_width, source_height, max_megapixels)
    if width == source_width and height == source_height:
        return image, width, height
    channels_first = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(channels_first, width, height, method, "disabled")
    return resized.movedim(1, -1), width, height


def _gpu_total_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
    except Exception:
        return 0.0


def _file_path(keys: Sequence[str], name: str) -> str | None:
    for key in keys:
        try:
            path = folder_paths.get_full_path(key, name)
            if path and os.path.isfile(path):
                return path
        except Exception:
            continue
    return None


def _file_size_gb(keys: Sequence[str], name: str) -> float | None:
    path = _file_path(keys, name)
    if not path:
        return None
    return os.path.getsize(path) / 1_000_000_000


class Flux2KleinGGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flux_gguf": (_gguf_diffusion_names(),),
                "dequant_dtype": (["target", "default", "float16", "bfloat16", "float32"], {"default": "target"}),
                "patch_dtype": (["default", "target", "float16", "bfloat16", "float32"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "load_info")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Loads FLUX.2 Klein 4B GGUF through ComfyUI-GGUF without occupying VRAM until sampling."

    def load(self, flux_gguf, dequant_dtype="target", patch_dtype="default", patch_on_device=False):
        low = flux_gguf.lower()
        if "base" in low:
            raise ValueError("This pack is tuned for the distilled 4-step Klein checkpoint, not the Base checkpoint.")
        if any(token in low for token in ("9b", "12b", "14b", "32b")):
            raise ValueError("Select FLUX.2 Klein 4B. Larger FLUX.2 checkpoints do not fit this 8GB workflow.")

        model = _invoke(
            "UnetLoaderGGUFAdvanced",
            unet_name=flux_gguf,
            dequant_dtype=dequant_dtype,
            patch_dtype=patch_dtype,
            patch_on_device=patch_on_device,
        )[0]
        size = _file_size_gb(("unet_gguf", "diffusion_models", "unet"), flux_gguf)
        size_text = f"{size:.2f} GB" if size is not None else "size unavailable"
        info = f"FLUX stage ready: {flux_gguf} ({size_text}); dequant={dequant_dtype}."
        return model, info


class Flux2StagedQwenEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_gguf": (_gguf_qwen_names(),),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "A cinematic photograph with natural light and fine detail"}),
                "force_gpu_stage": ("BOOLEAN", {"default": True}),
                "release_after_encode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative_zeroed", "stage_info")
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Loads Qwen3 4B on GPU, encodes once, moves conditioning to CPU, then releases Qwen VRAM."

    def encode(self, qwen_gguf, prompt, force_gpu_stage=True, release_after_encode=True):
        low = qwen_gguf.lower()
        if "qwen" not in low:
            LOG.warning("Selected text encoder does not include 'Qwen' in its filename: %s", qwen_gguf)
        if any(token in low for token in ("8b", "14b", "30b", "32b")):
            raise ValueError("FLUX.2 Klein 4B requires Qwen3 4B. Do not use the 8B or larger encoder.")
        if not prompt.strip():
            raise ValueError("The prompt is empty.")

        # The loader creates the encoder offloaded on CPU. ComfyUI then stages as
        # much as safely fits on the active GPU for this one encode operation.
        clip = _invoke("CLIPLoaderGGUF", clip_name=qwen_gguf, type="flux2")[0]
        patcher = getattr(clip, "patcher", None)
        try:
            if force_gpu_stage and patcher is not None:
                model_management.load_models_gpu([patcher])
            positive = _invoke("CLIPTextEncode", clip=clip, text=prompt)[0]
            positive = _to_cpu(positive)
            negative = _invoke("ConditioningZeroOut", conditioning=positive)[0]
            negative = _to_cpu(negative)
        finally:
            if release_after_encode:
                _release_patcher(patcher)
            del clip
            gc.collect()

        size = _file_size_gb(("clip_gguf", "text_encoders", "clip"), qwen_gguf)
        size_text = f"{size:.2f} GB" if size is not None else "size unavailable"
        info = f"Qwen staged on GPU and released: {qwen_gguf} ({size_text}). Conditioning is cached on CPU."
        return positive, negative, info


class Flux2KleinCanvas:
    PRESETS = list(OFFICIAL_PRESETS.keys()) + ["custom"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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

    def make(
        self,
        preset,
        custom_width=1024,
        custom_height=1024,
        batch_size=1,
        width_override=None,
        height_override=None,
    ):
        if width_override is not None and height_override is not None:
            width, height = _round16(width_override), _round16(height_override)
        elif preset == "custom":
            width, height = _round16(custom_width), _round16(custom_height)
        else:
            width, height = OFFICIAL_PRESETS[preset]
        latent = _invoke("EmptyFlux2LatentImage", width=width, height=height, batch_size=batch_size)[0]
        mp = width * height / 1_000_000
        warning = "" if mp <= 1.15 else " Higher than the recommended 8GB starting area; tiled VAE will be used."
        return latent, width, height, f"{width}x{height}, batch {batch_size}, {mp:.2f} MP.{warning}"


class Flux2KleinDistilledSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative_zeroed": ("CONDITIONING",),
                "latent": ("LATENT",),
                "width": ("INT", {"forceInput": True}),
                "height": ("INT", {"forceInput": True}),
                "seed": ("INT", {"default": 43, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 12}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "sampler_name": (_sampler_names(), {"default": "euler"}),
                "clean_vram_before_sampling": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("samples", "denoised_samples", "sample_info")
    FUNCTION = "sample"
    CATEGORY = CATEGORY
    DESCRIPTION = "Official FLUX.2 Klein distilled path: Euler + Flux2Scheduler, 4 steps, CFG 1."

    def sample(
        self,
        model,
        positive,
        negative_zeroed,
        latent,
        width,
        height,
        seed=43,
        steps=4,
        cfg=1.0,
        sampler_name="euler",
        clean_vram_before_sampling=True,
    ):
        if clean_vram_before_sampling:
            _release_everything()
        if steps != 4:
            LOG.warning("FLUX.2 Klein Distilled is validated at 4 steps; selected %s.", steps)
        if abs(float(cfg) - 1.0) > 0.001:
            LOG.warning("FLUX.2 Klein Distilled is validated at CFG 1.0; selected %s.", cfg)
        if sampler_name != "euler":
            LOG.warning("Euler is the validated Klein sampler; selected %s.", sampler_name)

        noise = _invoke("RandomNoise", noise_seed=seed)[0]
        sampler = _invoke("KSamplerSelect", sampler_name=sampler_name)[0]
        sigmas = _invoke("Flux2Scheduler", steps=steps, width=int(width), height=int(height))[0]
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
        output = result[0]
        denoised = result[1] if len(result) > 1 else output
        info = f"{width}x{height}; {steps} steps; CFG {cfg:g}; {sampler_name}; seed {seed}."
        return output, denoised, info


class Flux2ReferenceStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative_zeroed": ("CONDITIONING",),
                "vae": ("VAE",),
                "reference_1": ("IMAGE",),
                "output_size": (["first reference", "custom"], {"default": "first reference"}),
                "custom_width": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "custom_height": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 32}),
                "memory_budget_megapixels": ("FLOAT", {"default": 1.05, "min": 0.25, "max": 4.0, "step": 0.05}),
                "enforce_8gb_limit": ("BOOLEAN", {"default": True}),
                "resize_method": (["area", "lanczos", "bicubic", "bilinear"], {"default": "area"}),
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
    DESCRIPTION = "Fits references into an 8GB-safe total pixel budget, VAE-encodes them, applies ReferenceLatent, then releases the VAE stage."

    def apply(
        self,
        positive,
        negative_zeroed,
        vae,
        reference_1,
        output_size="first reference",
        custom_width=1024,
        custom_height=1024,
        tile_size=512,
        overlap=64,
        memory_budget_megapixels=1.05,
        enforce_8gb_limit=True,
        resize_method="area",
        reference_2=None,
        reference_3=None,
        reference_4=None,
    ):
        refs = [x for x in (reference_1, reference_2, reference_3, reference_4) if x is not None]
        source_sizes = [(int(image.shape[2]), int(image.shape[1])) for image in refs]
        if enforce_8gb_limit:
            # FLUX.2 attention cost includes every reference latent. Treat the
            # setting as a total budget, not a per-image budget, so four large
            # references do not silently become four times more expensive.
            per_reference_budget = float(memory_budget_megapixels) / len(refs)
        else:
            per_reference_budget = 1_000_000.0
        _release_everything()
        pos, neg = positive, negative_zeroed
        encoded_sizes = []
        try:
            for image in refs:
                image, encoded_width, encoded_height = _resize_reference(
                    image,
                    per_reference_budget,
                    resize_method,
                )
                encoded_sizes.append((encoded_width, encoded_height))
                pixels = int(image.shape[1]) * int(image.shape[2])
                use_tiled = _gpu_total_gb() <= 9.0 or pixels > 1_000_000
                if use_tiled:
                    latent = _invoke(
                        "VAEEncodeTiled",
                        pixels=image,
                        vae=vae,
                        tile_size=tile_size,
                        overlap=overlap,
                        temporal_size=64,
                        temporal_overlap=8,
                    )[0]
                else:
                    latent = _invoke("VAEEncode", pixels=image, vae=vae)[0]
                pos = _invoke("ReferenceLatent", conditioning=pos, latent=latent)[0]
                neg = _invoke("ReferenceLatent", conditioning=neg, latent=latent)[0]
        finally:
            _release_everything()

        if output_size == "first reference":
            width, height = source_sizes[0]
        else:
            width, height = int(custom_width), int(custom_height)
        if enforce_8gb_limit:
            width, height = _fit_megapixels(width, height, memory_budget_megapixels)
        else:
            width, height = _round16(width), _round16(height)

        encoded_text = ", ".join(
            f"{sw}x{sh}->{ew}x{eh}"
            for (sw, sh), (ew, eh) in zip(source_sizes, encoded_sizes)
        )
        info = (
            f"Applied {len(refs)} reference image(s); reference resize: {encoded_text}; "
            f"output {width}x{height}; total budget {memory_budget_megapixels:.2f} MP."
        )
        return pos, neg, width, height, info


class Flux2AutoVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "decode_mode": (["auto (8GB safe)", "always tiled", "normal"], {"default": "auto (8GB safe)"}),
                "tile_size": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 32}),
                "tile_trigger_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 8.0, "step": 0.05}),
                "release_after_decode": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "decode_info")
    FUNCTION = "decode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Unloads FLUX, then automatically uses tiled VAE decode on 8GB GPUs or large images."

    def decode(
        self,
        samples,
        vae,
        decode_mode="auto (8GB safe)",
        tile_size=512,
        overlap=64,
        tile_trigger_megapixels=1.0,
        release_after_decode=True,
    ):
        _release_everything()
        latent = samples["samples"]
        compression = 8
        try:
            compression = int(vae.spacial_compression_decode())
        except Exception:
            pass
        width = int(latent.shape[-1]) * compression
        height = int(latent.shape[-2]) * compression
        megapixels = width * height / 1_000_000
        gpu_gb = _gpu_total_gb()
        use_tiled = decode_mode == "always tiled"
        if decode_mode == "auto (8GB safe)":
            use_tiled = gpu_gb <= 9.0 or megapixels >= float(tile_trigger_megapixels)

        try:
            if use_tiled:
                image = _invoke(
                    "VAEDecodeTiled",
                    samples=samples,
                    vae=vae,
                    tile_size=tile_size,
                    overlap=overlap,
                    temporal_size=64,
                    temporal_overlap=8,
                )[0]
                method = f"tiled {tile_size}px / {overlap}px overlap"
            else:
                image = _invoke("VAEDecode", samples=samples, vae=vae)[0]
                method = "normal"
        finally:
            if release_after_decode:
                _release_everything()
        return image, f"Decoded {width}x{height} using {method}; FLUX was unloaded before VAE decode."


class Flux2MemoryReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_gguf": (_gguf_qwen_names(),),
                "flux_gguf": (_gguf_diffusion_names(),),
                "vae_name": (_vae_names(),),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "report"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def report(self, qwen_gguf, flux_gguf, vae_name):
        qwen = _file_size_gb(("clip_gguf", "text_encoders", "clip"), qwen_gguf)
        flux = _file_size_gb(("unet_gguf", "diffusion_models", "unet"), flux_gguf)
        vae = _file_size_gb(("vae",), vae_name)
        gpu_total = _gpu_total_gb()
        gpu_free = 0.0
        if torch.cuda.is_available():
            try:
                gpu_free = torch.cuda.mem_get_info()[0] / 2**30
            except Exception:
                pass
        try:
            import psutil

            ram_total = psutil.virtual_memory().total / 2**30
            ram_free = psutil.virtual_memory().available / 2**30
            ram_line = f"RAM: {ram_free:.1f}/{ram_total:.1f} GiB available/total"
        except Exception:
            ram_line = "RAM: unavailable"

        def fmt(value):
            return f"{value:.2f} GB" if value is not None else "not found"

        report = "\n".join(
            (
                "FLUX.2 Klein staged memory report",
                f"GPU: {gpu_free:.2f}/{gpu_total:.2f} GiB free/total",
                ram_line,
                f"Qwen: {qwen_gguf} — {fmt(qwen)}",
                f"FLUX: {flux_gguf} — {fmt(flux)}",
                f"VAE: {vae_name} — {fmt(vae)}",
                "Plan: Qwen GPU encode -> release -> FLUX sampling -> release -> tiled VAE decode.",
            )
        )
        return {"ui": {"text": [report]}, "result": (report,)}


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
    "Flux2StagedQwenEncoder": "Qwen3 4B Encode + Release (Staged GPU)",
    "Flux2KleinCanvas": "FLUX2 Klein Canvas Presets",
    "Flux2KleinDistilledSampler": "FLUX2 Klein 4-Step Sampler",
    "Flux2ReferenceStack": "FLUX2 Reference Stack (1-4 Images)",
    "Flux2AutoVAEDecode": "FLUX2 VAE Decode (Auto Tiled)",
    "Flux2MemoryReport": "FLUX2 Staged Memory Report",
}

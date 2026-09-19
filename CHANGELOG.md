# Changelog — FLUX.2 Klein 4B GGUF Staged Nodes

All notable changes to the node pack, the launcher scripts, and the CPU-only setup.
Work session: 2026-09-17 → 2026-09-19.

---

## [2.0.0] — Node pack rewrite (modes, caches, CPU support)

### Added
- **Performance modes** on every node: `balanced` (default), `low_vram`, `high_speed`, `cpu_only`.
- **Cleanup policies**: `auto`, `always_release`, `keep_loaded` (replaces the old "release everything always" behavior).
- **VAE policies** for encode and decode: `auto_by_mode`, `prefer_normal`, `always_tiled`, `normal_only`.
- **Prompt embedding cache** (exact-match only, 4 entries) — re-runs of the same prompt skip Qwen entirely.
- **Reference latent cache** (8 entries, keyed by image hash + VAE + settings) — same reference image is never re-encoded.
- **Scheduler sigma cache** (64 entries) — Flux2Scheduler results reused per (steps, width, height).
- **Patcher tracking** (weakrefs for clip / unet / vae) so cleanup unloads only what it should.
- **OOM safety net**: emergency cleanup + automatic retry with smaller tiles on encode, decode, and sampling.
- **CPU-only runtime**: ComfyUI CPU-state switch, CPU thread configuration, `cpu_force_float32` option, CPU warnings.
- **New node inputs**: `performance_mode`, `cleanup_policy`, encode/decode policy, tile trigger, cache toggles, OOM fallback toggle, `retry_after_oom`, `patch_on_device_policy`.
- **Memory Report upgrade**: resolved mode, CUDA visibility, device type, VRAM/RAM, active tracked models, cache stats, per-stage timings, cache-clear switch.
- **Canvas warnings** per mode for oversized resolutions and batch size > 1.

### Changed
- VAE decode now **prefers normal decode** when memory allows (balanced / high_speed); tiled is only a fallback or used in low_vram / cpu modes.
- Tile sizes per mode: 512 (low_vram / cpu), 768 (balanced), 1024 (high_speed).
- Cleanup is now **mode-driven** instead of unconditional full release after every stage.
- Loader `patch_on_device` became a policy (`auto / true / false`) with a VRAM-headroom check.
- Qwen encoder: optional keep-loaded behavior + prompt cache; GPU staging auto-disabled in `cpu_only`.

### Fixed (original file hygiene)
- License header was not commented out.
- `from future import annotations` → correct `from __future__ import annotations`.
- Trailing spaces removed from dictionary keys and display-name mapping keys.

### Quality guarantees (unchanged on purpose)
- Sampling math untouched: Euler, 4 steps, CFG 1.0.
- All **4 reference images** still supported, applied to both positive and negative conditioning.
- Caches only fire on exact input matches — no quality drift.
- Official Klein canvas presets kept.

---

## [2.0.1] — Tile seam (square grid) diagnosis — guidance only

- Squares in output = **tiled VAE decode seams** caused by too-small overlap.
- Fix A (best): `decode_policy = prefer_normal` → no tiles, no seams, auto-fallback if OOM.
- Fix B (if tiling required): overlap ≈ ¼ of tile size (e.g. tile 512 / overlap 128).
- Same rule applies to `encode_policy` on the Reference Stack if seams persist.

---

## [2.0.2] — CPU-only launcher script

- New `StartComfyCPU.bat` with ComfyUI's official `--cpu` flag.
- Removed GPU-only flags from the CPU bat: `PYTORCH_CUDA_ALLOC_CONF`, `--use-pytorch-cross-attention`, `--enable-dynamic-vram`, `--async-offload 2`.
- Original bat kept for GPU runs; recommended names: `StartComfy_GPU.bat` / `StartComfy_CPU.bat`.

---

## [2.0.3] — True CPU enforcement (ggml CUDA leak fix)

- **Problem:** log showed `Device: cpu`, yet GPU hit 90% during sampling (4 steps in 7s = GPU speed).
- **Cause:** ComfyUI-GGUF's ggml C++ backend has its own CUDA path and ignores PyTorch's CPU state.
- **Fix:** environment variables in the CPU bat:
  - `GGML_NO_CUDA=1`
  - `CUDA_VISIBLE_DEVICES=-1`
- **Result:** GPU at 0% for the whole run; ~5 min per image on CPU (expected and correct).

---

## [2.0.4] — Thread-forcing experiment (reverted)

- Tried `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `GGML_N_THREADS` pinned to core count.
- **Result:** slower (thread thrashing over RAM bandwidth) → reverted to the clean bat; engines auto-select threads.
- Documented why Task Manager shows low CPU%: RAM-bandwidth bottleneck + P-core/E-core averaging.

---

## [2.0.5] — CPU dtype fix (float32 → bfloat16)

- **Problem:** `cpu_only` mode's `cpu_force_float32=True` made the model run float32 on CPU → ~2× slower (194 s/step).
- **Fix:** on the GGUF Loader set `dequant_dtype` / `patch_dtype` = `bfloat16` (or `float16`) and uncheck `cpu_force_float32`.
- Noted as harmless: `IMPORT FAILED` for GPU-hardcoded custom nodes (SkinTokens, Trellis2) when the GPU is hidden.

---

## [2.0.6] — High-res 4-view character sheet (workflow fixes)

- **Problem:** 4096×2688 canvas silently shrank to ~1 MP.
- **Cause:** Reference Stack's `enforce_8gb_limit` clamps the width/height it outputs, and that clamped size was feeding the Sampler.
- **Fix (workflow):** wire **Canvas width/height → Sampler width/height** directly; keep the reference budget ON so only the reference sheet gets downscaled before encoding.
- **VAE at 11 MP:** `decode_policy = always_tiled`, tile 512, overlap 128 (seam-safe).
- **Mode fix:** `high_speed` on 8GB kept Qwen in VRAM → FLUX streamed from RAM (`0 MB usable, 4209 MB offloaded`, 33.7 s/step). Use **`balanced`** on 8GB so FLUX loads fully.
- **Recommended alternative for 4-side characters:** 4 separate portrait passes (e.g. 832×1248) using the same turnaround sheet as reference, then stitch — more detail per view, no OOM risk, single-view redo possible.

---

## Known limits / notes

- CPU-only works but is slow by design (~5 min at ~1 MP with bfloat16). It is a "does it run" mode, not a daily-driver mode.
- 11 MP output on 8GB VRAM is ~10× the validated area: expect long runs, tiled decode, or step-down resolutions (3072×2048 → 2048×1360) if sampling OOMs.
- Q8_0 GGUF remains the quality-safe quantization; Q6/Q5 are optional speed experiments with a quality trade-off.

## Planned / offered next

- Split the memory budget into two separate knobs: **reference budget** vs **output budget**, so the 8GB limit never touches output resolution again.
- Optional single global "force CPU everywhere" switch instead of per-node mode selection.

## 1.0.2 — 2026-08-29

- Resize large reference images before VAE encoding.
- Add a shared reference memory budget, defaulting to 1.05 megapixels.
- Limit the output canvas to an 8 GB-safe size by default.
- Add text-to-image, single-reference, and multi-reference workflows.

## 1.0.1

- Support ComfyUI's V3 `NodeOutput` return container.
- Fix the post-sampling `object of type 'NodeOutput' has no len()` error.

## 1.0.0

- Initial staged Qwen, FLUX-only sampling, and automatic tiled VAE release.

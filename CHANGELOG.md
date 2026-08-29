# Changelog

All notable changes are documented here.

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

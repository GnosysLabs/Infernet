# Infernet Image v1

Status: proposed runtime and package contract. The Chat/Image workspace switcher and Image shell are implemented; image inference is not connected yet.

## Product decision

- The user-facing edition is **Infernet Image**.
- Its upstream model is **Z-Image Turbo**.
- The first official diffusion-transformer artifact should be `z-image-turbo-Q4_K_M.gguf`.
- Users do not select quantizations, import GGUF files, or configure component paths.
- Infernet publishes and verifies one signed logical image release made from three separately checksummed components.

## Why Q4_K_M

The upstream transformer choices include:

| Quantization | Size | Decision |
| --- | ---: | --- |
| Q4_0 | 4.59 GB | Low-memory fallback candidate |
| Q4_K_M | 5.02 GB | Default candidate |
| Q5_K_M | 5.57 GB | Quality comparison candidate |
| Q8_0 | 7.22 GB | Too costly for the first replicated edition |
| BF16 | 12.3 GB | Reference-quality validation only |

Q4_K_M adds only 430 MB over Q4_0, but preserves more precision in sensitive layers through Unsloth Dynamic 2.0. It is 550 MB smaller than Q5_K_M and 2.20 GB smaller than Q8_0. Before release, compare Q4_K_M and Q5_K_M with a small deterministic prompt suite. If no material visual improvement is found, ship Q4_K_M.

Q2 and Q3 variants should not be the only official edition. They can be evaluated later as an explicit low-memory tier.

## Signed release components

The linked 5.02 GB GGUF contains only the diffusion transformer. A working Z-Image pipeline also needs a text encoder and VAE.

| Role | Candidate artifact | Approximate size | Candidate SHA-256 |
| --- | --- | ---: | --- |
| Diffusion transformer | [`z-image-turbo-Q4_K_M.gguf`](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/blob/main/z-image-turbo-Q4_K_M.gguf) | 5.02 GB | `e6494f87de6abaf6a561924f50317a5f271fc34bb4222aabbd801197df8f7daa` |
| Text encoder | [`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/blob/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf) | 2.50 GB | `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` |
| VAE | [`ae.safetensors`](https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/vae/ae.safetensors) | 335 MB | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |

The candidate weight download is about 7.86 GB decimal, or 7.32 GiB, before manifests and notices. Release engineering must pin immutable upstream revisions, re-read exact byte sizes, verify every checksum, and ship the Apache-2.0 model notices.

## Runtime boundary

Z-Image is a diffusion pipeline, not a llama.cpp text model. The existing `Demo` and `LlamaCpp` runtime kinds and `infernet-llama-*` package ABIs must not accept it.

Add a separate image runtime based on [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp), which supports Z-Image, GGUF, CUDA, Metal, Vulkan, CPU, and the three required component inputs.

Proposed contract:

- Runtime kind: `StableDiffusionCpp`
- Package ABI: `infernet-sdcpp-image-v1`
- Component roles: `diffusion_transformer`, `text_encoder`, and `vae`
- Default output: 1024 by 1024 PNG
- Default inference: 8 diffusion steps, neutral classifier-free guidance, deterministic seed when supplied
- Request: prompt, width, height, seed, and approved preset only
- Response: generated PNG reference plus seed, dimensions, duration, and release id

Use Flash Attention where supported. On an 8 GiB accelerator, keep the Q4_K_M transformer on the accelerator and offload the text encoder and VAE as needed. Treat 16 GiB unified memory as the recommended Apple Silicon tier. CPU-only execution remains unsupported as an interactive promise until measured.

## Network rollout

1. Package and run the complete pipeline locally on one capable node.
2. Persist finished images and metadata separately from chat messages.
3. Let Infernet distribute the three signed components through the existing verified model-transfer layer.
4. Route a generation request to one capable peer and return only the finished image plus metadata.
5. Benchmark inter-peer DiT splitting separately before adding it to the runtime contract.

Do not reuse stable-diffusion.cpp RPC as the network design. Its client-driven tensor and graph transfer is different from Infernet's worker-cached, signed package model.

## Release gate

- Generate a deterministic evaluation set on RTX 4060, RTX 3090, and both Apple Silicon test nodes.
- Compare Q4_K_M against Q5_K_M for prompt adherence, text rendering, faces, hands, and fine texture.
- Measure cold load, warm generation, peak accelerator memory, peak system memory, and PNG transfer time.
- Verify cancel, timeout, out-of-memory, corrupt-package, and disconnected-peer behavior.
- Keep GGUF names, quantization terms, component paths, and runtime flags out of the everyday UI.

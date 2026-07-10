# Infernet Image v1

Status: requester-local and authenticated multi-machine generation are
implemented. When two or more eligible physical machines are online, the
diffusion transformer is divided into contiguous block ranges across the
requester and every selected remote machine for every denoising step.

## Product decision

- The user-facing edition is **Infernet Image**.
- Its upstream model is **Z-Image Turbo**.
- The first official diffusion-transformer artifact is `z-image-turbo-Q4_K_M.gguf`.
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

| Role | Pinned artifact | Exact bytes | Pinned SHA-256 |
| --- | --- | ---: | --- |
| Diffusion transformer | [`z-image-turbo-Q4_K_M.gguf`](https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/blob/main/z-image-turbo-Q4_K_M.gguf) | 5,017,613,376 | `e6494f87de6abaf6a561924f50317a5f271fc34bb4222aabbd801197df8f7daa` |
| Text encoder | [`Qwen3-4B-Instruct-2507-Q4_K_M.gguf`](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/blob/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf) | 2,497,281,120 | `3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597` |
| VAE | [`ae.safetensors`](https://huggingface.co/Comfy-Org/z_image_turbo/blob/main/split_files/vae/ae.safetensors) | 335,304,388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |

The pinned weight download is 7,850,198,884 bytes, or about 7.31 GiB, before
manifests and notices. Every URL uses an immutable upstream revision. The
installer supports range-resume, checks exact byte counts, verifies every
SHA-256, and records a file-identity marker before the package can be advertised
or used.

## Runtime boundary

Z-Image is a diffusion pipeline, not a llama.cpp text model. The existing `Demo` and `LlamaCpp` runtime kinds and `infernet-llama-*` package ABIs must not accept it.

The requester-local slice uses a separate runtime based on [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp), pinned to commit `cc734292286f85f9c48305d94d7fd22f42838522`. It is not routed through the chat runtime.

Implemented local contract:

- Runtime kind: `StableDiffusionCpp`
- Package ABI: `infernet-sdcpp-image-v1`
- Component roles: `diffusion_transformer`, `text_encoder`, and `vae`
- Default output: 1024 by 1024 PNG
- Default inference: 8 diffusion steps, neutral classifier-free guidance, deterministic seed when supplied
- Request: prompt, width, height, seed, and approved preset only
- Response: generated PNG reference plus seed, dimensions, duration, and release id

Use Flash Attention where supported. On an 8 GiB accelerator, keep the Q4_K_M transformer on the accelerator and offload the text encoder and VAE as needed. Treat 16 GiB unified memory as the recommended Apple Silicon tier. CPU-only execution remains unsupported as an interactive promise until measured.

## Network rollout

Image inference follows the product-wide physical-machine placement invariant:

- Whenever two or more eligible distinct physical machines exist, Infernet
  always splits the pipeline across them. It never selects a sole-machine image
  plan while another eligible physical machine is available.
- A complete pipeline may run on one machine only when that machine is the
  requester's own physical machine and it is the sole eligible option.
- A sole eligible remote physical machine is not a valid plan. The request must
  wait for another eligible machine or fail with an actionable availability
  message.
- When the requester and a remote physical machine are both eligible, both
  participate in the split pipeline. Separate peer identities on one physical
  machine do not satisfy this requirement.

The first network topology is a sticky diffusion-transformer block split, not
whole-pipeline remote execution:

1. Package the text encoder, diffusion transformer, and VAE as independently
   verified, role-scoped components of one signed release.
2. Keep prompt encoding, scheduling, and VAE decode on the requester. These
   lighter stages do not count as the requester's distributed contribution.
3. Assign the diffusion transformer to an ordered device list whose first
   device is the requester's accelerator and whose remaining devices are one
   authenticated worker on every other eligible physical machine. Split
   contiguous transformer-block ranges across that complete list and reuse the
   same sticky assignment for all eight denoising steps.
4. Force a non-empty range on every selected machine even when one accelerator
   could hold the complete transformer. After generation, require substantial
   tensor traffic through every selected worker tunnel; discard the PNG if any
   participant did not actually execute its assigned DiT range.
5. Persist finished images and metadata separately from chat messages, and let
   Infernet distribute each signed component through the existing verified
   model-transfer layer.
6. If a participant disappears, cancel or acquire a new valid split plan. Never
   collapse the active request onto one remote machine. A new sole-machine plan
   is valid only if the requester is then the sole eligible machine.

The initial implementation uses stable-diffusion.cpp's layer-split scheduler
over an image-only authenticated GGML tunnel. The pinned image runtime and
worker link the exact same stable-diffusion.cpp GGML core; the llama.cpp worker
is kept separate even when it advertises the same RPC wire version. Raw worker
TCP endpoints are never advertised or accepted: the server stays on loopback,
each coordinator proxy is bound to an exact authenticated PeerId, and a failed
distributed plan never falls back to local-only or remote-only execution.

This first implementation transfers assigned weights and repeated activation
boundaries through the tunnel. A later worker-cached image protocol may reduce
warm-generation transfer cost, but it must preserve the same requester-plus-
remote DiT participation invariant.

## Release gate

- Generate a deterministic evaluation set on RTX 4060, RTX 3090, and both Apple Silicon test nodes.
- Compare Q4_K_M against Q5_K_M for prompt adherence, text rendering, faces, hands, and fine texture.
- Measure cold load, warm generation, peak accelerator memory, peak system memory, and PNG transfer time.
- Verify cancel, timeout, out-of-memory, corrupt-package, and disconnected-peer behavior.
- Verify that one eligible requester can run locally, one eligible remote
  machine is rejected, requester-plus-remote execution is split, multiple peers
  on one machine do not count as multiple machines, and no failure path
  collapses onto a sole remote machine.
- Compare the stage-split output against the pinned monolithic reference for the
  same prompt, seed, dimensions, and preset before enabling multi-machine image
  generation.
- Keep GGUF names, quantization terms, component paths, and runtime flags out of the everyday UI.

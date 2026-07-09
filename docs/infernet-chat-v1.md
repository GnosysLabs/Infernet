# Infernet Chat v1

## Product decision

Infernet launches with one curated model: **Infernet Chat**. Users do not import
GGUF files, paste repository URLs, or choose quantizations. Infernet builds,
tests, signs, publishes, and updates the official package.

The selected upstream release is **Gemma 4 26B A4B Instruct QAT Q4_0**:

- 25.2B total parameters and 3.8B active parameters per token
- 30 transformer layers
- 128 experts, with 8 routed experts plus 1 shared expert active per token
- 256K architectural context; Infernet v1 caps requests at 8K–16K
- 14.4 GB official text-model Q4_0 payload
- Apache 2.0 license

Primary references:

- [Google Gemma 4 overview](https://ai.google.dev/gemma/docs/core)
- [Google Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Official Gemma 4 26B A4B model](https://huggingface.co/google/gemma-4-26B-A4B-it)
- [Official QAT Q4_0 GGUF](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)

## Launch hardware

The first validation network is one RTX 3090, one RTX 4060, and two Apple
Silicon MacBook Pros. Nodes report their compute backend, device name, free
accelerator or unified memory, CPU cores, session load, queue depth, and
optional measured throughput. NVIDIA capacity comes from `nvidia-smi`; Apple
Silicon reports Metal with its currently available unified memory. Device names
are descriptive only—the planner uses the reported numbers.

The launch scheduler assigns immutable, content-addressed components in
contiguous layer ranges. It reserves model weights, KV cache, runtime scratch
space, and a safety margin before accepting an assignment. Saturated machines
are skipped; measured throughput and live load influence balancing. The 4060
can therefore receive only components that safely fit its actual free VRAM,
while the 3090 and higher-memory Macs can receive larger ranges. The scheduler
tries to include every useful reported machine (up to eight) when the fixed
components fit safely, then falls back to the largest feasible set. Within that
set it minimizes network boundaries.

## Package contract

`infernet-chat-v1.infermodel` is the signed model and execution manifest, and
`infernet-chat-v1` is the only model ID accepted by the launch catalog. It
references content-addressed `.infershard` components.

The temporary full-model compatibility release is pinned byte for byte while
the multi-component runtime is completed:

- release: `infernet-chat-v1-compatibility`
- version: `1.0.0-compat.1`
- upstream revision: `dfc00409adc70be497fee9c90bfe76b3ee130f2e`
- payload bytes: `14,439,361,440`
- SHA-256: `4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5`

These values come from Google's [raw repository pointer](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/raw/main/gemma-4-26B_q4_0-it.gguf).
The app rejects peers advertising the same model ID with any other version,
size, layer coverage, or checksum. This compatibility payload is a trust anchor
and single-node correctness fallback, not the distributed launch package.

The target component plan is:

- one component per transformer layer
- one shared component for tokenizer, chat template, tied embeddings, and final
  normalization
- the optional 1.19 GB vision projector as a separate, non-launch component

Each MoE layer keeps its router and all experts together. Expert-level network
routing is out of scope because it would create excessive per-token network
traffic.

The storage invariant is non-negotiable:

```text
sum(unique component payload bytes) <= upstream text payload + 1% metadata
```

Sharding must not duplicate global tensors. Additional physical copies are
allowed only as an explicit replication policy across peers, never as an
accidental side effect of package construction.

## Reliability boundaries

Infernet Chat v1 is text-only and uses ordinary autoregressive decoding.
Vision, MTP/speculative decoding, expert-level sharding, arbitrary context
lengths, and arbitrary upstream models are not launch dependencies.

The official package does not ship until all of these pass:

1. The pinned Gemma 4 build produces correct multi-token output on the 3090.
2. The same pinned runtime produces correct output on Metal.
3. Layer-range output matches the full graph at each boundary within the
   quantized-runtime tolerance.
4. The 3090, 4060, and both Macs complete a multi-node prompt without a full
   model copy on the 4060 or either Mac.
5. Total unique package bytes satisfy the storage invariant.
6. Repeated runs survive peer restart, missing-component, timeout, and checksum
   failure tests without freezing a host.

Until partial Gemma 4 MoE execution passes those gates, the compatibility
package remains a complete `0:N` package for correctness testing and is not
presented as the distributed launch release.

## Seeding the pinned compatibility release

This is a maintainer operation, not a user import flow. After obtaining the
exact official file, seed it on the 3090 host with:

```sh
cargo run -p infernet-worker -- model add-local \
  --gguf /path/to/gemma-4-26B_q4_0-it.gguf \
  --cache-dir /path/to/infernet-official-seed

cargo run -p infernet-worker -- model serve \
  --cache-dir /path/to/infernet-official-seed \
  --p2p-listen /ip4/0.0.0.0/tcp/9777
```

The release command checks the exact size and SHA-256 before placing anything
in the seed cache. Other nodes obtain only the pinned payload from Infernet's
network and verify it while downloading.

# Infernet Chat v1

## Product decision

Infernet launches with one curated model: **Infernet Chat**. Users do not import
GGUF files, paste repository URLs, or choose quantizations. Infernet builds,
tests, signs, publishes, and updates the official package.

The selected upstream release is **Gemma 4 26B A4B Instruct QAT Q4_0**:

- 25.2B total parameters and 3.8B active parameters per token
- 30 transformer layers
- 128 experts, with 8 routed experts plus 1 shared expert active per token
- 256K architectural context; Infernet v1 allocates 32,768 tokens per request
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

The launch scheduler groups app identities by physical machine and selects an
eligible CUDA/Metal execution topology. Whenever at least two distinct eligible
physical machines are available, pinned llama.cpp assigns contiguous layers and
their KV buffers across at least two of them in proportion to backend memory. A
local-only topology is accepted only when the requesting machine is itself the
sole eligible machine; a lone remote machine never receives another user's whole
request. Infernet reserves KV cache, runtime scratch space, and a safety margin
before accepting the topology. Saturated machines and duplicate app identities
on one physical host are skipped. The topology remains stable across prompts so
weights and distributed KV stay resident instead of being retransferred whenever
free-memory telemetry changes.

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
and packaging bridge while the distributed launch package is completed. It does
not authorize routing another user's whole request to one remote machine.

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
7. Scheduling uses at least two distinct eligible physical machines whenever
   available, permits local-only execution only for the sole eligible requester,
   and rejects a topology containing only one remote machine.

For launch, the compatibility package remains a complete `0:N` package on one
coordinator, while llama.cpp RPC distributes its tensors and KV allocations
across the selected machines at load time whenever the topology has multiple
eligible physical machines. The coordinator is never used as a one-remote
whole-request fallback. A local requester may use the package alone only when it
is the sole eligible machine. This delivers real multi-machine execution without
duplicating global tensors into 30+ GB of generated shard files. A future
native-component release can remove the remaining requirement that one
coordinator stores the complete source package.

## Seeding the pinned compatibility release

This is a maintainer operation, not a user import flow. After obtaining the
exact official file, seed it on the 3090 host with:

```sh
cargo run -p infernet-worker -- model add-local \
  --gguf /path/to/gemma-4-26B_q4_0-it.gguf \
  --consume-source \
  --cache-dir /path/to/infernet-official-seed

cargo run -p infernet-worker -- model serve \
  --cache-dir /path/to/infernet-official-seed \
  --p2p-listen /ip4/0.0.0.0/tcp/9777
```

The release command checks the exact size and SHA-256 before placing anything
in the seed cache. `--consume-source` moves that verified payload into the
Infernet package, leaving one 14.4 GB copy instead of retaining a second source
copy. Other nodes receive only the tensors assigned to their RPC backend in
memory; tensor disk caching is disabled for launch.

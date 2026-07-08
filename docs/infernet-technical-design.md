# Infernet Technical Design

Date: 2026-07-08

## Goal

Infernet is a peer-to-peer split-inference system for running models that no
single participating machine can hold alone. A model is divided into layer
ranges. Each peer owns one or more ranges, executes only those layers, and
forwards activation tensors to the next peer in the route.

The first implementation must prove the core claim with a tiny transformer-like
model split across four independent nodes. Performance, economics, and
large-model support are intentionally secondary until activation forwarding and
route construction are working.

## Research Summary

### Petals

Petals is the closest prior system. It explicitly uses the BitTorrent-style
framing for LLM inference: participants load a small part of a model and join a
network serving the remaining parts. Its paper describes servers hosting subsets
of model layers, usually transformer blocks, while clients form a chain of
pipeline-parallel consecutive servers to run the full model. The system also
uses routing, load balancing, low-latency peer selection, dynamic quantization,
and fault-tolerant inference algorithms.

Reusable ideas:

- Split by transformer blocks/layers, not arbitrary tensor slices, for WAN
  friendliness.
- Use a metadata overlay to discover peers serving layer ranges.
- Build a consecutive route for each inference session.
- Prefer low-latency, sufficiently capable peers.
- Treat peers as both clients and servers.
- Keep per-session state on the peers that own each layer range.

Limitations to account for:

- Petals acknowledges privacy risk because intermediate activations are sent to
  servers and may leak prompt information.
- Correctness is not guaranteed against malicious peers without validation or
  redundancy.
- The implementation is Python/PyTorch/Hivemind-oriented, while Infernet's core
  runtime is Rust and should ship as a cross-platform desktop/headless node.
- Public Petals-style swarms depend on enough volunteers per model and per
  layer; sparse layers become route bottlenecks.

Infernet difference:

Infernet should reuse Petals' layer-pipeline concept, but not copy its product
shape. Infernet is a desktop app and protocol where the local computer is simply
one peer in a distributed network, with Rust/libp2p networking, explicit UI
visualization, and llama.cpp/GGUF integration rather than a PyTorch-first
runtime.

Sources: [Petals GitHub](https://github.com/bigscience-workshop/petals),
[Petals paper](https://aclanthology.org/2023.acl-demo.54.pdf),
[Petals arXiv](https://arxiv.org/abs/2312.08361).

### Hivemind and DHT Metadata

Hivemind is useful background for decentralized neural network computation. Its
DHT is Kademlia-based and optimized for lightweight metadata, bulk get/store,
and caching. This maps well to model-shard advertisements, but Infernet should
prefer Rust libp2p's Kademlia implementation rather than depending on a Python
networking stack.

Reusable ideas:

- Store small, expiring advertisements in the DHT.
- Keep shard metadata lightweight and refresh periodically.
- Treat churn as normal; route construction should expect missing peers.

Infernet difference:

Infernet's DHT records should describe model identity, layer range, quantization
or runtime compatibility, peer capability, and expiry. The actual activation
traffic should not go through the DHT.

Sources: [Hivemind GitHub](https://github.com/learning-at-home/hivemind),
[Hivemind DHT docs](https://learning-at-home.readthedocs.io/en/latest/modules/dht.html).

### Layer-Wise Partitioning vs Tensor Parallelism

Layer-wise partitioning is inter-operator model parallelism: peer A runs layers
0-2, peer B runs 3-5, and so on. It has poor parallel utilization for one
request but low network fan-out: only the activation boundary between layer
ranges crosses the network.

Tensor parallelism is intra-operator parallelism: attention and feed-forward
matrices are split across devices. This is effective inside one server or
cluster, but it requires frequent collective communication such as all-reduce or
all-gather. Literature on edge transformer inference specifically notes that
tensor-parallel communication overhead is acceptable on high-bandwidth GPU
interconnects but problematic over slower networks.

Infernet decision:

- Use layer-wise partitioning across Internet peers.
- Reserve tensor parallelism for a future per-peer local runtime, for example a
  headless worker using multiple GPUs inside one machine or rack.
- Avoid WAN tensor parallelism for Phase 1 and Phase 2.

Sources:
[edge transformer distributed inference paper](https://iqua.ece.toronto.edu/papers/chenghao-icdcs24.pdf),
[NVIDIA parallelisms guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html),
[model parallelism survey](https://arxiv.org/html/2403.03699v1).

### llama.cpp and GGUF

llama.cpp is the best fit for Infernet's cross-platform real-model runtime. It
is C/C++, supports many CPU/GPU backends, and uses GGUF as the model format.
GGUF is a binary, mmap-friendly, single-file model format with a header,
metadata key-values, tensor descriptors, and tensor data. The format includes
architecture metadata such as block counts and tensor names that can be mapped
to layer ranges.

Useful existing llama.cpp ideas:

- GGUF as the canonical local model artifact.
- mmap loading for fast startup.
- CPU, CUDA, HIP, Vulkan, Metal, SYCL, OpenCL, and other backends.
- Existing multi-GPU split modes inside one llama.cpp process.
- KV cache controls and quantized KV cache types.

Current mismatch:

- llama.cpp's model split modes are intra-process and local-device oriented.
  They are not a ready-made Internet peer protocol.
- Infernet needs a runtime that can load only a contiguous layer range and expose
  an activation-in / activation-out interface. llama.cpp's public CLI/server
  surface is token-in / token-out for complete models.
- GGUF is a single-file deployment format, not a first-class distributed shard
  manifest. Infernet needs an additional manifest that maps tensors to layer
  ranges and validates shard hashes.

Infernet decision:

- Phase 1 uses a tiny Rust demo model to prove architecture without invasive
  llama.cpp work.
- Phase 2 targets one real GGUF model first.
- Phase 2 should either:
  - add a narrow llama.cpp integration layer that can evaluate a layer range, or
  - extract GGUF tensors into Infernet-owned shards and execute a small supported
    transformer architecture directly.
- Do not attempt broad GGUF model support until one model works end to end.

Sources:
[GGUF spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md),
[llama.cpp GitHub](https://github.com/ggml-org/llama.cpp),
[llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md),
[llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md),
[llama.cpp add-model guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md).

### vLLM, ONNX, and Serving Engines

vLLM is excellent for centralized serving. Its architecture has API server
processes, engine-core scheduler processes, GPU workers, KV cache management,
and distributed tensor/pipeline parallelism with Ray or multiprocessing. Its
PagedAttention work is directly relevant to future high-throughput serving
inside a peer.

Current mismatch:

- vLLM assumes coordinated server deployment, not untrusted peers joining and
  leaving a public network.
- Its scheduler is central to a deployment. Infernet must not depend on a
  centralized inference scheduler.
- vLLM optimizes batching and GPU utilization, while Infernet Phase 1 optimizes
  proof of distributed ownership.

ONNX can represent partitioned graphs, but the GGUF/llama.cpp ecosystem is more
aligned with consumer local LLMs and cross-platform quantized inference.

Infernet decision:

- Reuse vLLM concepts later inside powerful workers: paged KV cache, continuous
  batching, and scheduling.
- Do not use vLLM as the network architecture.
- Defer ONNX unless GGUF/llama.cpp proves too expensive for layer-range
  execution.

Sources:
[vLLM parallelism docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/),
[vLLM architecture overview](https://docs.vllm.ai/en/latest/design/arch_overview/),
[PagedAttention paper](https://arxiv.org/abs/2309.06180).

### libp2p

libp2p matches the peer-first architecture. The relevant building blocks are:

- Persistent cryptographic peer identity.
- mDNS for zero-config local discovery.
- Kademlia DHT for WAN peer routing and provider records.
- request-response substreams for directed activation RPC.
- ping/identify/autonat/circuit relay/DCUtR for reachability, latency, and NAT
  traversal.
- optional gossipsub for low-value network telemetry, not inference traffic.

Infernet decision:

- Use mDNS first for local four-node demos.
- Use Kademlia provider records for model/layer discovery after the local demo.
- Use request-response for activation forwarding.
- Keep bootstrap servers limited to discovery. They must never execute or
  schedule inference.

Sources:
[libp2p discovery overview](https://libp2p.io/docs/discovery-routing-overview/),
[libp2p mDNS](https://libp2p.io/docs/mdns/),
[rust-libp2p mDNS](https://docs.rs/libp2p/latest/libp2p/mdns/),
[libp2p Kademlia](https://libp2p.io/docs/kademlia-dht/),
[rust-libp2p request-response](https://docs.rs/libp2p-request-response/latest/libp2p_request_response/),
[rust-libp2p hole punching](https://docs.rs/libp2p/latest/libp2p/tutorials/hole_punching/index.html).

## Proposed Architecture

### Crate Layout

- `infernet-protocol`: versioned protocol types and wire frames.
- `infernet-model`: model manifests, layer ranges, shard metadata, and routing
  compatibility checks.
- `infernet-runtime`: execution trait plus demo-model runtime now and
  llama.cpp/GGUF runtime later.
- `infernet-router`: route construction from peer advertisements.
- `infernet-node`: transport abstraction, peer identity, discovery, and
  activation RPC.
- `infernet-worker`: headless CLI node for local and cloud peers.
- `infernet-ui`: Tauri/React desktop app after the headless proof works.

### Core Data Model

`ModelManifest`:

- `model_id`
- `display_name`
- `architecture`
- `layer_count`
- `hidden_size`
- `activation_dtype`
- `tokenizer`
- `shard_hashes`
- `runtime_requirements`

`ShardDescriptor`:

- `model_id`
- `layer_start`
- `layer_end`
- `runtime_kind`
- `quantization`
- `shard_hash`

`NodeAdvertisement`:

- `peer_id`
- `listen_addresses`
- `available_ram`
- `available_vram`
- `bandwidth_hint`
- `latency_hint`
- `hosted_shards`
- `expires_at`
- `signature`

`ActivationRequest` over `/infernet/activation/1`:

- `trace_id`
- `model_id`
- `route`
- `current_hop_index`
- `sequence_position`
- `hidden_size`
- `activation`
- `prompt` metadata for demo mode
- `trace`

`ActivationResponse` over `/infernet/activation/1`:

- `trace_id`
- `peer_id`
- `processed_layer_start`
- `processed_layer_end`
- `output_activation`
- `timing_ms`
- `trace`
- `output_text` for demo mode
- `error`

### Inference Flow

1. User chooses a model.
2. Client loads the model manifest.
3. Router queries local discovery or DHT provider records for peers hosting each
   required layer range.
4. Router constructs a route that covers `[0, layer_count)` exactly once with
   contiguous ranges.
5. Client tokenizes prompt and creates the first activation request.
6. Peer 0 executes its local layer range.
7. Peer 0 forwards the resulting activation request to peer 1 over
   `/infernet/activation/1`.
8. Each peer repeats execution and forwarding.
9. Final peer returns logits or generated token output through the
   request-response chain.
10. UI displays route, peers, latency, and trace events.

For generation, route stickiness is required. Each peer owning layers with
attention stores the KV cache for those layers and that invocation. If a peer
fails mid-session, the first fallback should recompute the session on a new
route. KV transfer can come later.

### First Demo Runtime

The tiny model should be deterministic, serializable, and small enough to run on
CPU everywhere:

- 12 layer-like blocks.
- Fixed hidden size, for example 16 or 32.
- Simple tokenizer maps prompt bytes to an activation vector.
- Each layer applies a deterministic transformation using per-layer weights.
- Output decoder maps the final activation to a short deterministic string.

This is not meant to be a useful language model. It is meant to test the
protocol invariants:

- no process owns all 12 layers;
- every process rejects requests outside its assigned layer range;
- route must cover every layer once, in order;
- activation frames are forwarded between processes;
- final trace proves all four nodes participated.

### Routing

Phase 1 route construction:

- Input: model manifest and a list of node advertisements.
- Filter advertisements by `model_id`, runtime compatibility, and expiry.
- Build interval coverage from `0..layer_count`.
- Prefer fewer hops and lower advertised latency.
- Reject routes with gaps or overlaps.

Future route construction:

- Probe actual RTT with libp2p ping.
- Include bandwidth estimates because prefill activation frames can be large.
- Include worker load and queue time.
- Replicate hot layers.
- Support alternate route repair when a peer drops.

### Discovery and Activation Transport

The runtime and router should not depend directly on libp2p. `infernet-node`
owns transport-specific details.

Transport contract:

- advertise hosted shards;
- discover compatible peers;
- send an activation request to a peer;
- receive an activation request and return a layer result;
- emit trace events.

The current libp2p implementation uses two separate traffic classes:

- mDNS for LAN discovery;
- gossipsub for small `NodeAdvertisement` metadata;
- request-response for directed activation payloads on
  `/infernet/activation/1`.

Discovery gossip is not in the data path. A client and workers use mDNS and
gossipsub to learn `peer_id`, libp2p listen addresses, `model_id`, hosted layer
ranges, and protocol version. Once the router has complete coverage, activation
traffic is sent as directed request-response substreams. The client sends only
to the first route peer; each worker forwards to the next route peer.

Current activation response behavior is a response chain: the final peer
returns the demo output to the previous peer, which relays it upstream until the
client receives it. This is simpler than direct-to-client final responses and
keeps the trace attached to one request-response path.

Future transport work:

- ping for route metrics;
- Kademlia provider records for WAN discovery;
- binary tensor framing instead of JSON `Vec<f32>`;
- route repair if a peer drops mid-inference.

### Security and Trust

Transport encryption protects activation bytes in transit, but the executing
peer must see activations. Phase 1 must not claim privacy from participating
peers.

Initial protections:

- persistent peer identities;
- signed advertisements;
- hash-verified shards;
- request size limits;
- invocation TTLs;
- explicit UI warning for distributed-inference privacy limitations.

Future protections:

- redundant execution for suspicious layers;
- sampled verification against trusted peers;
- reputation;
- private swarms;
- TEE experiments;
- research into activation privacy.

## Implementation Plan

### Phase 1A: Headless Split-Inference Core

- Create Rust workspace with the core crates.
- Implement manifests, layer ranges, advertisements, routes, and validation.
- Implement the demo runtime.
- Implement deterministic tests proving route coverage and layer execution.
- Implement a headless worker with direct local transport so four processes can
  demonstrate activation forwarding before libp2p complexity is introduced.

### Phase 1B: libp2p LAN Demo

- Add persistent node identity.
- Add mDNS peer discovery.
- Add request-response activation RPC.
- Replace direct route addresses with discovered peer IDs.
- Run four local nodes with separate layer ranges.

Current implementation note:

Phase 1B now uses libp2p mDNS for LAN peer discovery and gossipsub for shard
metadata advertisements. Each worker advertises its `peer_id`, libp2p listen
address, `model_id`, hosted layer range, and protocol version. A client command
discovers advertisements, inserts them into a shard registry, and asks the router
to construct a complete ordered route.

The activation data path has moved off direct TCP. Inference now uses the
libp2p request-response protocol `/infernet/activation/1`. The request includes
`trace_id`, `model_id`, the full route, `current_hop_index`, the demo activation
payload, and prompt metadata. Each worker validates that the current route hop
matches its peer id and hosted layer range, executes those layers, appends a
trace event, and either forwards to the next peer or returns the final demo
output.

The shard registry stores the latest advertisement per peer. Route construction
first computes coverage gaps for the requested model. If advertisements do not
cover `[0, layer_count)`, the route call fails with concrete missing ranges such
as `3:6, 9:12`. If coverage is complete, the router greedily advances from layer
0, choosing a compatible shard that covers the current cursor and extends the
route toward the final layer.

### Phase 1C: Desktop Visualization

Current implementation:

- `infernet-ui` is a Tauri v2 + React + TypeScript app.
- The UI presents available models rather than execution modes.
- The top bar shows the UI client's libp2p identity as one node in the network.
- The route view discovers workers through the existing mDNS/gossipsub path and
  displays peer advertisements, shard ranges, selected model, and coverage.
- The chat panel invokes Rust commands, runs the selected model route through
  `/infernet/activation/1`, and listens for `infernet-progress` events.
- Progress events currently include route discovery, hop start, hop completion,
  activation size, timing, checksum, and final output.

Current UI limitation:

The client receives definitive hop timing from the final response trace because
the activation protocol returns through a response chain. This is sufficient for
the demo visualization. A future telemetry side channel can make intermediate
hop completion appear in the UI at the exact moment each remote worker finishes.

### Phase 2: One Real Model

- Choose one small supported GGUF model.
- Build an Infernet model manifest from GGUF metadata.
- Prototype contiguous layer-range execution.
- Decide whether to patch llama.cpp narrowly or implement the specific
  architecture in Rust/C++ around extracted GGUF tensors.

Current Phase 2 research and implementation notes live in
[gguf-split-inference-design.md](gguf-split-inference-design.md). The selected
first target is `llama-3.2-1b`. The repository now supports GGUF sidecar
shard metadata and runtime-kind-aware routing, but real llama.cpp layer-range
execution is deliberately blocked until a native bridge can load only the
assigned layer range.

## Open Questions

- Which first GGUF model should Phase 2 target?
- Should the first real runtime preserve GGUF as one file per peer, or materialize
  per-layer-range shard files?
- How should the UI represent privacy limitations without making distributed
  inference feel unusable?
- What is the minimum correctness check acceptable before public WAN swarms?
- How much central infrastructure is acceptable for bootstrap and health
  dashboards while keeping inference peer-to-peer?

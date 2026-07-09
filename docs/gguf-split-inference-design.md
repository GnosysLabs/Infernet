# Infernet GGUF Split-Inference Design

Date: 2026-07-08

## Milestone Definition

The next milestone is to replace the toy transformer with one real GGUF language
model that can run across multiple independent Infernet peers. Infernet is now
always a distributed network: the local computer is one node in the same network
as every other node. Users select a model, not an execution mode. The scheduler
chooses whether the route is local-only, mixed local/remote, or remote-only.

The proof is correctness, not speed:

- a real GGUF model is partitioned by contiguous transformer layer ranges;
- each peer loads only its assigned layer range and required boundary tensors;
- activations are forwarded over `/infernet/activation/1`;
- route construction remains dynamic through peer advertisements;
- the UI shows available models, route coverage, participating peers, and hop
  timing without exposing a Local-vs-AI-Grid mode switch.

## Primary Model Choice

Target `Llama 3.2 1B` first.

Reasons:

- llama.cpp has explicit LLaMA-family model support and identifies the 16-layer
  Llama 3.2 1B shape in its LLaMA architecture loader.
- The architecture is simpler than Qwen 2.5 3B for a first narrow patch.
- Public GGUF builds exist and can be launched with llama.cpp-compatible tools,
  for example `bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M`.
- A 1B model keeps iteration cost low while still proving real tokenization,
  attention, MLP, normalization, logits, and sampling.

Qwen 2.5 3B remains a good second target after the LLaMA split path works.

Sources:

- GGUF spec: <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
- llama.cpp repository: <https://github.com/ggml-org/llama.cpp>
- llama.cpp add-model guide:
  <https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md>
- llama.cpp public header:
  <https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h>
- Llama 3.2 1B GGUF candidate:
  <https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF>

Local source inspection used llama.cpp commit
`a646006f09d2f76f2d62d6c0d5e8e8490d570720` from 2026-07-08.

## GGUF Findings

GGUF is a binary GGML model container designed for fast loading. It contains:

- a magic/version header;
- metadata key-value records;
- tensor descriptors;
- tensor byte data, usually alignment-padded;
- tokenizer metadata;
- architecture metadata such as block/layer count.

Relevant metadata and tensor conventions:

- `general.architecture` identifies the architecture family.
- `{arch}.block_count` records the transformer block count.
- Repeated transformer-block tensor names use a block index placeholder in the
  llama.cpp conversion path. In GGUF tensor names this becomes names such as
  `blk.0.attn_norm.weight`, `blk.0.attn_q.weight`, and so on.
- Token embedding, final normalization, and output head tensors are outside the
  repeated block namespace and must be treated as boundary tensors.

Reusable for Infernet:

- GGUF metadata can derive `model_id`, architecture, layer count, tokenizer
  compatibility, quantization/file type, and tensor names.
- Tensor names make layer-range ownership computable without parsing model code
  for every architecture.
- A shard builder can emit a sidecar manifest first, then later materialize a
  physical shard file containing only the tensors required for a layer range.

Limitations:

- GGUF file splits are storage splits, not semantic layer shards.
- A complete GGUF file can include all tensors; using it directly through stock
  llama.cpp would not prove that no peer possesses the full model.
- Tokenizer compatibility must be pinned by metadata and checksum, not inferred
  from model display name.

## llama.cpp Internals Findings

### Transformer Layer Representation

For LLaMA-family models, llama.cpp loads architecture hyperparameters, then
creates tensors for the token embedding, repeated transformer blocks, final norm,
and output head. The LLaMA model loader loops over `0..n_layer` and creates
per-layer tensors such as attention norm, Q/K/V/O projections, FFN norm, FFN
gate/down/up projections, and architecture-specific optional tensors.

The graph builder then:

1. builds token embedding input;
2. builds position and attention inputs;
3. loops over every transformer layer from `0..n_layer`;
4. records each layer input internally;
5. applies attention, residual, FFN, and residual;
6. after the loop applies final norm and output head.

This is a good conceptual match for contiguous layer ranges because the repeated
block loop has a natural `[layer_start, layer_end)` boundary.

### Can Contiguous Layer Ranges Be Initialized Independently?

Not through the stock public llama.cpp API.

The current model loader expects each required tensor for the complete
architecture to be present and creates graph state for the full model. It has
storage-level support for split GGUF files, but those split files are additional
pieces of one logical model, not independent layer-serving shards. Loading a
full GGUF and merely skipping unused layers would not satisfy Infernet's core
proof because a peer would still have access to the full model artifact.

Practical path:

- Add an Infernet-specific llama.cpp integration layer that accepts
  `layer_start`, `layer_end`, `include_token_embedding`, and `include_lm_head`.
- Change the LLaMA tensor creation path, behind that integration flag, to create
  only selected block tensors plus required boundary tensors.
- Allow partial tensor lookup counts for the Infernet loader path only.
- Keep upstream llama.cpp behavior unchanged for normal users.

### Can Activations Be Imported and Exported?

Partially, but not enough through the public API.

llama.cpp's `llama_batch` supports `embd` input, which can carry float embeddings
instead of token IDs. Internally, llama.cpp also has graph inputs that can treat
that embedding buffer as hidden state. llama.cpp can also expose selected layer
inputs through internal/extension APIs used for embedding inspection.

What is missing for Infernet:

- a public way to start the normal LLaMA graph at arbitrary layer `N`;
- a public way to stop at layer `M` and return the hidden activation instead of
  final logits;
- a public C API for a partial model that has only selected layer tensors;
- distributed KV-cache semantics for generation when attention state is split
  across peers.

Practical path:

- First patch only one architecture path: Llama 3.2 1B.
- Add a graph builder path that starts from token IDs only for the first shard,
  starts from activation tensors for non-first shards, and emits activation
  tensors for non-final shards.
- The final shard applies final norm, output head, and sampling.
- Keep a route sticky for a generation session so each layer-owning peer owns
  the KV cache for its layers.

## Least-Invasive llama.cpp Architecture

Add a narrow Infernet bridge rather than replacing llama.cpp.

Proposed C API shape:

```c
struct llama_infernet_shard_params {
    uint32_t layer_start;
    uint32_t layer_end;
    bool include_token_embedding;
    bool include_lm_head;
};

struct llama_infernet_activation {
    uint32_t n_tokens;
    uint32_t n_embd;
    enum ggml_type type;
    void * data;
};

llama_model * llama_infernet_load_shard_from_file(
    const char * path,
    struct llama_model_params model_params,
    struct llama_infernet_shard_params shard_params);

int llama_infernet_eval_shard(
    llama_context * ctx,
    const struct llama_infernet_activation * input,
    struct llama_infernet_activation * output);
```

This API can live in an Infernet-maintained llama.cpp patch while the experiment
is unstable. Once the semantics are proven, it can be proposed upstream in a
smaller form.

Boundary behavior:

- First shard: owns tokenizer compatibility, token embedding, and layers
  `[0, end)`.
- Middle shard: owns only layers `[start, end)` and receives hidden activations.
- Final shard: owns layers `[start, n_layer)`, final norm, output head, and
  sampler state.
- Generation route: remains fixed across tokens unless the session is restarted.

## Shard Builder

The shard builder should be implemented before the patched runtime is complete.
It has two stages.

Stage 1 sidecar metadata:

- parse GGUF header and metadata;
- compute the GGUF checksum;
- enumerate tensor names and assign them to layer ranges;
- emit Infernet shard metadata as JSON.

Stage 1 output contains:

- `model_id`;
- `source_gguf_path`;
- `source_gguf_checksum`;
- `architecture`;
- `layer_count`;
- `hidden_size` when discoverable;
- `tokenizer_checksum` or tokenizer metadata hash;
- `layer_start`;
- `layer_end`;
- `required_tensors`;
- `boundary_tensors`;
- `runtime_kind = llama_cpp`;
- `protocol_version`.

Stage 2 physical shards:

- materialize a shard artifact that excludes tensors outside the layer range;
- include tokenizer metadata only when needed;
- preserve enough GGUF metadata for the partial llama.cpp loader;
- compute a shard checksum independent of the source file checksum.

Stage 1 is useful immediately because peer advertisements and scheduler route
construction can stop being demo-only. Stage 2 is required for the final
architecture proof because each worker must not possess the full model file.

## Runtime Plan

Execution request fields that become important for real GGUF:

- model identifier and compatibility hash;
- route;
- current hop index;
- token position;
- activation dtype and shape;
- tensor payload;
- trace id;
- session id for KV-cache ownership;
- sampling metadata when the final peer produces logits/token output.

Correctness-first constraints:

- Use one model architecture first.
- Use one activation dtype first, likely `f32` or `f16`.
- Keep JSON request-response for control metadata, but move tensor bytes to a
  binary payload before large real activations.
- Support one request at a time per worker before batching.
- Keep the same route for the whole generation session.

Initial real generation flow:

1. Scheduler builds a complete route for the selected model.
2. First shard tokenizes the prompt or receives token IDs from the client and
   creates the embedding/hidden activation.
3. Each shard executes its contiguous layer range and forwards hidden state.
4. Final shard applies output projection, samples the next token, and returns
   token/output plus metrics.
5. The same route repeats for additional decode tokens with per-shard KV cache
   state keyed by session id.

## Scheduler Changes

Users no longer choose Local or AI Grid.

Scheduler inputs:

- selected model;
- local hosted shards;
- discovered peer shards;
- shard checksums and tokenizer compatibility;
- latency hints;
- measured ping/transfer latency;
- worker load when available.

Route construction should prefer:

- complete layer coverage;
- compatible runtime kind and model checksum;
- fewer hops when latency is otherwise similar;
- local shards when they reduce total latency;
- peers with the exact required shard checksum;
- peers with lower recent failure rate.

The current greedy route builder remains acceptable for the first GGUF metadata
step, but it must filter against `manifest.runtime_kind`, not hard-code demo
runtime compatibility.

## Metrics

Every hop should log and expose:

- trace id;
- session id;
- peer id;
- model id;
- layer range;
- activation dtype;
- activation shape;
- activation byte size;
- serialization time;
- transfer latency;
- compute latency;
- end-to-end latency;
- per-layer compute timing when the runtime can expose it;
- tokens per second for decode;
- checksum of input/output activations for debugging.

The UI should visualize:

- model list and selected model;
- peer coverage for that model;
- route used for the current request;
- hop status and timing;
- activation byte sizes;
- final output;
- current limitations, especially trust/privacy.

## Current Limitations and Risks

- Stock llama.cpp cannot prove Infernet's model-ownership claim without a
  layer-range loader/evaluator patch or an independent GGUF execution engine.
- Distributed KV cache is the hard part after single-pass activation forwarding.
- Activation tensors may be large enough that JSON `Vec<f32>` becomes unusable;
  binary tensor frames are needed for real models.
- Peers can see intermediate activations. This milestone must not claim prompt
  privacy from participating workers.
- Physical sharding is required before claiming that no peer has the full model.
- The first implementation should fail loudly for unsupported GGUF execution
  instead of falling back to the demo runtime.

## Immediate Implementation Sequence

1. Add model metadata for `llama-3.2-1b` and derive imported model identities from GGUF metadata.
2. Add shard metadata structures with tokenizer/checksum/runtime information.
3. Add a shard-builder command that emits sidecar metadata for a GGUF layer
   range.
4. Update routing to support `RuntimeKind::LlamaCpp`.
5. Make runtime execution explicitly select demo vs llama.cpp. Until the
   layer-range bridge lands, the desktop app may download a complete verified
   GGUF source over `/infernet/model-blob/1` and run local llama.cpp token
   generation as a correctness fallback.
6. Remove UI execution modes and present models as the primary navigation.
7. Add the llama.cpp bridge behind a feature flag or separate crate.
8. Add binary activation frames and session/KV metadata.
9. Materialize physical shards and enforce workers load only their shard file.

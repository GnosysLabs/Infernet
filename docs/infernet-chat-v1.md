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

The first validation network is one RTX 3090 and two Apple Silicon MacBook
Pros. The 3090 has 24 GB of VRAM, so the 14.4 GB text payload nominally fits on
that GPU with room for runtime allocations and a conservative KV cache. Exact
Mac layer assignments remain dynamic until each Mac's chip and unified-memory
capacity are recorded.

The launch scheduler assigns contiguous layer ranges. The 3090 receives the
largest range; each Mac receives a smaller range based on measured memory and
throughput. A three-node route crosses only two network boundaries per token.

## Package contract

`infernet-chat-v1.infermodel` is the signed model and execution manifest, and
`infernet-chat-v1` is the only model ID accepted by the launch catalog. It
references content-addressed `.infershard` components.

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
4. The 3090 and both Macs complete a three-node prompt without a full model copy
   on either Mac.
5. Total unique package bytes satisfy the storage invariant.
6. Repeated runs survive peer restart, missing-component, timeout, and checksum
   failure tests without freezing a host.

Until partial Gemma 4 MoE execution passes those gates, the compatibility
package remains a complete `0:N` package for correctness testing and is not
presented as the distributed launch release.

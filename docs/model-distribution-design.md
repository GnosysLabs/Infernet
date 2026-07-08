# Infernet Model Distribution Design

Date: 2026-07-08

## Protocol Boundary

Infernet now has three separate protocol families:

- Peer discovery: mDNS plus shard advertisements over gossipsub.
- Model distribution: `/infernet/model/1`.
- Distributed inference: `/infernet/activation/1`.

Model transfer is independent from activation forwarding. Inference must not
assume required model shards are already local. A node can discover, download,
verify, cache, advertise, and serve model shards before it ever participates in
an inference route.

## Shard Metadata

Each advertised model shard includes:

- `model_id`
- `layer_start`
- `layer_end`
- `checksum`
- `size_bytes`
- `version`
- `protocol_version`

This metadata is published in `NodeAdvertisement.model_shards`. Existing
`hosted_shards` remains the inference-capability advertisement. A peer may host
model files without serving inference, serve inference without seeding the
underlying model file, or do both.

## Transfer Flow

1. A node imports an initial shard into its local cache.
2. The node runs a distribution seeder and advertises `model_shards`.
3. A downloader discovers peers that advertise the requested shard.
4. The downloader sends a `/infernet/model/1` request directly to a source peer.
5. The source peer reads the shard from its cache and returns metadata plus bytes.
6. The downloader verifies checksum and size before storing.
7. Once stored, the downloader advertises and serves the shard too.

The `model mirror` command demonstrates step 7 by downloading once and then
staying online as a seeder.

## Local Cache

The cache stores shard payloads by checksum and a JSON sidecar metadata record.
Current knobs:

- cache root
- maximum storage bytes
- preferred models
- pinned models
- automatic cleanup
- LRU eviction for unpinned shards

Cleanup is deliberately conservative. Pinned models are not evicted. Preferred
models are tracked in config now so replication and eviction policy can use them
later.

## Verification

Every received payload is verified before it is accepted:

- SHA-256 checksum must match advertised metadata.
- byte size must match advertised metadata.
- local reads re-check SHA-256 before serving.

The current trust root is the advertised checksum. Future model manifests should
pin expected shard checksums so a malicious first advertiser cannot define a fake
checksum for a fake model.

## Replication Health

The current UI estimates replica count from discovered `model_shards`
advertisements. It reports a fixed target of 10 replicas. This is not yet an
automatic replication algorithm.

Future scheduler hooks:

- identify under-replicated shards;
- ask idle preferred nodes to mirror shards;
- balance by region and latency;
- account for popularity and storage pressure;
- repair shards when replicas disappear.

## CLI

Import a seed shard:

```sh
cargo run -p infernet-worker -- model import \
  --cache-dir /tmp/infernet-a \
  --model grid-demo-12 \
  --layers 0:3 \
  --file /path/to/shard.bin \
  --version v1
```

Serve local cached shards:

```sh
cargo run -p infernet-worker -- model serve \
  --cache-dir /tmp/infernet-a
```

Download and exit:

```sh
cargo run -p infernet-worker -- model fetch \
  --cache-dir /tmp/infernet-b \
  --model grid-demo-12 \
  --layers 0:3 \
  --checksum <sha256>
```

Download and immediately become a seeder:

```sh
cargo run -p infernet-worker -- model mirror \
  --cache-dir /tmp/infernet-b \
  --model grid-demo-12 \
  --layers 0:3 \
  --checksum <sha256>
```

Manual fallback descriptor:

```text
peer@/ip4/127.0.0.1/tcp/7001/p2p/peer#grid-demo-12:0:3:<sha256>:<size>:v1
```

## Current Limitations

- Transfers use libp2p request-response JSON, so large real shards need a binary
  streaming codec before production use.
- Downloads are single-source and non-resumable.
- No bandwidth throttling yet.
- No automatic replication assignment yet.
- No signed model manifest trust root yet.
- WAN discovery still needs Kademlia/bootstrap work.

These limits are intentional for the first self-hosting proof. The implemented
milestone proves peer-to-peer discovery, transfer, verification, caching, and
re-seeding without a central file server after the initial seed.

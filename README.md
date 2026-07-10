# Infernet

Infernet is an experimental peer-to-peer split-inference system for local and
community-owned AI compute.

The core idea is simple: the app is always connected to an AI grid. Your
computer is one node in that grid. An official Infernet model can be present as
many physical layer shards spread across many machines, and the scheduler builds
an ordered route through the peers that can execute those layers.

The user experience should feel like a normal AI chat app. The network exists
under the surface. Users choose an official model and send a message; Infernet
discovers peers, verifies available shards, downloads only the executable shards
that machine should host, builds a route, forwards activations hop by hop, and
returns a response.

Infernet is **not** a general-purpose GGUF runner or model-import tool. The
launch product supports only curated, signed Infernet model packages. There is
no user workflow for adding a local file, pasting a Hugging Face URL, or
converting an arbitrary model. This narrow contract lets the project make a
small number of distributed models genuinely reliable before expanding its
official catalog.

The launch flagship is **Infernet Chat**, based on **Gemma 4 26B A4B Instruct
QAT Q4_0**. Its upstream identity is provenance; users interact with the tested
Infernet edition.

## Current State

This repository is not a production LLM runtime yet. It is a working prototype
of three separate peer-to-peer protocols:

1. Peer discovery
2. Model shard distribution
3. Distributed activation forwarding

The current implementation includes:

- libp2p peer discovery over mDNS, gossipsub, static peers, and a public
  bootstrap node.
- A Tauri desktop app with a chat-first UI, model library, downloads view, and
  optional network activity details.
- Infernet-native `.infermodel` manifests and verified `.infershard` execution
  packages for official models.
- A local shard cache with verified checksums and executable-shard metadata.
- P2P model shard transfer over `/infernet/model-blob/1`.
- Binary P2P activation forwarding over `/infernet/activation/2`.
- Route construction from live peer advertisements.
- Persistent Infernet workers that load assigned contiguous layer ranges from
  each node's verified local package and retain weights and KV state.

The compatibility package keeps one byte-exact model payload on each worker,
without storage amplification. At inference time the scheduler assigns each
worker only the layers its reported accelerator memory can hold. Model weights
never cross the inference transport: only binary activation frames and sampled
token state move between peers. Advanced scheduling, proofs of correct
execution, and privacy protections remain future work.

Infernet intentionally does not ship a fake bridge or fake peers by default. If
the real runtime cannot build or load, that is treated as a real error.

## What Infernet Enables

Infernet is trying to prove that a model does not have to live on one computer.

In the target architecture:

- Infernet's release pipeline publishes a curated and signed model edition.
- Seed nodes introduce only that official package to the network.
- Other nodes can discover those shards.
- Nodes can download only the shards they should host.
- Downloaders immediately become seeders.
- The network can route inference through the machines that collectively cover
  the model.
- No single participating node needs to possess the whole model forever.

Today, the prototype has the first version of that loop:

- A release engineer can build one complete executable `.infershard`
  compatibility package for an approved upstream artifact.
- Official packages are published, cached by seed nodes, and advertised to
  peers.
- Other peers can fetch shard files peer-to-peer and start serving them.
- The app only treats a peer as inference-capable when it advertises executable
  Infernet shards, not loose metadata.

## Architecture

Infernet is split into three protocols. They are intentionally separate.

### 1. Peer Discovery

Discovery is metadata only. It tells the app which peers exist and what they
claim to host.

Discovery uses libp2p:

- mDNS for LAN discovery.
- gossipsub for repeated node advertisements.
- static peers for manual testing and fallback.
- public bootstrap peers for internet rendezvous.

Each node advertises:

- `peer_id`
- listen addresses
- protocol version
- hosted executable shard descriptors
- cached model shard records
- model id
- layer range
- checksum
- shard version

Only executable shard descriptors are used to build inference routes.

### 2. Model Distribution

Model distribution is independent from inference. It moves model shard files
between peers.

The model protocols are:

```text
/infernet/model/1
/infernet/model-blob/1
```

`/infernet/model/1` exists for model metadata and legacy record exchange.

`/infernet/model-blob/1` transfers shard payload bytes in verified chunks. A
peer asks for a specific model id, layer range, checksum, version, offset, and
chunk size. The receiver responds with bytes from the matching local shard file.

Every downloaded shard is verified before it is accepted:

- checksum must match
- byte size must match
- protocol version must match
- layer range must match

After verification, the downloader stores the shard in its local cache and
immediately advertises it. That makes every downloader a seeder.

### 3. Distributed Inference

Inference uses a separate libp2p request-response protocol:

```text
/infernet/activation/2
```

The client builds a route from discovered executable shard descriptors. It sends
the first activation request to the first peer in that route. Each peer:

1. validates that the request layer range matches its hosted shard
2. loads/evaluates its local contiguous layer range
3. records trace information
4. forwards the activation to the next peer

The final peer samples one token and returns it through the request chain. The
client passes that token through the same sticky route until generation ends;
each worker keeps its local layer KV cache resident between passes.

Trace events include:

- trace id
- current peer id
- processed layer range
- next peer id
- activation size
- timing in milliseconds
- activation checksum

There is no direct TCP activation path in the default inference flow.

## Infernet Shards

This is the most important current subsystem.

An official model release consists of a model-level `.infermodel` manifest and
one or more executable `.infershard` packages. The manifest defines model
identity, version, provenance, compatibility, execution topology, component
references, sizes, and checksums. Peers use it to determine exactly which
verified components they need.

GGUF may be used as an upstream tensor artifact inside Infernet's **offline,
maintainer-only release engineering pipeline**. It is never accepted through
the app and is not a user-facing Infernet model format.

The logical release layout is:

```text
infernet-chat-v1.infermodel
components/
  component-000.infershard/
  component-001.infershard/
  ...
```

The `.infermodel` manifest is the signed catalog and execution contract. An
`.infershard` is a content-addressed execution component that a peer can verify,
cache, run, and seed independently. This separation lets a later release change
its physical component plan without teaching users about tensor formats.

### What A Shard Contains

An `.infershard` is a directory package. The current package layout is:

```text
<model>-<layer-range>-<hash>.infershard/
  manifest.json
  tensors.gguf
```

`manifest.json` records the Infernet shard format version, runtime ABI, model
id, layer range, checksum, source checksum, tokenizer compatibility, and payload
metadata.

Complete compatibility packages use payload kind `infernet-full-model` and
runtime ABI `infernet-llama-full-v1`. That explicit capability tag prevents an
older partial-graph peer from treating the package as a layer shard. Network
advertisements omit release-build machine paths and other private provenance
details.

`tensors.gguf` is the current compatibility payload. The release packager keeps
one complete `0:N` package. On APFS it can create an isolated copy-on-write
clone; other filesystems fall back to one ordinary copy. The cache never shares
a mutable hard-link inode with its upstream release artifact.

### What Is Duplicated

The release packager does not duplicate global tensors across per-layer packages.
The earlier one-layer planner copied large embeddings into every layer package;
that path is disabled because a 5 GB model could expand beyond 30 GB. Physical
multi-package splitting remains experimental until each architecture has a
verified partial graph and a storage-safe tensor plan.

### What Is Not A Shard

Infernet no longer treats these as executable model capacity:

- metadata-only seed records
- arbitrary cache payloads produced by legacy/debug tooling
- bare `model_shards` rows without an executable seed manifest
- stale cache entries whose physical file is missing

Those records may exist in older caches, but they do not count toward route
construction and they should not produce model cards in the app.

### Official Shard Planning

Users do not choose a shard size or package layout. Each official model version
ships with one layout designed and validated for its architecture and Infernet's
target hardware. The initial compatibility layout is intentionally conservative:

- demo models use small fixed ranges
- Infernet Chat initially uses a verified complete `0:N` executable package
- physical layer packages ship only after their partial graph, tensor ownership,
  routing behavior, and storage footprint pass release validation

This prevents the earlier failure mode where duplicating global tensors across
layer packages could turn a 5 GB source artifact into more than 30 GB of cache.
Future Infernet Chat releases can adopt a storage-safe component layout without
changing the user experience.

## Official Model Release Pipeline

This is release engineering tooling for Infernet maintainers, not an app or user
workflow:

1. A maintainer selects the pinned upstream artifact for an approved model.
2. The offline packager validates architecture, layer count, hidden size,
   quantization, tokenizer metadata, and licensing/provenance metadata.
3. It plans a supported component layout and rejects storage amplification.
4. It writes executable `.infershard` packages and an `.infermodel` manifest.
5. Release validation exercises loading, routing, generation, integrity checks,
   and the target hardware matrix, beginning with RTX 3090- and RTX 4060-class
   nodes plus two Apple Silicon MacBook Pro nodes.
6. The final manifest and components are checksummed, signed, versioned, and
   published to official seed nodes.

Normal Infernet nodes only discover, download, verify, cache, execute, and seed
those published packages. They never convert or import a model.

## What Happens On Another Machine

When another node sees a model on the network, it should not assume the model is
already installed locally.

The current behavior is:

- discovery finds peers advertising executable shards
- the app can show that the model is available on the network
- chat uses already-ready local or remote components and never starts a hidden
  multi-gigabyte download
- storing the current 14.4 GB compatibility package requires the explicit
  **Store 14.4 GB here** action; future smaller assigned components may be
  acquired in the background
- downloaded shards are verified
- downloaded shards are cached locally
- the node immediately begins seeding those shards
- route construction uses the union of local and remote executable shards

This is the beginning of Infernet becoming a self-hosting official model
repository. Once seed nodes publish an approved model version, other nodes can
obtain its verified shards from peers instead of repeatedly relying on a central
HTTP file server.

## Route Construction

The router builds an ordered sequence of peers for a requested model.

It uses only `hosted_shards` that are executable:

- model id must match
- runtime kind must match
- layer ranges must cover `0..layer_count`
- the descriptor must match a valid executable Infernet package manifest

If no full route exists, Infernet returns a clear missing-range error:

```text
no complete route for model gemma-4-12b-it-iq4-xs; missing layer ranges: 0:48
```

That error means the network does not currently have executable shard coverage
for the full model. It does not mean the model name exists somewhere in a cache.
It means route construction cannot find every required layer.

## Desktop App

The app lives in `infernet-ui` and uses Tauri, React, and TypeScript.

The launch UI is chat-first:

- Chat
- Network
- Settings

The official model catalog and storage flows appear contextually from Chat.

The app no longer exposes "Local" and "AI Grid" modes. Infernet is always a
network. Your computer is one node in that network.

### Chat

The chat screen is the primary interface. Network details are hidden until they
matter.

Activity opens a first-person HUD for this computer. It shows what this node is
doing now, whether its compute and model are ready, and an oldest-to-newest
session journal of completed chat, compute, model, and sharing work. Aggregate
capacity and the other nodes belong on the Network screen.

### Models

The Models page is the official Infernet catalog. It shows curated model
editions, beginning with:

- **Infernet Chat** — based on Gemma 4 26B A4B Instruct QAT Q4_0

Users can install or activate an available official model; they cannot add a
local file, enter a Hugging Face repository, or convert another format. Model
layout, signatures, runtime compatibility, and update policy are part of the
published Infernet edition.

### Downloads

The Downloads page explains storage and contribution state in useful product
language:

- official models stored on this computer
- active verified model transfers
- storage used
- whether this computer is making model components available to peers
- recent model activity

Checksums, layer ranges, package versions, and peer identifiers stay out of the
everyday product UI. Automatic replication and rebalancing are future work.

## Setup Requirements

Required for development:

- Rust toolchain
- Node.js and npm
- Git
- CMake
- C++ compiler toolchain

Infernet downloads official llama.cpp prebuilt binaries when available, but the
Infernet split-layer bridge is an Infernet patch and may need to be built
locally. Setup does not silently install OS-level build tools.

Windows:

```powershell
winget install --id Kitware.CMake -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --wait"
```

After installing, open a new PowerShell so `cmake`, `cl`, and Visual Studio
environment discovery are visible.

macOS:

```sh
xcode-select --install
brew install cmake
```

Debian/Ubuntu:

```sh
sudo apt-get update
sudo apt-get install -y git cmake build-essential
```

If a bridge dependency is missing, `npm run prepare-runtime` fails instead of
launching a degraded app.

## Running The Desktop App

Install dependencies:

```sh
npm --prefix infernet-ui install
```

Launch the real Tauri app node:

```sh
scripts/ui-demo.sh
```

On Windows PowerShell:

```powershell
.\scripts\ui-demo.ps1
```

This starts the desktop app with the real Rust backend. It does not start fake
demo peers.

The old four-peer demo is still available, but it is explicit:

```sh
scripts/ui-demo.sh --with-demo-peers
```

```powershell
.\scripts\ui-demo.ps1 -WithDemoPeers
```

For frontend-only visual work:

```sh
cd infernet-ui
npm install
npm run dev
```

The browser-only Vite view uses mock data. Use Tauri for real networking,
sharding, and inference commands.

## Maintainer Release Tooling

These commands exist for local protocol development and official release
engineering. They are not supported user workflows and must not be exposed by
the desktop app.

Prepare a pinned upstream GGUF as one complete compatibility package in an
isolated maintainer cache:

```sh
cargo run -p infernet-worker -- model add-local \
  --cache-dir /tmp/infernet-node \
  --model infernet-chat-v1 \
  --gguf /path/to/gemma-4-26B_q4_0-it.gguf \
  --version 1.0.0-compat.1
```

Serve cached model shards:

```sh
cargo run -p infernet-worker -- model serve \
  --cache-dir /tmp/infernet-node
```

Fetch a specific shard from peers:

```sh
cargo run -p infernet-worker -- model fetch \
  --cache-dir /tmp/infernet-peer \
  --model infernet-chat-v1 \
  --layers 0:30 \
  --checksum 4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5 \
  --version 1.0.0-compat.1
```

Download and immediately mirror:

```sh
cargo run -p infernet-worker -- model mirror \
  --cache-dir /tmp/infernet-peer \
  --model infernet-chat-v1 \
  --layers 0:30 \
  --checksum 4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5 \
  --version 1.0.0-compat.1
```

List local cache state:

```sh
cargo run -p infernet-worker -- model list \
  --cache-dir /tmp/infernet-peer
```

`model add-local` is locked to the pinned Infernet Chat release in normal
builds. Other low-level model commands are protocol test tooling; they do not
make arbitrary models part of the Infernet catalog. Production desktop nodes
accept only the exact official model ID, version, byte count, layer coverage,
and SHA-256 trust anchor.

## Toy Split-Inference Demo

The toy demo is still useful for protocol testing.

Run the repeatable dynamic discovery smoke test:

```sh
scripts/smoke-demo.sh "hello infernet"
```

On Windows PowerShell:

```powershell
.\scripts\smoke-demo.ps1 "hello infernet"
```

The script starts four independent workers, lets them discover one another over
libp2p, builds a route dynamically, and runs inference without a hardcoded peer
order.

Manual local run:

```sh
cargo run -p infernet-worker -- serve --model grid-demo-12 --layers 0:3
cargo run -p infernet-worker -- serve --model grid-demo-12 --layers 3:6
cargo run -p infernet-worker -- serve --model grid-demo-12 --layers 6:9
cargo run -p infernet-worker -- serve --model grid-demo-12 --layers 9:12
```

Inspect peers:

```sh
cargo run -p infernet-worker -- peers
```

Build a route:

```sh
cargo run -p infernet-worker -- route --model grid-demo-12
```

Run inference:

```sh
cargo run -p infernet-worker -- infer \
  --model grid-demo-12 \
  --prompt "hello infernet"
```

## LAN And Internet Discovery

mDNS works across machines on the same LAN when local firewalls allow it.

For LAN testing, bind libp2p to a dialable interface:

```sh
cargo run -p infernet-worker -- serve \
  --model grid-demo-12 \
  --layers 0:3 \
  --p2p-listen /ip4/0.0.0.0/tcp/7001
```

For internet discovery, the desktop app dials the public bootstrap node:

```text
/ip4/217.77.11.197/tcp/9777/p2p/12D3KooWRJrnpHPQTWdThpDGZMwRCHhEBL4JCAxFMwYMfFavxa2h
```

`infernet.gnosyslabs.xyz` is also configured as a DNS bootstrap address. The DNS
record must be DNS-only for raw libp2p TCP. If it is proxied through Cloudflare,
raw libp2p TCP will not work through that hostname.

Current WAN limitations:

- public bootstrap helps peers discover each other
- private NAT-to-NAT connectivity still needs circuit relay or hole punching
- firewalls can still block inbound model and activation requests
- multi-source downloads and resume are not implemented yet

## Troubleshooting

### An official model appears and then says "unknown model"

That means the UI saw a stale or incomplete advertisement. Current builds trust
executable `infernet-shard` manifests, with legacy `gguf-shard` records accepted
only for compatibility. Refresh the official package on the seed node and make
sure every machine trusts the same Infernet release manifest.

### A route is missing `0:N`

No peer is advertising executable shard coverage for those layers. Adding a
model name is not enough. At least one peer must have verified `.infershard`
packages for the required layer ranges.

### An official model download reaches 100% and keeps working

The transfer may be complete while integrity verification is still reading the
downloaded bytes. The activity view should identify this stage as “Checking
model integrity,” rather than presenting it as an import or leaving the user
with an unchanged download percentage.

### The app says there are peers but no runnable model

Peer connectivity and model availability are different. A peer only helps
inference if it advertises executable shard descriptors for the selected model.

### Runtime preparation fails

Install the missing CMake/C++ build tools or provide real bridge binaries with:

```sh
INFERNET_LLAMA_CLI=/path/to/llama-cli \
INFERNET_LLAMA_BRIDGE=/path/to/infernet-llama-bridge \
npm --prefix infernet-ui run prepare-runtime
```

Infernet does not fall back to a non-functional bridge.

## Current Limitations

Infernet is still a prototype.

Important limitations:

- Infernet Chat initially uses one complete compatibility package; physical
  split planning remains disabled until its Gemma architecture support and
  storage behavior are verified.
- The launch catalog contains only official curated packages. Arbitrary model
  import is intentionally outside the product scope.
- Nodes now advertise CUDA, Metal, or CPU capacity, free compute memory, CPU
  cores, session load, queue depth, and optional measured throughput.
- The fixed-component planner is memory- and load-aware: it reserves weights,
  KV cache, runtime scratch space, and a safety margin, then assigns contiguous
  ranges without rewriting or duplicating component bytes. Bandwidth-aware
  scheduling is still future work.
- WAN connectivity still needs relay/hole-punching for difficult NAT cases.
- Downloads are chunked and verified, but not resumable.
- Multi-source downloads are not implemented.
- Automatic replication and self-healing are not implemented.
- Distributed split execution is experimental and currently focused on
  contiguous layer ranges for explicitly supported official models.
- Persistent distributed KV-cache streaming for reliable multi-token chat is
  still the release gate. Until it lands, partial Gemma packages remain
  unavailable in the launch catalog even though capacity planning and placement
  are implemented.
- There is no cryptographic proof that a peer executed a layer correctly.
- There is no privacy layer; peers can inspect the activations they process.
- The trust model currently assumes participants are cooperative.

## Design Documents

Read these for deeper architecture notes:

- [docs/infernet-technical-design.md](docs/infernet-technical-design.md)
- [docs/infernet-chat-v1.md](docs/infernet-chat-v1.md)
- [docs/infernet-image-v1.md](docs/infernet-image-v1.md)
- [docs/gguf-split-inference-design.md](docs/gguf-split-inference-design.md)
- [docs/model-distribution-design.md](docs/model-distribution-design.md)

import type { GridSnapshot, LocalIdentity, ProgressEvent, RouteHopView } from "./types";

const peers = [
  {
    peerId: "12D3KooWKZbL6YdYvS5o8jffzk8tudQHfY8b9Ztp9LmDkA1f9a10",
    shortPeerId: "12D3KooW...f9a10",
    addresses: ["/ip4/127.0.0.1/tcp/63239/p2p/12D3KooWKZbL6YdYvS5o8jffzk8tudQHfY8b9Ztp9LmDkA1f9a10"],
    protocolVersion: 1,
    shards: [{ modelId: "grid-demo-12", layerStart: 0, layerEnd: 3 }],
  },
  {
    peerId: "12D3KooWAgjvMSzS57j4GsnWdxY5ef9oNAf4rs2pM7J3bQ9a1122",
    shortPeerId: "12D3KooW...a1122",
    addresses: ["/ip4/127.0.0.1/tcp/63240/p2p/12D3KooWAgjvMSzS57j4GsnWdxY5ef9oNAf4rs2pM7J3bQ9a1122"],
    protocolVersion: 1,
    shards: [{ modelId: "grid-demo-12", layerStart: 3, layerEnd: 6 }],
  },
  {
    peerId: "12D3KooWBV3ycoVdK4L2TuXNGwmaEZckVK9myuZogTn2Ru6b3344",
    shortPeerId: "12D3KooW...b3344",
    addresses: ["/ip4/127.0.0.1/tcp/63241/p2p/12D3KooWBV3ycoVdK4L2TuXNGwmaEZckVK9myuZogTn2Ru6b3344"],
    protocolVersion: 1,
    shards: [{ modelId: "grid-demo-12", layerStart: 6, layerEnd: 9 }],
  },
  {
    peerId: "12D3KooWPmvXuwgFcuQ7y56RtJgH8npCyQvFrw24Bg9AQ9c5566",
    shortPeerId: "12D3KooW...c5566",
    addresses: ["/ip4/127.0.0.1/tcp/63242/p2p/12D3KooWPmvXuwgFcuQ7y56RtJgH8npCyQvFrw24Bg9AQ9c5566"],
    protocolVersion: 1,
    shards: [{ modelId: "grid-demo-12", layerStart: 9, layerEnd: 12 }],
  },
];

export const sampleIdentity: LocalIdentity = {
  peerId: "12D3KooWLocalUiPeer9c4df4c6ec8d2b18a77f",
  topic: "infernet/grid-demo/1",
  listen: "/ip4/0.0.0.0/tcp/0",
};

export const sampleSnapshot: GridSnapshot = {
  localPeerId: sampleIdentity.peerId,
  topic: sampleIdentity.topic,
  selectedModel: "grid-demo-12",
  availableModels: [
    {
      modelId: "grid-llama-3.2-1b",
      displayName: "Grid Llama 3.2 1B",
      runtimeKind: "llama_cpp",
      layerCount: 16,
      activationDtype: "f16",
    },
    {
      modelId: "grid-demo-12",
      displayName: "Grid Demo 12",
      runtimeKind: "demo",
      layerCount: 12,
      activationDtype: "f32",
    },
  ],
  layerCount: 12,
  peers,
  route: peers.map((peer) => ({
    peerId: peer.peerId,
    shortPeerId: peer.shortPeerId,
    address: peer.addresses[0],
    layerStart: peer.shards[0].layerStart,
    layerEnd: peer.shards[0].layerEnd,
  })),
  missingRanges: null,
  coverage: Array.from({ length: 12 }, (_, layer) => {
    const owner = peers.find((peer) => {
      const shard = peer.shards[0];
      return shard.layerStart <= layer && layer < shard.layerEnd;
    });
    const shard = owner?.shards[0];
    return {
      layer,
      covered: Boolean(owner),
      peerId: owner?.peerId,
      layerStart: shard?.layerStart,
      layerEnd: shard?.layerEnd,
    };
  }),
  distribution: {
    installedModels: ["grid-demo-12"],
    installedShards: [
      {
        modelId: "grid-demo-12",
        layerStart: 0,
        layerEnd: 3,
        checksum: "76fc3428fc95ccd2652606c8690997376b939a63a5a1a946b6d0fa5e7cc3aaf8",
        sizeBytes: 28,
        version: "v1",
      },
    ],
    storageUsedBytes: 28,
    maxStorageBytes: 50 * 1024 * 1024 * 1024,
    currentUploads: 4,
    currentDownloads: 0,
    replicationHealth: [
      { modelId: "grid-demo-12", layerStart: 0, layerEnd: 3, replicas: 2, targetReplicas: 10 },
      { modelId: "grid-demo-12", layerStart: 3, layerEnd: 6, replicas: 1, targetReplicas: 10 },
    ],
  },
};

export async function runMockDemo(
  route: RouteHopView[],
  emit: (event: ProgressEvent) => void,
): Promise<string> {
  emit({ type: "routeDiscovered", route });

  const checksums = ["c9eb921ebdbaf96a", "f4352a6f0840b0b0", "68650fd2e5b6d760", "2ac2b3eefacb67db"];

  for (const [index, hop] of route.entries()) {
    emit({
      type: "hopStarted",
      traceId: "mock-trace",
      peerId: hop.peerId,
      shortPeerId: hop.shortPeerId,
      layerStart: hop.layerStart,
      layerEnd: hop.layerEnd,
      activationSizeBytes: 64,
    });
    await new Promise((resolve) => window.setTimeout(resolve, 260));
    emit({
      type: "hopCompleted",
      traceId: "mock-trace",
      peerId: hop.peerId,
      shortPeerId: hop.shortPeerId,
      layerStart: hop.layerStart,
      layerEnd: hop.layerEnd,
      nextPeerId: route[index + 1]?.peerId ?? null,
      activationSizeBytes: 64,
      timingMs: 11 + index * 4,
      activationChecksum: checksums[index] ?? checksums[0],
    });
  }

  const output = "infernet-demo-2ac2b3eefacb67db";
  emit({ type: "finalOutput", traceId: "mock-trace", output });
  return output;
}

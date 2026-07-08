use infernet_model::{LayerRange, ModelManifest, RuntimeKind, ShardDescriptor};
use infernet_protocol::NodeAdvertisement;
use infernet_router::{MissingRanges, RouterError, ShardRegistry};

fn advertisement(peer_id: &str, model_id: &str, layers: LayerRange) -> NodeAdvertisement {
    NodeAdvertisement {
        protocol_version: 1,
        peer_id: peer_id.to_owned(),
        addresses: vec![format!("127.0.0.1:{}", 7000 + layers.start)],
        available_ram_bytes: None,
        available_vram_bytes: None,
        latency_hint_ms: None,
        hosted_shards: vec![ShardDescriptor {
            model_id: model_id.to_owned(),
            layers,
            runtime_kind: RuntimeKind::Demo,
            tokenizer: None,
            metadata: None,
            shard_hash: None,
        }],
        model_shards: Vec::new(),
    }
}

#[test]
fn route_error_lists_missing_layer_ranges() {
    let manifest = ModelManifest::demo();
    let mut registry = ShardRegistry::new();

    registry.upsert(advertisement(
        "peer-a",
        &manifest.model_id,
        LayerRange::new(0, 3).unwrap(),
    ));
    registry.upsert(advertisement(
        "peer-c",
        &manifest.model_id,
        LayerRange::new(6, 9).unwrap(),
    ));

    assert_eq!(
        registry.route_for_model(&manifest),
        Err(RouterError::MissingRanges {
            model_id: manifest.model_id,
            missing_ranges: MissingRanges(vec![
                LayerRange::new(3, 6).unwrap(),
                LayerRange::new(9, 12).unwrap(),
            ])
        })
    );
}

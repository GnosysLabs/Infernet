use std::{
    collections::BTreeMap,
    fmt,
    time::{Duration, Instant},
};

use infernet_model::{LayerRange, ModelError, ModelManifest, validate_contiguous_coverage};
use infernet_protocol::{NodeAdvertisement, RouteHop};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingRanges(pub Vec<LayerRange>);

impl fmt::Display for MissingRanges {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, range) in self.0.iter().enumerate() {
            if index > 0 {
                write!(formatter, ", ")?;
            }

            write!(formatter, "{}:{}", range.start, range.end)?;
        }

        Ok(())
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RouterError {
    #[error("no complete route for model {model_id}; missing layer ranges: {missing_ranges}")]
    MissingRanges {
        model_id: String,
        missing_ranges: MissingRanges,
    },
    #[error("route has invalid coverage: {0}")]
    InvalidCoverage(#[from] ModelError),
}

#[derive(Debug, Clone)]
pub struct ShardRegistry {
    advertisements: BTreeMap<String, RegistryEntry>,
    ttl: Duration,
}

#[derive(Debug, Clone)]
struct RegistryEntry {
    advertisement: NodeAdvertisement,
    seen_at: Instant,
}

impl Default for ShardRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardRegistry {
    pub fn new() -> Self {
        Self::with_ttl(Duration::from_secs(45))
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            advertisements: BTreeMap::new(),
            ttl,
        }
    }

    pub fn upsert(&mut self, advertisement: NodeAdvertisement) {
        self.advertisements.insert(
            advertisement.peer_id.clone(),
            RegistryEntry {
                advertisement,
                seen_at: Instant::now(),
            },
        );
    }

    pub fn merge(&mut self, advertisement: NodeAdvertisement) {
        self.advertisements
            .entry(advertisement.peer_id.clone())
            .and_modify(|existing| {
                merge_advertisement(&mut existing.advertisement, &advertisement);
                existing.seen_at = Instant::now();
            })
            .or_insert_with(|| RegistryEntry {
                advertisement,
                seen_at: Instant::now(),
            });
    }

    pub fn extend(&mut self, advertisements: impl IntoIterator<Item = NodeAdvertisement>) {
        for advertisement in advertisements {
            self.upsert(advertisement);
        }
    }

    pub fn len(&self) -> usize {
        self.advertisements().len()
    }

    pub fn is_empty(&self) -> bool {
        self.advertisements().is_empty()
    }

    pub fn advertisements(&self) -> Vec<NodeAdvertisement> {
        let now = Instant::now();
        self.advertisements
            .values()
            .filter(|entry| now.duration_since(entry.seen_at) <= self.ttl)
            .map(|entry| entry.advertisement.clone())
            .collect()
    }

    pub fn route_for_model(&self, manifest: &ModelManifest) -> Result<Vec<RouteHop>, RouterError> {
        build_greedy_route(manifest, &self.advertisements())
    }
}

fn merge_advertisement(existing: &mut NodeAdvertisement, advertisement: &NodeAdvertisement) {
    for address in &advertisement.addresses {
        if !existing.addresses.contains(address) {
            existing.addresses.push(address.clone());
        }
    }

    for shard in &advertisement.hosted_shards {
        if let Some(existing_shard) = existing.hosted_shards.iter_mut().find(|existing| {
            existing.model_id == shard.model_id
                && existing.layers == shard.layers
                && existing.runtime_kind == shard.runtime_kind
        }) {
            if existing_shard.seed_manifest.is_none() && shard.seed_manifest.is_some() {
                existing_shard.seed_manifest = shard.seed_manifest.clone();
            }
        } else {
            existing.hosted_shards.push(shard.clone());
        }
    }

    for shard in &advertisement.model_shards {
        if !existing
            .model_shards
            .iter()
            .any(|existing| existing == shard)
        {
            existing.model_shards.push(shard.clone());
        }
    }

    if advertisement.available_ram_bytes.is_some() {
        existing.available_ram_bytes = advertisement.available_ram_bytes;
    }
    if advertisement.available_vram_bytes.is_some() {
        existing.available_vram_bytes = advertisement.available_vram_bytes;
    }
    if advertisement.latency_hint_ms.is_some() {
        existing.latency_hint_ms = advertisement.latency_hint_ms;
    }
}

pub fn build_greedy_route(
    manifest: &ModelManifest,
    advertisements: &[NodeAdvertisement],
) -> Result<Vec<RouteHop>, RouterError> {
    let missing_ranges = missing_layer_ranges(manifest, advertisements);
    if !missing_ranges.is_empty() {
        return Err(RouterError::MissingRanges {
            model_id: manifest.model_id.clone(),
            missing_ranges: MissingRanges(missing_ranges),
        });
    }

    let mut cursor = 0;
    let mut route = Vec::new();

    while cursor < manifest.layer_count {
        let candidate = advertisements
            .iter()
            .flat_map(|advertisement| {
                advertisement
                    .hosted_shards
                    .iter()
                    .map(move |shard| (advertisement, shard))
            })
            .filter(|(_, shard)| {
                shard.model_id == manifest.model_id
                    && shard.runtime_kind == manifest.runtime_kind
                    && shard.layers.start <= cursor
                    && shard.layers.end > cursor
                    && shard.layers.end <= manifest.layer_count
            })
            .min_by_key(|(advertisement, shard)| {
                (
                    advertisement.latency_hint_ms.unwrap_or(u32::MAX),
                    u32::MAX - shard.layers.end,
                )
            })
            .expect("missing ranges are checked before route construction");

        let (advertisement, shard) = candidate;
        let address = advertisement.addresses.first().cloned().unwrap_or_default();
        let layers = LayerRange::new(cursor, shard.layers.end)?;

        route.push(RouteHop {
            peer_id: advertisement.peer_id.clone(),
            address,
            layers,
        });

        cursor = shard.layers.end;
    }

    validate_contiguous_coverage(manifest.layer_count, route.iter().map(|hop| hop.layers))?;

    Ok(route)
}

pub fn route_ranges(route: &[RouteHop]) -> Vec<LayerRange> {
    route.iter().map(|hop| hop.layers).collect()
}

pub fn missing_layer_ranges(
    manifest: &ModelManifest,
    advertisements: &[NodeAdvertisement],
) -> Vec<LayerRange> {
    let mut ranges = advertisements
        .iter()
        .flat_map(|advertisement| advertisement.hosted_shards.iter())
        .filter(|shard| {
            shard.model_id == manifest.model_id && shard.runtime_kind == manifest.runtime_kind
        })
        .filter_map(|shard| {
            let start = shard.layers.start.min(manifest.layer_count);
            let end = shard.layers.end.min(manifest.layer_count);

            (start < end).then_some(LayerRange { start, end })
        })
        .collect::<Vec<_>>();

    ranges.sort_by_key(|range| (range.start, range.end));

    let mut cursor = 0;
    let mut missing = Vec::new();

    for range in ranges {
        if range.end <= cursor {
            continue;
        }

        if range.start > cursor {
            missing.push(LayerRange {
                start: cursor,
                end: range.start,
            });
        }

        cursor = cursor.max(range.end);
    }

    if cursor < manifest.layer_count {
        missing.push(LayerRange {
            start: cursor,
            end: manifest.layer_count,
        });
    }

    missing
}

#[cfg(test)]
mod tests {
    use super::*;
    use infernet_model::{RuntimeKind, ShardDescriptor};

    fn advertisement(peer_id: &str, model_id: &str, layers: LayerRange) -> NodeAdvertisement {
        NodeAdvertisement {
            protocol_version: 1,
            peer_id: peer_id.to_owned(),
            addresses: vec![format!("127.0.0.1:70{}", layers.start)],
            available_ram_bytes: None,
            available_vram_bytes: None,
            latency_hint_ms: Some(layers.start + 1),
            hosted_shards: vec![ShardDescriptor {
                model_id: model_id.to_owned(),
                layers,
                runtime_kind: RuntimeKind::Demo,
                tokenizer: None,
                metadata: None,
                shard_hash: None,
                seed_manifest: None,
            }],
            model_shards: Vec::new(),
        }
    }

    fn advertisement_for_manifest(
        peer_id: &str,
        manifest: &ModelManifest,
        layers: LayerRange,
    ) -> NodeAdvertisement {
        NodeAdvertisement {
            protocol_version: 1,
            peer_id: peer_id.to_owned(),
            addresses: vec![format!("127.0.0.1:80{}", layers.start)],
            available_ram_bytes: None,
            available_vram_bytes: None,
            latency_hint_ms: Some(layers.start + 1),
            hosted_shards: vec![ShardDescriptor::for_manifest(manifest, layers)],
            model_shards: Vec::new(),
        }
    }

    #[test]
    fn builds_route_for_demo_model() {
        let manifest = ModelManifest::demo();
        let ads = [0, 3, 6, 9]
            .into_iter()
            .map(|start| {
                let end = start + 3;
                advertisement(
                    &format!("peer-{start}"),
                    &manifest.model_id,
                    LayerRange::new(start, end).unwrap(),
                )
            })
            .collect::<Vec<_>>();

        let route = build_greedy_route(&manifest, &ads).unwrap();

        assert_eq!(route.len(), 4);
        assert_eq!(route[0].layers, LayerRange::new(0, 3).unwrap());
        assert_eq!(route[3].layers, LayerRange::new(9, 12).unwrap());
    }

    #[test]
    fn rejects_missing_layer() {
        let manifest = ModelManifest::demo();
        let ads = vec![advertisement(
            "peer-0",
            &manifest.model_id,
            LayerRange::new(0, 3).unwrap(),
        )];

        assert_eq!(
            build_greedy_route(&manifest, &ads),
            Err(RouterError::MissingRanges {
                model_id: manifest.model_id,
                missing_ranges: MissingRanges(vec![LayerRange::new(3, 12).unwrap()])
            })
        );
    }

    #[test]
    fn reports_all_missing_ranges() {
        let manifest = ModelManifest::demo();
        let ads = vec![
            advertisement("peer-0", &manifest.model_id, LayerRange::new(0, 3).unwrap()),
            advertisement("peer-6", &manifest.model_id, LayerRange::new(6, 9).unwrap()),
        ];

        assert_eq!(
            missing_layer_ranges(&manifest, &ads),
            vec![
                LayerRange::new(3, 6).unwrap(),
                LayerRange::new(9, 12).unwrap()
            ]
        );
    }

    #[test]
    fn registry_builds_route() {
        let manifest = ModelManifest::demo();
        let mut registry = ShardRegistry::new();

        for start in [0, 3, 6, 9] {
            registry.upsert(advertisement(
                &format!("peer-{start}"),
                &manifest.model_id,
                LayerRange::new(start, start + 3).unwrap(),
            ));
        }

        assert_eq!(registry.route_for_model(&manifest).unwrap().len(), 4);
    }

    #[test]
    fn registry_expires_stale_advertisements() {
        let manifest = ModelManifest::demo();
        let mut registry = ShardRegistry::with_ttl(Duration::from_millis(1));
        registry.upsert(advertisement(
            "peer-0",
            &manifest.model_id,
            LayerRange::new(0, 12).unwrap(),
        ));

        std::thread::sleep(Duration::from_millis(5));

        assert!(registry.advertisements().is_empty());
        assert!(registry.route_for_model(&manifest).is_err());
    }

    #[test]
    fn builds_route_for_llama_cpp_model() {
        let manifest = ModelManifest::llama32_1b();
        let ads = [0, 4, 8, 12]
            .into_iter()
            .map(|start| {
                advertisement_for_manifest(
                    &format!("llama-peer-{start}"),
                    &manifest,
                    LayerRange::new(start, start + 4).unwrap(),
                )
            })
            .collect::<Vec<_>>();

        let route = build_greedy_route(&manifest, &ads).unwrap();

        assert_eq!(route.len(), 4);
        assert_eq!(route[0].layers, LayerRange::new(0, 4).unwrap());
        assert_eq!(route[3].layers, LayerRange::new(12, 16).unwrap());
    }
}

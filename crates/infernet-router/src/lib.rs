use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fmt,
    time::{Duration, Instant},
};

use infernet_model::{
    LayerRange, ModelError, ModelManifest, ShardDescriptor, validate_contiguous_coverage,
};
use infernet_protocol::{NodeAdvertisement, NodeCapabilities, RouteHop};
use thiserror::Error;

const BASIS_POINTS: u64 = 10_000;

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

/// An immutable, content-addressed model component available for placement.
///
/// The capacity planner assigns these components as-is. It never splits,
/// rewrites, or estimates a new payload size at routing time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedModelComponent {
    pub content_hash: String,
    pub layers: LayerRange,
    pub weight_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityPlanningConfig {
    /// Memory retained for the active KV cache for each transformer layer.
    pub kv_cache_bytes_per_layer: u64,
    /// Per-worker runtime workspace, allocator, and kernel scratch memory.
    pub scratch_bytes_per_peer: u64,
    /// An absolute safety reserve. The larger of this and the percentage
    /// reserve is used.
    pub safety_margin_bytes: u64,
    /// Percentage safety reserve in basis points (1_000 = 10%).
    pub safety_margin_basis_points: u16,
    /// Product policy can require distribution across more than one peer.
    /// The planner still uses the fewest boundaries at or above this value.
    pub minimum_peer_count: usize,
}

impl Default for CapacityPlanningConfig {
    fn default() -> Self {
        Self {
            kv_cache_bytes_per_layer: 0,
            scratch_bytes_per_peer: 0,
            safety_margin_bytes: 0,
            safety_margin_basis_points: 1_000,
            minimum_peer_count: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentAssignment {
    pub peer_id: String,
    pub address: String,
    pub layers: LayerRange,
    pub component_hashes: Vec<String>,
    pub weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub scratch_bytes: u64,
    pub safety_bytes: u64,
    pub total_reserved_bytes: u64,
    pub reported_available_memory_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapacityPlan {
    pub assignments: Vec<ComponentAssignment>,
}

impl CapacityPlan {
    pub fn route(&self) -> Vec<RouteHop> {
        self.assignments
            .iter()
            .map(|assignment| RouteHop {
                peer_id: assignment.peer_id.clone(),
                address: assignment.address.clone(),
                layers: assignment.layers,
            })
            .collect()
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CapacityPlannerError {
    #[error("cannot plan an empty component set")]
    EmptyComponents,
    #[error("component {index} has no content hash")]
    MissingContentHash { index: usize },
    #[error("component {index} has an empty payload")]
    EmptyComponent { index: usize },
    #[error("fixed components do not provide contiguous model coverage: {0}")]
    InvalidCoverage(#[from] ModelError),
    #[error("no peer reports usable compute memory and an available session")]
    NoEligiblePeers,
    #[error(
        "minimum peer count {requested} cannot be met by {eligible} eligible peers for {components} components"
    )]
    MinimumPeerCountUnavailable {
        requested: usize,
        eligible: usize,
        components: usize,
    },
    #[error("no capacity-safe contiguous placement exists for the fixed components")]
    NoFeasiblePlacement,
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
    if advertisement.capabilities.is_some() {
        existing.capabilities = advertisement.capabilities.clone();
    }
}

#[derive(Debug, Clone, Copy)]
struct EligiblePeer<'a> {
    advertisement: &'a NodeAdvertisement,
    available_memory_bytes: u64,
    scratch_bytes: u64,
    safety_bytes: u64,
    effective_throughput: f64,
}

#[derive(Debug, Clone)]
struct WorkingAssignment {
    assignment: ComponentAssignment,
    effective_throughput: f64,
    latency_ms: u32,
}

#[derive(Debug)]
struct ScoredPlan {
    plan: CapacityPlan,
    maximum_work_per_throughput: f64,
    total_work_per_throughput: f64,
    total_latency_ms: u64,
    stable_key: Vec<(String, u32, u32)>,
}

/// Places immutable model components onto peers with reported capacity.
///
/// Components remain indivisible and retain their content hashes. Each peer is
/// assigned at most one contiguous run. Plans are ranked first by the number
/// of peer boundaries, then by the measured-throughput/load balance, and then
/// by latency. `minimum_peer_count` lets product policy require a genuinely
/// distributed route while retaining the fewest possible hand-offs.
pub fn plan_fixed_components(
    components: &[FixedModelComponent],
    advertisements: &[NodeAdvertisement],
    config: CapacityPlanningConfig,
) -> Result<CapacityPlan, CapacityPlannerError> {
    validate_fixed_components(components)?;

    let peers = eligible_peers(advertisements, config);
    if peers.is_empty() {
        return Err(CapacityPlannerError::NoEligiblePeers);
    }

    let minimum_peer_count = config.minimum_peer_count.max(1);
    if minimum_peer_count > peers.len() || minimum_peer_count > components.len() {
        return Err(CapacityPlannerError::MinimumPeerCountUnavailable {
            requested: minimum_peer_count,
            eligible: peers.len(),
            components: components.len(),
        });
    }

    let mut search = PlacementSearch {
        components,
        peers: &peers,
        config,
        minimum_peer_count,
        used_peers: vec![false; peers.len()],
        assignments: Vec::new(),
        best: None,
    };
    search.visit(0);

    search
        .best
        .map(|best| best.plan)
        .ok_or(CapacityPlannerError::NoFeasiblePlacement)
}

fn validate_fixed_components(
    components: &[FixedModelComponent],
) -> Result<(), CapacityPlannerError> {
    if components.is_empty() {
        return Err(CapacityPlannerError::EmptyComponents);
    }

    for (index, component) in components.iter().enumerate() {
        if component.content_hash.trim().is_empty() {
            return Err(CapacityPlannerError::MissingContentHash { index });
        }
        if component.weight_bytes == 0 {
            return Err(CapacityPlannerError::EmptyComponent { index });
        }
    }

    let layer_count = components
        .last()
        .expect("empty components are rejected above")
        .layers
        .end;
    validate_contiguous_coverage(
        layer_count,
        components.iter().map(|component| component.layers),
    )?;

    Ok(())
}

fn eligible_peers<'a>(
    advertisements: &'a [NodeAdvertisement],
    config: CapacityPlanningConfig,
) -> Vec<EligiblePeer<'a>> {
    let mut by_peer_id = BTreeMap::<String, EligiblePeer<'a>>::new();

    for advertisement in advertisements {
        let Some(available_memory_bytes) = reported_available_compute_memory(advertisement) else {
            continue;
        };

        if available_memory_bytes == 0 || !session_available(advertisement.capabilities.as_ref()) {
            continue;
        }

        let percentage_safety = available_memory_bytes
            .saturating_mul(u64::from(config.safety_margin_basis_points))
            / BASIS_POINTS;
        let safety_bytes = config.safety_margin_bytes.max(percentage_safety);
        let fixed_reserve = safety_bytes.saturating_add(config.scratch_bytes_per_peer);
        if fixed_reserve >= available_memory_bytes {
            continue;
        }

        let candidate = EligiblePeer {
            advertisement,
            available_memory_bytes,
            scratch_bytes: config.scratch_bytes_per_peer,
            safety_bytes,
            effective_throughput: effective_throughput(advertisement.capabilities.as_ref()),
        };

        by_peer_id
            .entry(advertisement.peer_id.clone())
            .and_modify(|existing| {
                if candidate.available_memory_bytes > existing.available_memory_bytes
                    || (candidate.available_memory_bytes == existing.available_memory_bytes
                        && candidate.effective_throughput > existing.effective_throughput)
                {
                    *existing = candidate;
                }
            })
            .or_insert(candidate);
    }

    let mut peers = by_peer_id.into_values().collect::<Vec<_>>();
    peers.sort_by(|left, right| {
        right
            .effective_throughput
            .total_cmp(&left.effective_throughput)
            .then_with(|| {
                right
                    .available_memory_bytes
                    .cmp(&left.available_memory_bytes)
            })
            .then_with(|| left.advertisement.peer_id.cmp(&right.advertisement.peer_id))
    });
    peers
}

struct PlacementSearch<'a> {
    components: &'a [FixedModelComponent],
    peers: &'a [EligiblePeer<'a>],
    config: CapacityPlanningConfig,
    minimum_peer_count: usize,
    used_peers: Vec<bool>,
    assignments: Vec<WorkingAssignment>,
    best: Option<ScoredPlan>,
}

impl PlacementSearch<'_> {
    fn visit(&mut self, cursor: usize) {
        if cursor == self.components.len() {
            if self.assignments.len() >= self.minimum_peer_count {
                let candidate = score_plan(&self.assignments);
                if self
                    .best
                    .as_ref()
                    .is_none_or(|best| scored_plan_is_better(&candidate, best))
                {
                    self.best = Some(candidate);
                }
            }
            return;
        }

        if self
            .best
            .as_ref()
            .is_some_and(|best| self.assignments.len() >= best.plan.assignments.len())
        {
            return;
        }

        let unused_peer_count = self.used_peers.iter().filter(|used| !**used).count();
        if self.assignments.len() + unused_peer_count < self.minimum_peer_count {
            return;
        }

        let minimum_remaining_assignments = self
            .minimum_peer_count
            .saturating_sub(self.assignments.len() + 1);
        let last_end = self
            .components
            .len()
            .saturating_sub(minimum_remaining_assignments);

        for peer_index in 0..self.peers.len() {
            if self.used_peers[peer_index] {
                continue;
            }

            self.used_peers[peer_index] = true;
            for end in (cursor + 1..=last_end).rev() {
                let Some(assignment) = build_assignment(
                    self.components,
                    cursor,
                    end,
                    self.peers[peer_index],
                    self.config,
                ) else {
                    continue;
                };

                self.assignments.push(assignment);
                self.visit(end);
                self.assignments.pop();
            }
            self.used_peers[peer_index] = false;
        }
    }
}

fn build_assignment(
    components: &[FixedModelComponent],
    start: usize,
    end: usize,
    peer: EligiblePeer<'_>,
    config: CapacityPlanningConfig,
) -> Option<WorkingAssignment> {
    let selected = components.get(start..end)?;
    let first = selected.first()?;
    let last = selected.last()?;
    let layer_count = last.layers.end.saturating_sub(first.layers.start);
    let weight_bytes = selected.iter().fold(0_u64, |total, component| {
        total.saturating_add(component.weight_bytes)
    });
    let kv_cache_bytes = config
        .kv_cache_bytes_per_layer
        .saturating_mul(u64::from(layer_count));
    let total_reserved_bytes = weight_bytes
        .saturating_add(kv_cache_bytes)
        .saturating_add(peer.scratch_bytes)
        .saturating_add(peer.safety_bytes);

    if total_reserved_bytes > peer.available_memory_bytes {
        return None;
    }

    Some(WorkingAssignment {
        assignment: ComponentAssignment {
            peer_id: peer.advertisement.peer_id.clone(),
            address: peer
                .advertisement
                .addresses
                .first()
                .cloned()
                .unwrap_or_default(),
            layers: LayerRange {
                start: first.layers.start,
                end: last.layers.end,
            },
            component_hashes: selected
                .iter()
                .map(|component| component.content_hash.clone())
                .collect(),
            weight_bytes,
            kv_cache_bytes,
            scratch_bytes: peer.scratch_bytes,
            safety_bytes: peer.safety_bytes,
            total_reserved_bytes,
            reported_available_memory_bytes: peer.available_memory_bytes,
        },
        effective_throughput: peer.effective_throughput,
        latency_ms: peer.advertisement.latency_hint_ms.unwrap_or(u32::MAX),
    })
}

fn score_plan(assignments: &[WorkingAssignment]) -> ScoredPlan {
    let mut maximum_work_per_throughput = 0.0_f64;
    let mut total_work_per_throughput = 0.0_f64;
    let mut total_latency_ms = 0_u64;

    for assignment in assignments {
        let work_bytes = assignment
            .assignment
            .weight_bytes
            .saturating_add(assignment.assignment.kv_cache_bytes);
        let work_per_throughput = work_bytes as f64 / assignment.effective_throughput.max(0.001);
        maximum_work_per_throughput = maximum_work_per_throughput.max(work_per_throughput);
        total_work_per_throughput += work_per_throughput;
        total_latency_ms = total_latency_ms.saturating_add(u64::from(assignment.latency_ms));
    }

    ScoredPlan {
        plan: CapacityPlan {
            assignments: assignments
                .iter()
                .map(|working| working.assignment.clone())
                .collect(),
        },
        maximum_work_per_throughput,
        total_work_per_throughput,
        total_latency_ms,
        stable_key: assignments
            .iter()
            .map(|assignment| {
                (
                    assignment.assignment.peer_id.clone(),
                    assignment.assignment.layers.start,
                    assignment.assignment.layers.end,
                )
            })
            .collect(),
    }
}

fn scored_plan_is_better(candidate: &ScoredPlan, current: &ScoredPlan) -> bool {
    candidate
        .plan
        .assignments
        .len()
        .cmp(&current.plan.assignments.len())
        .then_with(|| {
            candidate
                .maximum_work_per_throughput
                .total_cmp(&current.maximum_work_per_throughput)
        })
        .then_with(|| {
            candidate
                .total_work_per_throughput
                .total_cmp(&current.total_work_per_throughput)
        })
        .then_with(|| candidate.total_latency_ms.cmp(&current.total_latency_ms))
        .then_with(|| candidate.stable_key.cmp(&current.stable_key))
        == Ordering::Less
}

fn session_available(capabilities: Option<&NodeCapabilities>) -> bool {
    capabilities.is_none_or(|capabilities| {
        capabilities.max_sessions > 0 && capabilities.active_sessions < capabilities.max_sessions
    })
}

fn reported_available_compute_memory(advertisement: &NodeAdvertisement) -> Option<u64> {
    if let Some(capabilities) = advertisement.capabilities.as_ref() {
        return Some(if capabilities.available_accelerator_memory_bytes > 0 {
            capabilities.available_accelerator_memory_bytes
        } else {
            capabilities.available_ram_bytes
        });
    }

    advertisement
        .available_vram_bytes
        .or(advertisement.available_ram_bytes)
}

fn measured_throughput(capabilities: &NodeCapabilities) -> Option<f64> {
    let prefill = positive_metric(capabilities.measured_prefill_tokens_per_second);
    let decode = positive_metric(capabilities.measured_decode_tokens_per_second);

    match (prefill, decode) {
        (Some(prefill), Some(decode)) => Some(2.0 / (prefill.recip() + decode.recip())),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

fn positive_metric(value: Option<f32>) -> Option<f64> {
    value
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(f64::from)
}

fn effective_throughput(capabilities: Option<&NodeCapabilities>) -> f64 {
    let Some(capabilities) = capabilities else {
        return 1.0;
    };

    let measured = measured_throughput(capabilities).unwrap_or(1.0);
    let session_headroom = if capabilities.max_sessions == 0 {
        0.0
    } else {
        f64::from(
            capabilities
                .max_sessions
                .saturating_sub(capabilities.active_sessions),
        ) / f64::from(capabilities.max_sessions)
    };
    let queue_factor = 1.0 / (1.0 + f64::from(capabilities.queue_depth));

    measured * session_headroom.max(0.05) * queue_factor
}

fn compare_route_candidates(
    left: &(&NodeAdvertisement, &ShardDescriptor),
    right: &(&NodeAdvertisement, &ShardDescriptor),
) -> Ordering {
    if left.0.capabilities.is_none() && right.0.capabilities.is_none() {
        return legacy_route_candidate_cmp(left, right);
    }

    capability_availability_rank(left.0)
        .cmp(&capability_availability_rank(right.0))
        .then_with(|| available_session_slots(right.0).cmp(&available_session_slots(left.0)))
        .then_with(|| {
            compare_optional_metric_desc(
                left.0.capabilities.as_ref().and_then(measured_throughput),
                right.0.capabilities.as_ref().and_then(measured_throughput),
            )
        })
        .then_with(|| capability_load(left.0).total_cmp(&capability_load(right.0)))
        .then_with(|| {
            reported_available_compute_memory(right.0)
                .unwrap_or_default()
                .cmp(&reported_available_compute_memory(left.0).unwrap_or_default())
        })
        .then_with(|| legacy_route_candidate_cmp(left, right))
}

fn legacy_route_candidate_cmp(
    left: &(&NodeAdvertisement, &ShardDescriptor),
    right: &(&NodeAdvertisement, &ShardDescriptor),
) -> Ordering {
    left.0
        .latency_hint_ms
        .unwrap_or(u32::MAX)
        .cmp(&right.0.latency_hint_ms.unwrap_or(u32::MAX))
        .then_with(|| right.1.layers.end.cmp(&left.1.layers.end))
}

fn capability_availability_rank(advertisement: &NodeAdvertisement) -> u8 {
    match advertisement.capabilities.as_ref() {
        Some(capabilities)
            if capabilities.max_sessions > 0
                && capabilities.active_sessions < capabilities.max_sessions =>
        {
            0
        }
        None => 1,
        Some(_) => 2,
    }
}

fn available_session_slots(advertisement: &NodeAdvertisement) -> u32 {
    advertisement
        .capabilities
        .as_ref()
        .map(|capabilities| {
            capabilities
                .max_sessions
                .saturating_sub(capabilities.active_sessions)
        })
        .unwrap_or_default()
}

fn capability_load(advertisement: &NodeAdvertisement) -> f64 {
    let Some(capabilities) = advertisement.capabilities.as_ref() else {
        return 0.5;
    };
    if capabilities.max_sessions == 0 {
        return f64::INFINITY;
    }

    let session_load =
        f64::from(capabilities.active_sessions) / f64::from(capabilities.max_sessions);
    session_load + f64::from(capabilities.queue_depth)
}

fn compare_optional_metric_desc(left: Option<f64>, right: Option<f64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => right.total_cmp(&left),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
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
            .min_by(compare_route_candidates)
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

    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    fn advertisement(peer_id: &str, model_id: &str, layers: LayerRange) -> NodeAdvertisement {
        NodeAdvertisement {
            protocol_version: 1,
            peer_id: peer_id.to_owned(),
            addresses: vec![format!("127.0.0.1:70{}", layers.start)],
            available_ram_bytes: None,
            available_vram_bytes: None,
            latency_hint_ms: Some(layers.start + 1),
            capabilities: None,
            hosted_shards: vec![ShardDescriptor {
                model_id: model_id.to_owned(),
                layers,
                runtime_kind: RuntimeKind::Demo,
                resident: false,
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
            capabilities: None,
            hosted_shards: vec![ShardDescriptor::for_manifest(manifest, layers)],
            model_shards: Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn capacity_advertisement(
        peer_id: &str,
        compute_backend: &str,
        available_memory_bytes: u64,
        max_sessions: u32,
        active_sessions: u32,
        queue_depth: u32,
        prefill_tokens_per_second: Option<f32>,
        decode_tokens_per_second: Option<f32>,
        latency_hint_ms: u32,
    ) -> NodeAdvertisement {
        NodeAdvertisement {
            protocol_version: 1,
            peer_id: peer_id.to_owned(),
            addresses: vec![format!("127.0.0.1:{}", 9000 + latency_hint_ms)],
            available_ram_bytes: None,
            available_vram_bytes: None,
            latency_hint_ms: Some(latency_hint_ms),
            capabilities: Some(NodeCapabilities {
                os: "test-os".to_owned(),
                arch: "test-arch".to_owned(),
                compute_backend: compute_backend.to_owned(),
                device_name: String::new(),
                machine_id: None,
                logical_cpu_cores: 8,
                total_ram_bytes: available_memory_bytes.saturating_add(2 * GIB),
                available_ram_bytes: available_memory_bytes,
                total_accelerator_memory_bytes: available_memory_bytes.saturating_add(2 * GIB),
                available_accelerator_memory_bytes: available_memory_bytes,
                unified_memory: compute_backend == "metal",
                max_sessions,
                active_sessions,
                measured_prefill_tokens_per_second: prefill_tokens_per_second,
                measured_decode_tokens_per_second: decode_tokens_per_second,
                queue_depth,
                llama_rpc: None,
            }),
            hosted_shards: Vec::new(),
            model_shards: Vec::new(),
        }
    }

    fn fixed_components(count: u32, weight_bytes: u64) -> Vec<FixedModelComponent> {
        (0..count)
            .map(|index| FixedModelComponent {
                content_hash: format!("sha256:{index:064x}"),
                layers: LayerRange::new(index, index + 1).unwrap(),
                weight_bytes,
            })
            .collect()
    }

    fn full_model_advertisement(
        peer_id: &str,
        manifest: &ModelManifest,
        latency_hint_ms: u32,
        capabilities: Option<NodeCapabilities>,
    ) -> NodeAdvertisement {
        NodeAdvertisement {
            protocol_version: 1,
            peer_id: peer_id.to_owned(),
            addresses: vec![format!("127.0.0.1:{}", 10_000 + latency_hint_ms)],
            available_ram_bytes: None,
            available_vram_bytes: None,
            latency_hint_ms: Some(latency_hint_ms),
            capabilities,
            hosted_shards: vec![ShardDescriptor::for_manifest(
                manifest,
                LayerRange::new(0, manifest.layer_count).unwrap(),
            )],
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

    #[test]
    fn plans_fixed_components_across_mixed_accelerators() {
        let components = fixed_components(20, GIB);
        let mut peer_a = capacity_advertisement(
            "peer-a",
            "cuda",
            22 * GIB,
            4,
            0,
            0,
            Some(160.0),
            Some(80.0),
            12,
        );
        peer_a
            .capabilities
            .as_mut()
            .unwrap()
            .total_accelerator_memory_bytes = 24 * GIB;
        let mut peer_b = capacity_advertisement(
            "peer-b",
            "cuda",
            7 * GIB,
            2,
            0,
            0,
            Some(80.0),
            Some(40.0),
            9,
        );
        peer_b
            .capabilities
            .as_mut()
            .unwrap()
            .total_accelerator_memory_bytes = 8 * GIB;
        let advertisements = vec![
            peer_a,
            peer_b,
            capacity_advertisement(
                "peer-c",
                "metal",
                14 * GIB,
                2,
                0,
                0,
                Some(60.0),
                Some(30.0),
                7,
            ),
            capacity_advertisement(
                "peer-d",
                "metal",
                10 * GIB,
                2,
                0,
                0,
                Some(40.0),
                Some(20.0),
                6,
            ),
        ];
        let config = CapacityPlanningConfig {
            kv_cache_bytes_per_layer: 64 * MIB,
            scratch_bytes_per_peer: 512 * MIB,
            safety_margin_bytes: 512 * MIB,
            safety_margin_basis_points: 500,
            minimum_peer_count: 4,
        };

        let plan = plan_fixed_components(&components, &advertisements, config).unwrap();

        assert_eq!(plan.assignments.len(), 4);
        validate_contiguous_coverage(20, plan.assignments.iter().map(|item| item.layers)).unwrap();
        assert!(plan.assignments.iter().all(|assignment| {
            assignment.total_reserved_bytes <= assignment.reported_available_memory_bytes
        }));

        let assigned_hashes = plan
            .assignments
            .iter()
            .flat_map(|assignment| assignment.component_hashes.iter().cloned())
            .collect::<Vec<_>>();
        let expected_hashes = components
            .iter()
            .map(|component| component.content_hash.clone())
            .collect::<Vec<_>>();
        assert_eq!(assigned_hashes, expected_hashes);

        let layers_for = |peer_id: &str| {
            plan.assignments
                .iter()
                .find(|assignment| assignment.peer_id == peer_id)
                .map(|assignment| assignment.layers.len())
                .unwrap()
        };
        assert!(layers_for("peer-a") > layers_for("peer-b"));
        assert!(layers_for("peer-b") >= layers_for("peer-d"));
    }

    #[test]
    fn uses_the_fewest_peers_allowed_by_policy() {
        let components = fixed_components(4, GIB);
        let advertisements = vec![
            capacity_advertisement(
                "fast-peer",
                "cuda",
                16 * GIB,
                2,
                0,
                0,
                Some(100.0),
                Some(50.0),
                20,
            ),
            capacity_advertisement(
                "near-peer",
                "metal",
                16 * GIB,
                2,
                0,
                0,
                Some(20.0),
                Some(10.0),
                1,
            ),
        ];

        let plan = plan_fixed_components(
            &components,
            &advertisements,
            CapacityPlanningConfig {
                safety_margin_basis_points: 0,
                ..CapacityPlanningConfig::default()
            },
        )
        .unwrap();

        assert_eq!(plan.assignments.len(), 1);
        assert_eq!(plan.assignments[0].peer_id, "fast-peer");
        assert_eq!(plan.assignments[0].component_hashes.len(), 4);
    }

    #[test]
    fn measured_load_reduces_a_peers_assigned_share() {
        let components = fixed_components(6, GIB);
        let advertisements = vec![
            capacity_advertisement(
                "idle-peer",
                "metal",
                16 * GIB,
                2,
                0,
                0,
                Some(60.0),
                Some(30.0),
                1,
            ),
            capacity_advertisement(
                "queued-peer",
                "cuda",
                16 * GIB,
                2,
                0,
                1,
                Some(60.0),
                Some(30.0),
                1,
            ),
        ];

        let plan = plan_fixed_components(
            &components,
            &advertisements,
            CapacityPlanningConfig {
                safety_margin_basis_points: 0,
                minimum_peer_count: 2,
                ..CapacityPlanningConfig::default()
            },
        )
        .unwrap();

        let layers_for = |peer_id: &str| {
            plan.assignments
                .iter()
                .find(|assignment| assignment.peer_id == peer_id)
                .map(|assignment| assignment.layers.len())
                .unwrap()
        };
        assert!(layers_for("idle-peer") > layers_for("queued-peer"));
    }

    #[test]
    fn includes_kv_scratch_and_safety_in_memory_reservation() {
        let components = fixed_components(1, 5 * GIB);
        let advertisements = vec![capacity_advertisement(
            "peer-a",
            "cuda",
            8 * GIB,
            1,
            0,
            0,
            None,
            None,
            1,
        )];
        let config = CapacityPlanningConfig {
            kv_cache_bytes_per_layer: GIB,
            scratch_bytes_per_peer: GIB,
            safety_margin_bytes: GIB,
            safety_margin_basis_points: 0,
            minimum_peer_count: 1,
        };

        let plan = plan_fixed_components(&components, &advertisements, config).unwrap();
        let assignment = &plan.assignments[0];

        assert_eq!(assignment.weight_bytes, 5 * GIB);
        assert_eq!(assignment.kv_cache_bytes, GIB);
        assert_eq!(assignment.scratch_bytes, GIB);
        assert_eq!(assignment.safety_bytes, GIB);
        assert_eq!(assignment.total_reserved_bytes, 8 * GIB);
    }

    #[test]
    fn never_splits_an_oversized_fixed_component() {
        let components = vec![
            FixedModelComponent {
                content_hash: "sha256:first".to_owned(),
                layers: LayerRange::new(0, 1).unwrap(),
                weight_bytes: 6 * GIB,
            },
            FixedModelComponent {
                content_hash: "sha256:second".to_owned(),
                layers: LayerRange::new(1, 2).unwrap(),
                weight_bytes: GIB,
            },
        ];
        let advertisements = vec![
            capacity_advertisement("peer-a", "cuda", 5 * GIB, 1, 0, 0, None, None, 1),
            capacity_advertisement("peer-b", "metal", 5 * GIB, 1, 0, 0, None, None, 1),
        ];

        assert_eq!(
            plan_fixed_components(
                &components,
                &advertisements,
                CapacityPlanningConfig {
                    safety_margin_basis_points: 0,
                    ..CapacityPlanningConfig::default()
                }
            ),
            Err(CapacityPlannerError::NoFeasiblePlacement)
        );
    }

    #[test]
    fn saturated_and_memoryless_peers_are_ineligible() {
        let components = fixed_components(1, GIB);
        let saturated =
            capacity_advertisement("saturated", "cuda", 8 * GIB, 1, 1, 0, None, None, 1);
        let mut memoryless =
            capacity_advertisement("memoryless", "metal", 8 * GIB, 1, 0, 0, None, None, 1);
        let capabilities = memoryless.capabilities.as_mut().unwrap();
        capabilities.available_accelerator_memory_bytes = 0;
        capabilities.available_ram_bytes = 0;

        assert_eq!(
            plan_fixed_components(
                &components,
                &[saturated, memoryless],
                CapacityPlanningConfig::default()
            ),
            Err(CapacityPlannerError::NoEligiblePeers)
        );
    }

    #[test]
    fn accepts_legacy_capacity_for_placement() {
        let components = fixed_components(2, GIB);
        let mut legacy = advertisement(
            "legacy-peer",
            &ModelManifest::demo().model_id,
            LayerRange::new(0, 3).unwrap(),
        );
        legacy.available_vram_bytes = Some(4 * GIB);

        let plan = plan_fixed_components(
            &components,
            &[legacy],
            CapacityPlanningConfig {
                safety_margin_basis_points: 0,
                ..CapacityPlanningConfig::default()
            },
        )
        .unwrap();

        assert_eq!(plan.assignments[0].peer_id, "legacy-peer");
    }

    #[test]
    fn rejects_invalid_fixed_component_manifests() {
        let peer = capacity_advertisement("peer-a", "cuda", 8 * GIB, 1, 0, 0, None, None, 1);

        assert_eq!(
            plan_fixed_components(
                &[],
                std::slice::from_ref(&peer),
                CapacityPlanningConfig::default()
            ),
            Err(CapacityPlannerError::EmptyComponents)
        );

        let missing_hash = vec![FixedModelComponent {
            content_hash: "  ".to_owned(),
            layers: LayerRange::new(0, 1).unwrap(),
            weight_bytes: GIB,
        }];
        assert_eq!(
            plan_fixed_components(
                &missing_hash,
                std::slice::from_ref(&peer),
                CapacityPlanningConfig::default()
            ),
            Err(CapacityPlannerError::MissingContentHash { index: 0 })
        );

        let non_contiguous = vec![
            FixedModelComponent {
                content_hash: "sha256:a".to_owned(),
                layers: LayerRange::new(0, 1).unwrap(),
                weight_bytes: GIB,
            },
            FixedModelComponent {
                content_hash: "sha256:b".to_owned(),
                layers: LayerRange::new(2, 3).unwrap(),
                weight_bytes: GIB,
            },
        ];
        assert!(matches!(
            plan_fixed_components(&non_contiguous, &[peer], CapacityPlanningConfig::default()),
            Err(CapacityPlannerError::InvalidCoverage(
                ModelError::NonContiguous { .. }
            ))
        ));
    }

    #[test]
    fn legacy_route_candidates_keep_latency_first_ordering() {
        let manifest = ModelManifest::demo();
        let slow_full = full_model_advertisement("slow-full", &manifest, 20, None);
        let fast_full = full_model_advertisement("fast-full", &manifest, 2, None);

        let route = build_greedy_route(&manifest, &[slow_full, fast_full]).unwrap();

        assert_eq!(route[0].peer_id, "fast-full");
    }

    #[test]
    fn legacy_route_candidates_keep_farthest_coverage_tie_break() {
        let manifest = ModelManifest::demo();
        let mut short = advertisement("short", &manifest.model_id, LayerRange::new(0, 3).unwrap());
        short.latency_hint_ms = Some(5);
        let mut full = advertisement(
            "full",
            &manifest.model_id,
            LayerRange::new(0, manifest.layer_count).unwrap(),
        );
        full.latency_hint_ms = Some(5);

        let route = build_greedy_route(&manifest, &[short, full]).unwrap();

        assert_eq!(route.len(), 1);
        assert_eq!(route[0].peer_id, "full");
    }

    #[test]
    fn capable_route_candidates_prefer_throughput_and_headroom_before_latency() {
        let manifest = ModelManifest::demo();
        let slower = capacity_advertisement(
            "slower",
            "metal",
            8 * GIB,
            4,
            0,
            0,
            Some(20.0),
            Some(10.0),
            1,
        )
        .capabilities;
        let faster = capacity_advertisement(
            "faster",
            "cuda",
            8 * GIB,
            4,
            0,
            0,
            Some(100.0),
            Some(50.0),
            20,
        )
        .capabilities;
        let advertisements = vec![
            full_model_advertisement("slower", &manifest, 1, slower),
            full_model_advertisement("faster", &manifest, 20, faster),
        ];

        let route = build_greedy_route(&manifest, &advertisements).unwrap();

        assert_eq!(route[0].peer_id, "faster");
    }

    #[test]
    fn saturated_capable_route_falls_behind_legacy_availability() {
        let manifest = ModelManifest::demo();
        let saturated = capacity_advertisement(
            "saturated",
            "cuda",
            8 * GIB,
            1,
            1,
            0,
            Some(100.0),
            Some(50.0),
            1,
        )
        .capabilities;
        let advertisements = vec![
            full_model_advertisement("saturated", &manifest, 1, saturated),
            full_model_advertisement("legacy", &manifest, 20, None),
        ];

        let route = build_greedy_route(&manifest, &advertisements).unwrap();

        assert_eq!(route[0].peer_id, "legacy");
    }
}

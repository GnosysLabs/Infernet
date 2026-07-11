use infernet_model::{
    INFERNET_CHAT_KV_CACHE_BYTES_PER_LAYER, LayerRange, ModelManifest, OfficialModelRelease,
};
use infernet_protocol::{
    MIN_DISTRIBUTED_MACHINE_COUNT, NodeAdvertisement, NodeCapabilities, RouteHop,
};
use infernet_router::{ShardRegistry, execution_advertisement_is_eligible};
use serde::Serialize;
use std::collections::BTreeSet;

const SAFETY_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;
const RUNTIME_SCRATCH_BYTES: u64 = 768 * 1024 * 1024;
const UNIFIED_MEMORY_MAX_FRACTION_DENOMINATOR: u64 = 2;
const MAX_PIPELINE_WORKERS: usize = 8;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionParticipantView {
    pub peer_id: String,
    pub short_peer_id: String,
    pub role: &'static str,
    pub compute_backend: String,
    pub device_name: String,
    pub available_memory_bytes: u64,
    pub estimated_share_percent: u32,
}

#[derive(Debug, Clone)]
pub struct WorkerExecutionPlan {
    pub route: Vec<RouteHop>,
    pub participants: Vec<ExecutionParticipantView>,
    origin_machine_id: Option<String>,
    peer_bindings: Vec<String>,
    machine_bindings: Vec<String>,
}

#[derive(Debug, Clone)]
struct Candidate {
    peer_id: String,
    machine_id: String,
    address: String,
    compute_backend: String,
    device_name: String,
    available_memory_bytes: u64,
    usable_memory_bytes: u64,
    layer_capacity: u32,
    assigned_layers: u32,
}

pub fn plan_worker_execution(
    registry: &ShardRegistry,
    model: &ModelManifest,
    origin_peer_id: &str,
) -> Result<WorkerExecutionPlan, String> {
    if model.layer_count == 0 {
        return Err("The selected model has no executable layers.".to_owned());
    }
    let bytes_per_layer = estimated_bytes_per_layer(model);

    let advertisements = registry.advertisements();
    let origin_machine_id = advertisements
        .iter()
        .find(|advertisement| advertisement.peer_id == origin_peer_id)
        .and_then(advertisement_machine_id)
        .map(str::to_owned);
    let mut candidates = advertisements
        .iter()
        .filter(|advertisement| hosts_verified_full_model(advertisement, model))
        .filter_map(|advertisement| candidate(advertisement, model, bytes_per_layer))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .usable_memory_bytes
            .cmp(&left.usable_memory_bytes)
            .then_with(|| left.peer_id.cmp(&right.peer_id))
    });

    let mut machines = BTreeSet::new();
    candidates.retain(|candidate| machines.insert(candidate.machine_id.clone()));
    let eligible_machine_count = candidates.len();
    let candidate_limit = MAX_PIPELINE_WORKERS.min(model.layer_count as usize);
    if eligible_machine_count >= MIN_DISTRIBUTED_MACHINE_COUNT
        && candidate_limit < MIN_DISTRIBUTED_MACHINE_COUNT
    {
        return Err(
            "The selected model cannot be divided across the available computers.".to_owned(),
        );
    }
    if candidates.len() > candidate_limit {
        let requester_index = origin_machine_id.as_ref().and_then(|origin_machine_id| {
            candidates
                .iter()
                .position(|candidate| candidate.machine_id == *origin_machine_id)
        });
        if let Some(requester_index) = requester_index.filter(|index| *index >= candidate_limit) {
            let requester = candidates.remove(requester_index);
            candidates.truncate(candidate_limit - 1);
            candidates.push(requester);
        } else {
            candidates.truncate(candidate_limit);
        }
    }
    let requester_only = eligible_machine_count == 1
        && candidates.len() == 1
        && origin_machine_id.as_ref() == Some(&candidates[0].machine_id);
    if candidates.len() < MIN_DISTRIBUTED_MACHINE_COUNT && !requester_only {
        return Err(
            "Infernet will not assign an entire request to one remote computer. Waiting for another computer to join."
                .to_owned(),
        );
    }
    let total_capacity = candidates
        .iter()
        .map(|candidate| candidate.layer_capacity)
        .sum::<u32>();
    if total_capacity < model.layer_count {
        return Err(format!(
            "The workers with the model on disk can safely hold {} of {} layers.",
            total_capacity, model.layer_count
        ));
    }

    // Every selected machine participates. One layer is assigned first, then
    // remaining layers are placed on the machine with the most memory per
    // assigned layer. This intentionally keeps the model split even when one
    // worker could hold all of it.
    for candidate in &mut candidates {
        candidate.assigned_layers = 1;
    }
    let mut remaining = model.layer_count.saturating_sub(candidates.len() as u32);
    while remaining > 0 {
        let Some(index) = candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.assigned_layers < candidate.layer_capacity)
            .max_by(|(_, left), (_, right)| {
                let left_score =
                    u128::from(left.usable_memory_bytes) * u128::from(right.assigned_layers + 1);
                let right_score =
                    u128::from(right.usable_memory_bytes) * u128::from(left.assigned_layers + 1);
                left_score
                    .cmp(&right_score)
                    .then_with(|| right.peer_id.cmp(&left.peer_id))
            })
            .map(|(index, _)| index)
        else {
            return Err("The worker layer allocator exhausted reported capacity.".to_owned());
        };
        candidates[index].assigned_layers += 1;
        remaining -= 1;
    }

    // The largest worker owns embeddings. Put the next-largest at the tail so
    // the output head and sampler also land on a high-capacity machine.
    if candidates.len() > 2 {
        let second = candidates.remove(1);
        candidates.push(second);
    }

    let mut layer_start = 0_u32;
    let mut route = Vec::with_capacity(candidates.len());
    let mut participants = Vec::with_capacity(candidates.len());
    let mut allocated_percent = 0_u32;
    for (index, candidate) in candidates.iter().enumerate() {
        let layer_end = layer_start + candidate.assigned_layers;
        route.push(RouteHop {
            peer_id: candidate.peer_id.clone(),
            machine_id: candidate.machine_id.clone(),
            address: candidate.address.clone(),
            layers: LayerRange {
                start: layer_start,
                end: layer_end,
            },
        });
        let role = if index == 0 {
            "entry worker"
        } else if index + 1 == candidates.len() {
            "sampling worker"
        } else {
            "worker"
        };
        let estimated_share_percent = if index + 1 == candidates.len() {
            100_u32.saturating_sub(allocated_percent)
        } else {
            let remaining_workers = (candidates.len() - index - 1) as u32;
            let share = (candidate.assigned_layers * 100 / model.layer_count)
                .max(1)
                .min(100_u32.saturating_sub(allocated_percent + remaining_workers));
            allocated_percent += share;
            share
        };
        participants.push(ExecutionParticipantView {
            peer_id: candidate.peer_id.clone(),
            short_peer_id: short_peer_id(&candidate.peer_id),
            role,
            compute_backend: candidate.compute_backend.clone(),
            device_name: candidate.device_name.clone(),
            available_memory_bytes: candidate.available_memory_bytes,
            estimated_share_percent,
        });
        layer_start = layer_end;
    }

    Ok(WorkerExecutionPlan {
        origin_machine_id,
        peer_bindings: route.iter().map(|hop| hop.peer_id.clone()).collect(),
        machine_bindings: candidates
            .iter()
            .map(|candidate| candidate.machine_id.clone())
            .collect(),
        route,
        participants,
    })
}

impl WorkerExecutionPlan {
    pub fn remains_usable(&self, registry: &ShardRegistry, model: &ModelManifest) -> bool {
        let advertisements = registry.advertisements();
        let bytes_per_layer = estimated_bytes_per_layer(model);
        let eligible_machine_ids = advertisements
            .iter()
            .filter(|advertisement| hosts_verified_full_model(advertisement, model))
            .filter_map(|advertisement| candidate(advertisement, model, bytes_per_layer))
            .map(|candidate| candidate.machine_id)
            .collect::<BTreeSet<_>>();
        let machine_count = self.machine_bindings.iter().collect::<BTreeSet<_>>().len();
        let requester_only =
            machine_count == 1 && self.machine_bindings.first() == self.origin_machine_id.as_ref();
        if eligible_machine_ids.len() >= MIN_DISTRIBUTED_MACHINE_COUNT {
            if machine_count < MIN_DISTRIBUTED_MACHINE_COUNT {
                return false;
            }
            if self.origin_machine_id.as_ref().is_some_and(|origin| {
                eligible_machine_ids.contains(origin) && !self.machine_bindings.contains(origin)
            }) {
                return false;
            }
        } else if eligible_machine_ids.len() != 1
            || !requester_only
            || self
                .origin_machine_id
                .as_ref()
                .is_none_or(|origin| !eligible_machine_ids.contains(origin))
        {
            return false;
        }

        if self.route.len() != self.peer_bindings.len()
            || self.route.len() != self.machine_bindings.len()
            || !self
                .machine_bindings
                .iter()
                .all(|machine_id| eligible_machine_ids.contains(machine_id))
        {
            return false;
        }

        self.route
            .iter()
            .zip(&self.peer_bindings)
            .zip(&self.machine_bindings)
            .all(|((hop, peer_id), machine_id)| {
                advertisements.iter().any(|advertisement| {
                    advertisement.peer_id == *peer_id
                        && hop.peer_id == *peer_id
                        && hop.machine_id == *machine_id
                        && advertisement
                            .capabilities
                            .as_ref()
                            .and_then(|capabilities| capabilities.machine_id.as_ref())
                            == Some(machine_id)
                        && hosts_verified_full_model(advertisement, model)
                        && candidate(advertisement, model, bytes_per_layer).is_some_and(
                            |candidate| {
                                candidate.machine_id == *machine_id
                                    && candidate.layer_capacity >= hop.layers.len()
                            },
                        )
                })
            })
    }
}

fn estimated_bytes_per_layer(model: &ModelManifest) -> u64 {
    OfficialModelRelease::infernet_chat_v1_compatibility()
        .expected_total_bytes
        .div_ceil(u64::from(model.layer_count))
        .saturating_add(INFERNET_CHAT_KV_CACHE_BYTES_PER_LAYER)
}

fn advertisement_machine_id(advertisement: &NodeAdvertisement) -> Option<&str> {
    advertisement
        .capabilities
        .as_ref()?
        .machine_id
        .as_deref()
        .map(str::trim)
        .filter(|machine_id| !machine_id.is_empty())
}

fn candidate(
    advertisement: &NodeAdvertisement,
    model: &ModelManifest,
    bytes_per_layer: u64,
) -> Option<Candidate> {
    let capabilities = advertisement.capabilities.as_ref()?;
    if !execution_advertisement_is_eligible(model, advertisement)
        || !worker_is_available(capabilities)
    {
        return None;
    }
    let machine_id = advertisement_machine_id(advertisement)?.to_owned();
    let address = preferred_address(advertisement)?;
    // Enforce the unified-memory ceiling again at planning time so an older
    // peer cannot overcommit a host by advertising all reclaimable RAM.
    let available_memory_bytes = safe_advertised_accelerator_memory(capabilities);
    let usable_memory_bytes = safe_model_memory(available_memory_bytes);
    let resident_layer_capacity = advertisement
        .hosted_shards
        .iter()
        .filter(|shard| {
            shard.model_id == model.model_id
                && shard.runtime_kind == model.runtime_kind
                && shard.resident
        })
        .map(|shard| shard.layers.len())
        .max()
        .unwrap_or(0);
    let contribution_layer_capacity = capabilities
        .vram_contribution_limit_bytes
        .map(|limit| {
            safe_model_memory(limit)
                .checked_div(bytes_per_layer)
                .unwrap_or(0) as u32
        })
        .unwrap_or(u32::MAX);
    let layer_capacity = (usable_memory_bytes.checked_div(bytes_per_layer)? as u32)
        .max(resident_layer_capacity)
        .min(contribution_layer_capacity);
    if layer_capacity == 0 {
        return None;
    }
    Some(Candidate {
        peer_id: advertisement.peer_id.clone(),
        machine_id,
        address,
        compute_backend: capabilities.compute_backend.clone(),
        device_name: capabilities.device_name.clone(),
        available_memory_bytes,
        usable_memory_bytes,
        layer_capacity,
        assigned_layers: 0,
    })
}

fn worker_is_available(capabilities: &NodeCapabilities) -> bool {
    matches!(capabilities.compute_backend.as_str(), "cuda" | "metal")
        && capabilities.active_sessions < capabilities.max_sessions
        && capabilities.available_accelerator_memory_bytes > 0
}

pub fn worker_is_usable(capabilities: &NodeCapabilities) -> bool {
    worker_is_available(capabilities)
}

fn hosts_verified_full_model(advertisement: &NodeAdvertisement, model: &ModelManifest) -> bool {
    advertisement.hosted_shards.iter().any(|shard| {
        shard.model_id == model.model_id
            && shard.runtime_kind == model.runtime_kind
            && shard.layers.start == 0
            && shard.layers.end == model.layer_count
            && shard
                .shard_hash
                .as_ref()
                .is_some_and(|hash| !hash.is_empty())
    })
}

fn preferred_address(advertisement: &NodeAdvertisement) -> Option<String> {
    advertisement
        .addresses
        .iter()
        .find(|address| address.contains("/p2p-circuit"))
        .or_else(|| {
            advertisement.addresses.iter().find(|address| {
                !address.contains("/ip4/127.0.0.1/") && !address.contains("/ip6/::1/")
            })
        })
        .or_else(|| advertisement.addresses.first())
        .cloned()
}

fn safe_model_memory(available_memory_bytes: u64) -> u64 {
    let safety = SAFETY_RESERVE_BYTES.max(available_memory_bytes / 10);
    available_memory_bytes
        .saturating_sub(RUNTIME_SCRATCH_BYTES)
        .saturating_sub(safety)
}

fn safe_advertised_accelerator_memory(capabilities: &NodeCapabilities) -> u64 {
    if capabilities.unified_memory {
        capabilities
            .available_accelerator_memory_bytes
            .min(capabilities.total_ram_bytes / UNIFIED_MEMORY_MAX_FRACTION_DENOMINATOR)
    } else {
        capabilities.available_accelerator_memory_bytes
    }
}

fn short_peer_id(peer_id: &str) -> String {
    if peer_id.len() <= 16 {
        return peer_id.to_owned();
    }
    format!("{}...{}", &peer_id[..8], &peer_id[peer_id.len() - 6..])
}

#[cfg(test)]
mod tests {
    use infernet_model::{RuntimeKind, ShardDescriptor};
    use infernet_protocol::{NodeAdvertisement, NodeCapabilities, PROTOCOL_VERSION};

    use super::*;

    #[test]
    fn unified_memory_never_advertises_more_than_half_of_host_ram() {
        let model = ModelManifest::infernet_chat_v1();
        let mut peer = advertisement("worker-mac", 16, &model);
        let capabilities = peer.capabilities.as_mut().unwrap();
        capabilities.unified_memory = true;
        capabilities.total_ram_bytes = 16 * 1024 * 1024 * 1024;
        capabilities.available_accelerator_memory_bytes = 14 * 1024 * 1024 * 1024;

        assert_eq!(
            safe_advertised_accelerator_memory(capabilities),
            8 * 1024 * 1024 * 1024,
        );
    }

    #[test]
    fn splits_across_every_ready_worker_even_when_one_can_host_the_model() {
        let model = ModelManifest::infernet_chat_v1();
        let mut registry = ShardRegistry::new();
        registry.upsert(advertisement("worker-3090", 24, &model));
        registry.upsert(advertisement("worker-4060", 8, &model));
        registry.upsert(advertisement("worker-mac", 16, &model));

        let plan = plan_worker_execution(&registry, &model, "worker-3090").unwrap();
        assert_eq!(plan.route.len(), 3);
        assert_eq!(plan.route.first().unwrap().layers.start, 0);
        assert_eq!(plan.route.last().unwrap().layers.end, model.layer_count);
        for pair in plan.route.windows(2) {
            assert_eq!(pair[0].layers.end, pair[1].layers.start);
        }
        assert!(
            plan.participants
                .iter()
                .all(|participant| participant.estimated_share_percent > 0)
        );
    }

    #[test]
    fn ignores_gpu_peers_without_the_verified_model() {
        let model = ModelManifest::infernet_chat_v1();
        let mut registry = ShardRegistry::new();
        let mut peer = advertisement("worker", 24, &model);
        peer.hosted_shards.clear();
        registry.upsert(peer);
        assert!(plan_worker_execution(&registry, &model, "worker").is_err());
    }

    #[test]
    fn allows_the_requesters_own_machine_when_it_is_the_only_candidate() {
        let model = ModelManifest::infernet_chat_v1();
        let mut registry = ShardRegistry::new();
        let mut peer = advertisement("resident-worker", 1, &model);
        peer.hosted_shards[0].resident = true;
        registry.upsert(peer);

        let plan = plan_worker_execution(&registry, &model, "resident-worker").unwrap();
        assert_eq!(plan.route.len(), 1);
        assert_eq!(plan.route[0].peer_id, "resident-worker");
    }

    #[test]
    fn requester_machine_identity_allows_a_separate_local_worker_process() {
        let model = ModelManifest::infernet_chat_v1();
        let mut registry = ShardRegistry::new();
        let mut requester = advertisement("requester-app", 1, &model);
        requester.hosted_shards.clear();
        requester.capabilities.as_mut().unwrap().machine_id = Some("local-machine".to_owned());
        let mut worker = advertisement("local-worker", 24, &model);
        worker.capabilities.as_mut().unwrap().machine_id = Some("local-machine".to_owned());
        registry.upsert(requester);
        registry.upsert(worker);

        let plan = plan_worker_execution(&registry, &model, "requester-app").unwrap();
        assert_eq!(plan.route.len(), 1);
        assert_eq!(plan.route[0].peer_id, "local-worker");
    }

    #[test]
    fn local_only_lease_expires_when_another_machine_becomes_eligible() {
        let model = ModelManifest::infernet_chat_v1();
        let mut registry = ShardRegistry::new();
        registry.upsert(advertisement("requester", 24, &model));

        let plan = plan_worker_execution(&registry, &model, "requester").unwrap();
        assert!(plan.remains_usable(&registry, &model));

        registry.upsert(advertisement("remote-worker", 8, &model));
        assert!(!plan.remains_usable(&registry, &model));

        let replacement = plan_worker_execution(&registry, &model, "requester").unwrap();
        assert_eq!(replacement.route.len(), 2);
    }

    #[test]
    fn lease_expires_when_a_worker_can_no_longer_hold_its_assigned_layers() {
        let model = ModelManifest::infernet_chat_v1();
        let mut registry = ShardRegistry::new();
        registry.upsert(advertisement("requester", 24, &model));
        registry.upsert(advertisement("remote-worker", 24, &model));

        let plan = plan_worker_execution(&registry, &model, "requester").unwrap();
        let remote_layers = plan
            .route
            .iter()
            .find(|hop| hop.peer_id == "remote-worker")
            .expect("remote worker participates")
            .layers
            .len();
        assert!(remote_layers > 1);
        assert!(plan.remains_usable(&registry, &model));

        let mut constrained = advertisement("remote-worker", 24, &model);
        constrained
            .capabilities
            .as_mut()
            .unwrap()
            .vram_contribution_limit_bytes = Some(3 * 1024 * 1024 * 1024);
        registry.upsert(constrained);

        assert!(!plan.remains_usable(&registry, &model));
    }

    #[test]
    fn rejects_one_remote_machine_even_when_it_can_hold_every_layer() {
        let model = ModelManifest::infernet_chat_v1();
        let mut registry = ShardRegistry::new();
        registry.upsert(advertisement("remote-worker", 24, &model));

        assert!(plan_worker_execution(&registry, &model, "requester").is_err());
    }

    #[test]
    fn rejects_two_peer_ids_on_the_same_physical_machine() {
        let model = ModelManifest::infernet_chat_v1();
        let mut registry = ShardRegistry::new();
        let mut first = advertisement("worker-a", 24, &model);
        let mut second = advertisement("worker-b", 24, &model);
        first.capabilities.as_mut().unwrap().machine_id = Some("same-machine".to_owned());
        second.capabilities.as_mut().unwrap().machine_id = Some("same-machine".to_owned());
        registry.upsert(first);
        registry.upsert(second);

        assert!(plan_worker_execution(&registry, &model, "requester").is_err());
    }

    #[test]
    fn contribution_limit_caps_resident_model_layers() {
        let model = ModelManifest::infernet_chat_v1();
        let mut registry = ShardRegistry::new();
        let mut peer = advertisement("limited-resident-worker", 24, &model);
        peer.hosted_shards[0].resident = true;
        peer.capabilities
            .as_mut()
            .unwrap()
            .vram_contribution_limit_bytes = Some(4 * 1024 * 1024 * 1024);
        registry.upsert(peer);

        assert!(plan_worker_execution(&registry, &model, "limited-resident-worker").is_err());
    }

    fn advertisement(peer_id: &str, memory_gib: u64, model: &ModelManifest) -> NodeAdvertisement {
        let memory = memory_gib * 1024 * 1024 * 1024;
        NodeAdvertisement {
            protocol_version: PROTOCOL_VERSION,
            peer_id: peer_id.to_owned(),
            addresses: vec![format!("/ip4/192.168.1.10/tcp/9777/p2p/{peer_id}")],
            available_ram_bytes: Some(memory),
            available_vram_bytes: Some(memory),
            latency_hint_ms: Some(1),
            capabilities: Some(NodeCapabilities {
                os: "linux".to_owned(),
                arch: "x86_64".to_owned(),
                compute_backend: "cuda".to_owned(),
                device_name: "GPU".to_owned(),
                machine_id: Some(format!("machine-{peer_id}")),
                logical_cpu_cores: 16,
                total_ram_bytes: memory,
                available_ram_bytes: memory,
                total_accelerator_memory_bytes: memory,
                available_accelerator_memory_bytes: memory,
                vram_contribution_limit_bytes: None,
                unified_memory: false,
                max_sessions: 1,
                active_sessions: 0,
                measured_prefill_tokens_per_second: None,
                measured_decode_tokens_per_second: None,
                queue_depth: 0,
                llama_rpc: None,
                image_rpc: None,
            }),
            hosted_shards: vec![ShardDescriptor {
                model_id: model.model_id.clone(),
                layers: LayerRange {
                    start: 0,
                    end: model.layer_count,
                },
                runtime_kind: RuntimeKind::LlamaCpp,
                resident: false,
                tokenizer: None,
                metadata: None,
                shard_hash: Some("verified".to_owned()),
                seed_manifest: None,
            }],
            model_shards: Vec::new(),
            model_components: Vec::new(),
            coarse_location: None,
        }
    }
}

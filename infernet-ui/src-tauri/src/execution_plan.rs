use infernet_model::{ModelManifest, OfficialModelRelease};
use infernet_node::{INFERNET_LLAMA_RPC_RUNTIME_ABI, LLAMA_RPC_PROTOCOL_VERSION};
use infernet_protocol::{LLAMA_RPC_TUNNEL_PROTOCOL, NodeAdvertisement, NodeCapabilities, RouteHop};
use infernet_router::ShardRegistry;
use serde::Serialize;
use std::collections::BTreeSet;

const SAFETY_RESERVE_BYTES: u64 = 1024 * 1024 * 1024;
const RUNTIME_SCRATCH_BYTES: u64 = 768 * 1024 * 1024;
const KV_CACHE_BYTES_PER_LAYER: u64 = 32 * 1024 * 1024;
const MINIMUM_WORKER_CAPACITY_BYTES: u64 = 512 * 1024 * 1024;

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
pub struct RpcExecutionPlan {
    pub coordinator_peer_id: String,
    pub worker_peer_ids: Vec<String>,
    pub participants: Vec<ExecutionParticipantView>,
    worker_bindings: Vec<String>,
}

#[derive(Debug, Clone)]
struct Candidate {
    peer_id: String,
    machine_id: Option<String>,
    tunnel_ready: bool,
    compute_backend: String,
    device_name: String,
    available_memory_bytes: u64,
    usable_memory_bytes: u64,
    coordinator: bool,
}

pub fn plan_rpc_execution(
    registry: &ShardRegistry,
    route: &[RouteHop],
    model: &ModelManifest,
) -> Result<RpcExecutionPlan, String> {
    let coordinator_hop = route
        .first()
        .ok_or_else(|| "Infernet Chat is not available on a coordinator yet.".to_owned())?;
    if route.len() != 1
        || coordinator_hop.layers.start != 0
        || coordinator_hop.layers.end != model.layer_count
    {
        return Err(
            "Infernet Chat requires one verified full-model coordinator before it can distribute execution."
                .to_owned(),
        );
    }

    let advertisements = registry.advertisements();
    let coordinator_advertisement = advertisements
        .iter()
        .find(|advertisement| advertisement.peer_id == coordinator_hop.peer_id)
        .ok_or_else(|| "The selected model coordinator stopped advertising capacity.".to_owned())?;
    let coordinator = candidate_for_coordinator(coordinator_advertisement)?;
    let coordinator_machine_id = coordinator_advertisement
        .capabilities
        .as_ref()
        .and_then(|capabilities| capabilities.machine_id.as_deref());

    let mut workers = advertisements
        .iter()
        .filter(|advertisement| advertisement.peer_id != coordinator.peer_id)
        .filter_map(|advertisement| candidate_for_worker(advertisement, coordinator_machine_id))
        .collect::<Vec<_>>();
    workers.sort_by(|left, right| {
        right
            .usable_memory_bytes
            .cmp(&left.usable_memory_bytes)
            .then_with(|| left.peer_id.cmp(&right.peer_id))
    });

    let mut seen_machines = BTreeSet::new();
    workers.retain(|candidate| seen_machines.insert(candidate.physical_key()));
    if workers.is_empty() {
        return Err(
            "Distributed inference needs at least one other GPU or Apple-silicon machine with its compute service ready."
                .to_owned(),
        );
    }

    let mut candidates = Vec::with_capacity(workers.len() + 1);
    candidates.push(coordinator);
    candidates.extend(workers);
    let total_usable_memory = candidates
        .iter()
        .map(|candidate| candidate.usable_memory_bytes)
        .sum::<u64>();
    if total_usable_memory == 0 {
        return Err("The available machines did not report usable model memory.".to_owned());
    }
    let release = OfficialModelRelease::infernet_chat_v1_compatibility();
    let required_memory = release
        .expected_total_bytes
        .saturating_add(KV_CACHE_BYTES_PER_LAYER.saturating_mul(u64::from(model.layer_count)));
    if total_usable_memory < required_memory {
        return Err(format!(
            "The ready machines have {} GB of safe model memory, but Infernet Chat needs at least {} GB.",
            total_usable_memory / 1_000_000_000,
            required_memory.div_ceil(1_000_000_000),
        ));
    }

    let mut allocated_percent = 0_u32;
    let last_index = candidates.len() - 1;
    let participants = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let estimated_share_percent = if index == last_index {
                100_u32.saturating_sub(allocated_percent)
            } else {
                let proportional = ((candidate.usable_memory_bytes as u128 * 100)
                    / total_usable_memory as u128) as u32;
                let remaining_participants = (last_index - index) as u32;
                let maximum = 100_u32
                    .saturating_sub(allocated_percent)
                    .saturating_sub(remaining_participants);
                let share = proportional.max(1).min(maximum.max(1));
                allocated_percent = allocated_percent.saturating_add(share);
                share
            };
            ExecutionParticipantView {
                peer_id: candidate.peer_id.clone(),
                short_peer_id: short_peer_id(&candidate.peer_id),
                role: if candidate.coordinator {
                    "coordinator"
                } else {
                    "worker"
                },
                compute_backend: candidate.compute_backend.clone(),
                device_name: candidate.device_name.clone(),
                available_memory_bytes: candidate.available_memory_bytes,
                estimated_share_percent,
            }
        })
        .collect();
    let worker_peer_ids = candidates
        .iter()
        .filter(|candidate| !candidate.coordinator && candidate.tunnel_ready)
        .map(|candidate| candidate.peer_id.clone())
        .collect();

    let coordinator_peer_id = coordinator_hop.peer_id.clone();
    let worker_bindings = candidates
        .iter()
        .filter(|candidate| !candidate.coordinator && candidate.tunnel_ready)
        .map(|candidate| candidate.peer_id.clone())
        .collect();

    Ok(RpcExecutionPlan {
        coordinator_peer_id,
        worker_peer_ids,
        participants,
        worker_bindings,
    })
}

impl RpcExecutionPlan {
    pub fn remains_usable(&self, registry: &ShardRegistry, route: &[RouteHop]) -> bool {
        if route.len() != 1 || route[0].peer_id != self.coordinator_peer_id {
            return false;
        }
        let advertisements = registry.advertisements();
        let coordinator_online = advertisements
            .iter()
            .any(|advertisement| advertisement.peer_id == self.coordinator_peer_id);
        coordinator_online
            && self.worker_bindings.iter().all(|peer_id| {
                advertisements.iter().any(|advertisement| {
                    advertisement.peer_id == *peer_id
                        && advertisement
                            .capabilities
                            .as_ref()
                            .is_some_and(rpc_endpoint_is_usable)
                })
            })
    }
}

fn candidate_for_coordinator(advertisement: &NodeAdvertisement) -> Result<Candidate, String> {
    let capabilities = advertisement.capabilities.as_ref().ok_or_else(|| {
        "The selected model coordinator has not reported its hardware capacity.".to_owned()
    })?;
    if !matches!(capabilities.compute_backend.as_str(), "cuda" | "metal") {
        return Err("The model coordinator needs a CUDA GPU or Apple silicon.".to_owned());
    }
    if capabilities.active_sessions >= capabilities.max_sessions {
        return Err("The model coordinator is already being used by another request.".to_owned());
    }
    let available_memory_bytes = capabilities.available_accelerator_memory_bytes;
    if !capabilities.unified_memory && capabilities.available_ram_bytes < 2 * SAFETY_RESERVE_BYTES {
        return Err(
            "The model coordinator does not currently have enough free system memory.".to_owned(),
        );
    }
    let usable_memory_bytes = safe_model_memory(available_memory_bytes);
    if usable_memory_bytes < MINIMUM_WORKER_CAPACITY_BYTES {
        return Err("The model coordinator does not currently have enough free memory.".to_owned());
    }
    Ok(Candidate {
        peer_id: advertisement.peer_id.clone(),
        machine_id: capabilities.machine_id.clone(),
        tunnel_ready: false,
        compute_backend: capabilities.compute_backend.clone(),
        device_name: capabilities.device_name.clone(),
        available_memory_bytes,
        usable_memory_bytes,
        coordinator: true,
    })
}

fn candidate_for_worker(
    advertisement: &NodeAdvertisement,
    coordinator_machine_id: Option<&str>,
) -> Option<Candidate> {
    let capabilities = advertisement.capabilities.as_ref()?;
    if !matches!(capabilities.compute_backend.as_str(), "cuda" | "metal")
        || capabilities.active_sessions >= capabilities.max_sessions
    {
        return None;
    }
    if !rpc_endpoint_is_usable(capabilities)
        || coordinator_machine_id
            .zip(capabilities.machine_id.as_deref())
            .is_some_and(|(coordinator, worker)| coordinator == worker)
    {
        return None;
    }
    let available_memory_bytes = capabilities.available_accelerator_memory_bytes;
    let usable_memory_bytes = safe_model_memory(available_memory_bytes);
    if usable_memory_bytes < MINIMUM_WORKER_CAPACITY_BYTES {
        return None;
    }
    Some(Candidate {
        peer_id: advertisement.peer_id.clone(),
        machine_id: capabilities.machine_id.clone(),
        tunnel_ready: true,
        compute_backend: capabilities.compute_backend.clone(),
        device_name: capabilities.device_name.clone(),
        available_memory_bytes,
        usable_memory_bytes,
        coordinator: false,
    })
}

impl Candidate {
    fn physical_key(&self) -> String {
        self.machine_id
            .clone()
            .unwrap_or_else(|| format!("peer:{}", self.peer_id))
    }
}

fn safe_model_memory(available_memory_bytes: u64) -> u64 {
    let safety = SAFETY_RESERVE_BYTES.max(available_memory_bytes / 10);
    available_memory_bytes
        .saturating_sub(RUNTIME_SCRATCH_BYTES)
        .saturating_sub(safety)
}

pub fn rpc_endpoint_is_usable(capabilities: &NodeCapabilities) -> bool {
    capabilities.llama_rpc.as_ref().is_some_and(|endpoint| {
        endpoint.ready
            && endpoint.rpc_protocol_version == LLAMA_RPC_PROTOCOL_VERSION
            && endpoint.runtime_abi == INFERNET_LLAMA_RPC_RUNTIME_ABI
            && endpoint.backend == capabilities.compute_backend
            && endpoint.tunnel_protocol.as_deref() == Some(LLAMA_RPC_TUNNEL_PROTOCOL)
    })
}

fn short_peer_id(peer_id: &str) -> String {
    if peer_id.len() <= 16 {
        return peer_id.to_owned();
    }
    format!("{}...{}", &peer_id[..8], &peer_id[peer_id.len() - 6..])
}

#[cfg(test)]
mod tests {
    use infernet_model::LayerRange;
    use infernet_protocol::{LlamaRpcEndpoint, NodeCapabilities, PROTOCOL_VERSION};

    use super::*;

    #[test]
    fn requires_and_splits_across_a_distinct_ready_worker() {
        let mut registry = ShardRegistry::new();
        registry.upsert(advertisement("coordinator-peer", "192.168.1.10", 24));
        registry.upsert(advertisement("worker-peer", "192.168.1.11", 8));
        let model = ModelManifest::infernet_chat_v1();
        let route = vec![RouteHop {
            peer_id: "coordinator-peer".to_owned(),
            address: "/ip4/192.168.1.10/tcp/9777".to_owned(),
            layers: LayerRange {
                start: 0,
                end: model.layer_count,
            },
        }];

        let plan = plan_rpc_execution(&registry, &route, &model).unwrap();

        assert_eq!(plan.worker_peer_ids, vec!["worker-peer"]);
        assert_eq!(plan.participants.len(), 2);
        assert_eq!(plan.participants[0].role, "coordinator");
        assert_eq!(plan.participants[1].role, "worker");
        assert_eq!(
            plan.participants
                .iter()
                .map(|participant| participant.estimated_share_percent)
                .sum::<u32>(),
            100
        );
    }

    #[test]
    fn raw_rpc_address_is_not_used_for_tunnel_selection() {
        let mut registry = ShardRegistry::new();
        registry.upsert(advertisement("coordinator-peer", "192.168.1.10", 24));
        registry.upsert(advertisement("worker-peer", "203.0.113.10", 8));
        let model = ModelManifest::infernet_chat_v1();
        let route = vec![RouteHop {
            peer_id: "coordinator-peer".to_owned(),
            address: "/ip4/192.168.1.10/tcp/9777".to_owned(),
            layers: LayerRange {
                start: 0,
                end: model.layer_count,
            },
        }];

        let plan = plan_rpc_execution(&registry, &route, &model).unwrap();
        assert_eq!(plan.worker_peer_ids, vec!["worker-peer"]);
    }

    #[test]
    fn two_identities_on_one_physical_machine_do_not_count_twice() {
        let mut registry = ShardRegistry::new();
        let coordinator = advertisement("coordinator-peer", "192.168.1.10", 24);
        let mut duplicate = advertisement("other-app-peer", "100.64.1.10", 8);
        duplicate.capabilities.as_mut().unwrap().machine_id = coordinator
            .capabilities
            .as_ref()
            .unwrap()
            .machine_id
            .clone();
        registry.upsert(coordinator);
        registry.upsert(duplicate);
        let model = ModelManifest::infernet_chat_v1();
        let route = vec![RouteHop {
            peer_id: "coordinator-peer".to_owned(),
            address: "/ip4/192.168.1.10/tcp/9777".to_owned(),
            layers: LayerRange {
                start: 0,
                end: model.layer_count,
            },
        }];

        assert!(plan_rpc_execution(&registry, &route, &model).is_err());
    }

    fn advertisement(peer_id: &str, host: &str, memory_gib: u64) -> NodeAdvertisement {
        let memory = memory_gib * 1024 * 1024 * 1024;
        NodeAdvertisement {
            protocol_version: PROTOCOL_VERSION,
            peer_id: peer_id.to_owned(),
            addresses: Vec::new(),
            available_ram_bytes: Some(memory),
            available_vram_bytes: Some(memory),
            latency_hint_ms: Some(1),
            capabilities: Some(NodeCapabilities {
                os: "linux".to_owned(),
                arch: "x86_64".to_owned(),
                compute_backend: "cuda".to_owned(),
                device_name: "NVIDIA GPU".to_owned(),
                machine_id: Some(format!("machine-{peer_id}")),
                logical_cpu_cores: 16,
                total_ram_bytes: memory,
                available_ram_bytes: memory,
                total_accelerator_memory_bytes: memory,
                available_accelerator_memory_bytes: memory,
                unified_memory: false,
                max_sessions: 1,
                active_sessions: 0,
                measured_prefill_tokens_per_second: None,
                measured_decode_tokens_per_second: None,
                queue_depth: 0,
                llama_rpc: Some(LlamaRpcEndpoint {
                    host: host.to_owned(),
                    port: 50052,
                    rpc_protocol_version: LLAMA_RPC_PROTOCOL_VERSION.to_owned(),
                    runtime_abi: INFERNET_LLAMA_RPC_RUNTIME_ABI.to_owned(),
                    backend: "cuda".to_owned(),
                    ready: true,
                    tunnel_protocol: Some(infernet_protocol::LLAMA_RPC_TUNNEL_PROTOCOL.to_owned()),
                }),
            }),
            hosted_shards: Vec::new(),
            model_shards: Vec::new(),
        }
    }
}

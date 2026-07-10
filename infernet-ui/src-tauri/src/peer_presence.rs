use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use infernet_protocol::NodeAdvertisement;
use serde::Serialize;

/// A normal Activity-sidebar refresh starts discovery every six seconds. Keep
/// a machine online through several missed gossip windows so a transient
/// libp2p path closure does not make the UI (or planner) flap.
const CONNECTED_GRACE: Duration = Duration::from_secs(30);
/// After the connected grace expires, retain the machine as reconnecting so
/// the UI explains what is happening instead of removing and re-adding it.
const RECONNECTING_GRACE: Duration = Duration::from_secs(60);
/// Remember an unreachable machine long enough to make a real outage visible.
/// It is never returned as a routable advertisement in this state.
const UNREACHABLE_RETENTION: Duration = Duration::from_secs(5 * 60);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum ConnectionStatus {
    Connected,
    Reconnecting,
    Unreachable,
}

impl ConnectionStatus {
    pub(crate) fn is_connected(self) -> bool {
        self == Self::Connected
    }

    pub(crate) fn priority(self) -> u8 {
        match self {
            Self::Connected => 2,
            Self::Reconnecting => 1,
            Self::Unreachable => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PresenceRecord {
    pub(crate) advertisement: NodeAdvertisement,
    pub(crate) status: ConnectionStatus,
    pub(crate) last_seen_age: Duration,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PresenceSnapshot {
    records: Vec<PresenceRecord>,
}

impl PresenceSnapshot {
    pub(crate) fn records(&self) -> &[PresenceRecord] {
        &self.records
    }

    pub(crate) fn routable_advertisements(&self) -> impl Iterator<Item = NodeAdvertisement> + '_ {
        self.records
            .iter()
            .filter(|record| record.status.is_connected())
            .map(|record| record.advertisement.clone())
    }
}

#[derive(Debug, Clone)]
struct PresenceEntry {
    advertisement: NodeAdvertisement,
    last_seen: Instant,
}

#[derive(Debug)]
pub(crate) struct PeerPresence {
    entries: BTreeMap<String, PresenceEntry>,
    connected_grace: Duration,
    reconnecting_grace: Duration,
    retention: Duration,
}

impl Default for PeerPresence {
    fn default() -> Self {
        Self::with_thresholds(CONNECTED_GRACE, RECONNECTING_GRACE, UNREACHABLE_RETENTION)
    }
}

impl PeerPresence {
    fn with_thresholds(
        connected_grace: Duration,
        reconnecting_grace: Duration,
        retention: Duration,
    ) -> Self {
        debug_assert!(connected_grace <= reconnecting_grace);
        debug_assert!(reconnecting_grace <= retention);
        Self {
            entries: BTreeMap::new(),
            connected_grace,
            reconnecting_grace,
            retention,
        }
    }

    pub(crate) fn observe(
        &mut self,
        advertisements: impl IntoIterator<Item = NodeAdvertisement>,
    ) -> PresenceSnapshot {
        self.observe_at(advertisements, Instant::now())
    }

    fn observe_at(
        &mut self,
        advertisements: impl IntoIterator<Item = NodeAdvertisement>,
        now: Instant,
    ) -> PresenceSnapshot {
        for mut advertisement in advertisements {
            let peer_id = advertisement.peer_id.clone();
            if let Some(previous) = self.entries.get(&peer_id) {
                // A capability advertisement is a complete current report.
                // Replace capacity and model state, but retain additional
                // working addresses learned on earlier private interfaces.
                for address in &previous.advertisement.addresses {
                    if !advertisement.addresses.contains(address) {
                        advertisement.addresses.push(address.clone());
                    }
                }
            }
            self.entries.insert(
                peer_id,
                PresenceEntry {
                    advertisement,
                    last_seen: now,
                },
            );
        }

        self.entries
            .retain(|_, entry| now.saturating_duration_since(entry.last_seen) <= self.retention);

        let mut records = self
            .entries
            .values()
            .map(|entry| {
                let last_seen_age = now.saturating_duration_since(entry.last_seen);
                PresenceRecord {
                    advertisement: entry.advertisement.clone(),
                    status: self.status_for_age(last_seen_age),
                    last_seen_age,
                }
            })
            .collect::<Vec<_>>();
        records.sort_by(|left, right| left.advertisement.peer_id.cmp(&right.advertisement.peer_id));
        PresenceSnapshot { records }
    }

    fn status_for_age(&self, age: Duration) -> ConnectionStatus {
        if age <= self.connected_grace {
            ConnectionStatus::Connected
        } else if age <= self.reconnecting_grace {
            ConnectionStatus::Reconnecting
        } else {
            ConnectionStatus::Unreachable
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use infernet_node::empty_advertisement;

    fn advertisement(peer_id: &str, address: &str) -> NodeAdvertisement {
        empty_advertisement(peer_id.to_owned(), address.to_owned())
    }

    #[test]
    fn one_missed_discovery_window_does_not_disconnect_a_machine() {
        let mut presence = PeerPresence::with_thresholds(
            Duration::from_secs(10),
            Duration::from_secs(20),
            Duration::from_secs(30),
        );
        let now = Instant::now();
        presence.observe_at([advertisement("peer-a", "/ip4/10.0.0.2/tcp/9777")], now);

        let snapshot = presence.observe_at([], now + Duration::from_secs(8));

        assert_eq!(snapshot.records().len(), 1);
        assert_eq!(snapshot.records()[0].status, ConnectionStatus::Connected);
        assert_eq!(snapshot.routable_advertisements().count(), 1);
    }

    #[test]
    fn a_real_outage_is_reconnecting_then_unreachable_then_forgotten() {
        let mut presence = PeerPresence::with_thresholds(
            Duration::from_secs(10),
            Duration::from_secs(20),
            Duration::from_secs(30),
        );
        let now = Instant::now();
        presence.observe_at([advertisement("peer-a", "/ip4/10.0.0.2/tcp/9777")], now);

        let reconnecting = presence.observe_at([], now + Duration::from_secs(11));
        assert_eq!(
            reconnecting.records()[0].status,
            ConnectionStatus::Reconnecting
        );
        assert_eq!(reconnecting.routable_advertisements().count(), 0);

        let unreachable = presence.observe_at([], now + Duration::from_secs(21));
        assert_eq!(
            unreachable.records()[0].status,
            ConnectionStatus::Unreachable
        );
        assert_eq!(unreachable.routable_advertisements().count(), 0);

        let forgotten = presence.observe_at([], now + Duration::from_secs(31));
        assert!(forgotten.records().is_empty());
    }

    #[test]
    fn fresh_report_replaces_obsolete_seeding_state_and_keeps_known_addresses() {
        let mut presence = PeerPresence::default();
        let now = Instant::now();
        let mut first = advertisement("peer-a", "/ip4/192.168.1.20/tcp/9777");
        first.model_shards.push(infernet_protocol::ModelShardInfo {
            model_id: "infernet-chat-v1".to_owned(),
            layers: infernet_model::LayerRange::new(0, 1).unwrap(),
            checksum: "old".to_owned(),
            size_bytes: 1,
            version: "1".to_owned(),
            protocol_version: infernet_protocol::PROTOCOL_VERSION,
        });
        presence.observe_at([first], now);

        let snapshot = presence.observe_at(
            [advertisement("peer-a", "/ip4/100.64.0.20/tcp/9777")],
            now + Duration::from_secs(1),
        );
        let current = &snapshot.records()[0].advertisement;

        assert!(current.model_shards.is_empty());
        assert_eq!(current.addresses.len(), 2);
    }
}

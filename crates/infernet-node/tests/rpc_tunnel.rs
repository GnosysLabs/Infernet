#[path = "../src/rpc_tunnel.rs"]
mod rpc_tunnel;

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use libp2p::swarm::Swarm;
use libp2p_stream as stream;
use libp2p_swarm_test::SwarmExt as _;
use rpc_tunnel::{
    RPC_TUNNEL_PROTOCOL, RpcTunnelAdmission, RpcTunnelAdmissionLimits, RpcTunnelOpenError,
    RpcTunnelProxy, RpcTunnelProxyConfig, RpcTunnelRejectReason, RpcTunnelTicket, RpcTunnelWorker,
    RpcTunnelWorkerConfig, open_rpc_tunnel_stream,
};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep};
use uuid::Uuid;

struct TunnelHarness {
    coordinator_control: stream::Control,
    worker_peer_id: libp2p::PeerId,
    worker: RpcTunnelWorker,
    coordinator_swarm: JoinHandle<()>,
    worker_swarm: JoinHandle<()>,
}

impl Drop for TunnelHarness {
    fn drop(&mut self) {
        self.coordinator_swarm.abort();
        self.worker_swarm.abort();
    }
}

async fn tunnel_harness(
    target_addr: SocketAddr,
    ticket: RpcTunnelTicket,
    limits: RpcTunnelAdmissionLimits,
) -> TunnelHarness {
    let mut coordinator: Swarm<stream::Behaviour> =
        Swarm::new_ephemeral_tokio(|_| stream::Behaviour::new());
    let mut worker: Swarm<stream::Behaviour> =
        Swarm::new_ephemeral_tokio(|_| stream::Behaviour::new());
    let coordinator_peer_id = *coordinator.local_peer_id();
    let worker_peer_id = *worker.local_peer_id();

    let coordinator_control = coordinator.behaviour().new_control();
    let mut worker_control = worker.behaviour().new_control();
    let incoming = worker_control.accept(RPC_TUNNEL_PROTOCOL).unwrap();
    let admission = RpcTunnelAdmission::reserved(limits).unwrap();
    admission.grant_ticket(coordinator_peer_id, ticket);
    let tunnel_worker =
        RpcTunnelWorker::spawn(incoming, RpcTunnelWorkerConfig::new(target_addr, admission))
            .unwrap();

    worker.listen().with_memory_addr_external().await;
    coordinator.connect(&mut worker).await;
    let coordinator_swarm = tokio::spawn(coordinator.loop_on_next());
    let worker_swarm = tokio::spawn(worker.loop_on_next());

    TunnelHarness {
        coordinator_control,
        worker_peer_id,
        worker: tunnel_worker,
        coordinator_swarm,
        worker_swarm,
    }
}

async fn spawn_echo_sidecar() -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        loop {
            let Ok((socket, _)) = listener.accept().await else {
                return;
            };
            tokio::spawn(async move {
                let (mut reader, mut writer) = socket.into_split();
                let _ = tokio::io::copy(&mut reader, &mut writer).await;
            });
        }
    });
    (address, task)
}

async fn wait_until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(3);
    while !condition() {
        assert!(Instant::now() < deadline, "condition did not become true");
        sleep(Duration::from_millis(10)).await;
    }
}

#[test]
fn configs_refuse_non_loopback_tcp_exposure() {
    let admission =
        RpcTunnelAdmission::allow_authenticated_peers(RpcTunnelAdmissionLimits::default()).unwrap();
    let worker = RpcTunnelWorkerConfig::new(
        SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 50_052),
        admission,
    );
    assert!(worker.validate().is_err());

    let mut proxy = RpcTunnelProxyConfig::new(libp2p::PeerId::random(), RpcTunnelTicket::random());
    proxy.bind_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0);
    assert!(proxy.validate().is_err());
}

#[test]
fn tickets_are_redacted_in_debug_output() {
    let ticket = RpcTunnelTicket::from_bytes([0x5a; 32]);
    let debug = format!("{ticket:?}");
    assert!(debug.contains("redacted"));
    assert!(!debug.contains("5a"));
}

#[tokio::test]
async fn loopback_proxy_carries_an_unmodified_duplex_byte_stream() {
    let (sidecar_addr, sidecar_task) = spawn_echo_sidecar().await;
    let ticket = RpcTunnelTicket::random();
    let harness = tunnel_harness(sidecar_addr, ticket, RpcTunnelAdmissionLimits::default()).await;
    let proxy = RpcTunnelProxy::bind(
        harness.coordinator_control.clone(),
        RpcTunnelProxyConfig::new(harness.worker_peer_id, ticket),
    )
    .await
    .unwrap();

    assert_eq!(proxy.local_addr().ip(), IpAddr::V4(Ipv4Addr::LOCALHOST));
    assert_eq!(
        proxy.llama_cpp_endpoint(),
        format!("127.0.0.1:{}", proxy.local_addr().port())
    );

    let payload = b"opaque ggml rpc bytes\0\xff\x01";
    let mut client = TcpStream::connect(proxy.local_addr()).await.unwrap();
    client.write_all(payload).await.unwrap();
    let mut echoed = vec![0_u8; payload.len()];
    client.read_exact(&mut echoed).await.unwrap();
    assert_eq!(echoed, payload);
    drop(client);

    wait_until(|| proxy.telemetry().completed_sessions == 1).await;
    wait_until(|| harness.worker.telemetry().completed_sessions == 1).await;
    let proxy_telemetry = proxy.telemetry();
    assert_eq!(proxy_telemetry.accepted_sessions, 1);
    assert!(proxy_telemetry.bytes_local_to_p2p >= payload.len() as u64);
    assert!(proxy_telemetry.bytes_p2p_to_local >= payload.len() as u64);
    let worker_telemetry = harness.worker.telemetry();
    assert_eq!(worker_telemetry.accepted_sessions, 1);
    assert!(worker_telemetry.bytes_local_to_p2p >= payload.len() as u64);
    assert!(worker_telemetry.bytes_p2p_to_local >= payload.len() as u64);

    proxy.shutdown().await;
    sidecar_task.abort();
}

#[tokio::test]
async fn wrong_reservation_ticket_is_rejected_before_tcp_forwarding() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let expected_ticket = RpcTunnelTicket::random();
    let mut harness = tunnel_harness(
        listener.local_addr().unwrap(),
        expected_ticket,
        RpcTunnelAdmissionLimits::default(),
    )
    .await;

    let error = open_rpc_tunnel_stream(
        &mut harness.coordinator_control,
        harness.worker_peer_id,
        Uuid::new_v4(),
        RpcTunnelTicket::random(),
        Duration::from_secs(2),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        RpcTunnelOpenError::Rejected(RpcTunnelRejectReason::Unauthorized)
    ));

    assert!(
        tokio::time::timeout(Duration::from_millis(100), listener.accept())
            .await
            .is_err(),
        "unauthorized stream reached the loopback sidecar"
    );
    wait_until(|| harness.worker.telemetry().rejected_sessions == 1).await;
    assert_eq!(harness.worker.telemetry().accepted_sessions, 0);
}

#[tokio::test]
async fn per_peer_session_limit_returns_busy() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
    let target_addr = listener.local_addr().unwrap();
    let accept_task = tokio::spawn(async move {
        let mut held_connections = Vec::new();
        loop {
            let Ok((socket, _)) = listener.accept().await else {
                return;
            };
            held_connections.push(socket);
        }
    });
    let ticket = RpcTunnelTicket::random();
    let mut harness = tunnel_harness(
        target_addr,
        ticket,
        RpcTunnelAdmissionLimits {
            max_sessions: 1,
            max_sessions_per_peer: 1,
        },
    )
    .await;

    let first = open_rpc_tunnel_stream(
        &mut harness.coordinator_control,
        harness.worker_peer_id,
        Uuid::new_v4(),
        ticket,
        Duration::from_secs(2),
    )
    .await
    .unwrap();
    let error = open_rpc_tunnel_stream(
        &mut harness.coordinator_control,
        harness.worker_peer_id,
        Uuid::new_v4(),
        ticket,
        Duration::from_secs(2),
    )
    .await
    .unwrap_err();
    assert!(
        matches!(
            &error,
            RpcTunnelOpenError::Rejected(RpcTunnelRejectReason::Busy)
        ),
        "got {error:?}"
    );

    drop(first);
    accept_task.abort();
}

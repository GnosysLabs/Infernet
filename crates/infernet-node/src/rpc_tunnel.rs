//! Authenticated libp2p transport for llama.cpp's unmodified TCP RPC protocol.
//!
//! `ggml-rpc-server` has no authentication of its own.  It must stay bound to
//! loopback.  This module bridges it to an authenticated libp2p peer without
//! ever putting a TCP destination on the wire:
//!
//! ```text
//! llama.cpp -> 127.0.0.1 proxy -> encrypted libp2p stream
//!                                      -> 127.0.0.1 ggml-rpc-server
//! ```
//!
//! The libp2p swarm is responsible for transport authentication and encryption
//! (Infernet uses Noise).  The stream's `PeerId` is therefore the authenticated
//! remote identity.  A reservation ticket can additionally bind a stream to a
//! job selected by the application.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use futures::{AsyncReadExt as _, AsyncWriteExt as _, StreamExt as _};
use libp2p::{PeerId, StreamProtocol, swarm::Stream};
use libp2p_stream::{Control, IncomingStreams, OpenStreamError};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::timeout;
use tokio_util::compat::FuturesAsyncReadCompatExt as _;
use uuid::Uuid;

/// Raw-stream protocol negotiated between an Infernet coordinator and worker.
pub const RPC_TUNNEL_PROTOCOL: StreamProtocol = StreamProtocol::new("/infernet/llama-rpc-tunnel/1");

const OPEN_MAGIC: [u8; 4] = *b"IRPC";
const ACK_MAGIC: [u8; 4] = *b"IRPA";
const TUNNEL_WIRE_VERSION: u16 = 1;
const TICKET_BYTES: usize = 32;
const OPEN_FRAME_BYTES: usize = 4 + 2 + 16 + TICKET_BYTES;
const ACK_FRAME_BYTES: usize = 4 + 2 + 1;
const MAX_PENDING_REJECTIONS: usize = 32;

/// Unforgeable per-job credential delivered to both selected peers over their
/// already authenticated control plane.
///
/// Its `Debug` implementation is deliberately redacted so logs cannot leak a
/// live reservation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RpcTunnelTicket([u8; TICKET_BYTES]);

impl RpcTunnelTicket {
    /// Generates 244 random bits using two UUID v4 values.
    pub fn random() -> Self {
        let first = Uuid::new_v4();
        let second = Uuid::new_v4();
        let mut bytes = [0_u8; TICKET_BYTES];
        bytes[..16].copy_from_slice(first.as_bytes());
        bytes[16..].copy_from_slice(second.as_bytes());
        Self(bytes)
    }

    pub const fn from_bytes(bytes: [u8; TICKET_BYTES]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; TICKET_BYTES] {
        &self.0
    }
}

impl Default for RpcTunnelTicket {
    fn default() -> Self {
        Self([0_u8; TICKET_BYTES])
    }
}

impl fmt::Debug for RpcTunnelTicket {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("RpcTunnelTicket([redacted])")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RpcTunnelAdmissionLimits {
    pub max_sessions: usize,
    pub max_sessions_per_peer: usize,
}

impl Default for RpcTunnelAdmissionLimits {
    fn default() -> Self {
        Self {
            max_sessions: 8,
            max_sessions_per_peer: 2,
        }
    }
}

impl RpcTunnelAdmissionLimits {
    fn validate(self) -> Result<()> {
        if self.max_sessions == 0 {
            bail!("RPC tunnel max_sessions must be greater than zero");
        }
        if self.max_sessions_per_peer == 0 {
            bail!("RPC tunnel max_sessions_per_peer must be greater than zero");
        }
        if self.max_sessions_per_peer > self.max_sessions {
            bail!("RPC tunnel per-peer session limit cannot exceed the global limit");
        }
        Ok(())
    }
}

/// Thread-safe admission state shared by the swarm task and the application's
/// reservation control plane.
///
/// `reserved` is the safe default: no peer can use compute until explicitly
/// granted. `allow_authenticated_peers` is available for volunteer nodes whose
/// product policy intentionally accepts any Noise-authenticated Infernet peer.
#[derive(Clone)]
pub struct RpcTunnelAdmission {
    inner: Arc<Mutex<AdmissionState>>,
    limits: RpcTunnelAdmissionLimits,
}

impl fmt::Debug for RpcTunnelAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state();
        formatter
            .debug_struct("RpcTunnelAdmission")
            .field(
                "allow_authenticated_peers",
                &state.allow_authenticated_peers,
            )
            .field("granted_peers", &state.grants.len())
            .field("active_sessions", &state.active_sessions)
            .field("limits", &self.limits)
            .finish()
    }
}

struct AdmissionState {
    allow_authenticated_peers: bool,
    grants: HashMap<PeerId, Option<RpcTunnelTicket>>,
    active_sessions: usize,
    active_by_peer: HashMap<PeerId, usize>,
}

impl RpcTunnelAdmission {
    pub fn reserved(limits: RpcTunnelAdmissionLimits) -> Result<Self> {
        Self::new(false, limits)
    }

    pub fn allow_authenticated_peers(limits: RpcTunnelAdmissionLimits) -> Result<Self> {
        Self::new(true, limits)
    }

    fn new(allow_authenticated_peers: bool, limits: RpcTunnelAdmissionLimits) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(AdmissionState {
                allow_authenticated_peers,
                grants: HashMap::new(),
                active_sessions: 0,
                active_by_peer: HashMap::new(),
            })),
            limits,
        })
    }

    /// Allows an authenticated peer without requiring a reservation ticket.
    pub fn grant_peer(&self, peer: PeerId) {
        self.state().grants.insert(peer, None);
    }

    /// Allows only this authenticated peer presenting this exact ticket.
    pub fn grant_ticket(&self, peer: PeerId, ticket: RpcTunnelTicket) {
        self.state().grants.insert(peer, Some(ticket));
    }

    /// Prevents future sessions. Existing sessions are allowed to finish.
    pub fn revoke(&self, peer: &PeerId) {
        self.state().grants.remove(peer);
    }

    pub fn active_sessions(&self) -> usize {
        self.state().active_sessions
    }

    fn state(&self) -> MutexGuard<'_, AdmissionState> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn max_sessions(&self) -> usize {
        self.limits.max_sessions
    }

    fn reserve(&self, peer: PeerId) -> std::result::Result<AdmissionPermit, RpcTunnelRejectReason> {
        let mut state = self.state();
        if !state.allow_authenticated_peers && !state.grants.contains_key(&peer) {
            return Err(RpcTunnelRejectReason::Unauthorized);
        }
        let peer_active = state.active_by_peer.get(&peer).copied().unwrap_or(0);
        if state.active_sessions >= self.limits.max_sessions
            || peer_active >= self.limits.max_sessions_per_peer
        {
            return Err(RpcTunnelRejectReason::Busy);
        }

        state.active_sessions += 1;
        state.active_by_peer.insert(peer, peer_active + 1);
        drop(state);
        Ok(AdmissionPermit {
            admission: self.clone(),
            peer,
        })
    }

    fn authorizes(&self, peer: &PeerId, ticket: &RpcTunnelTicket) -> bool {
        let state = self.state();
        if state.allow_authenticated_peers {
            return true;
        }
        match state.grants.get(peer) {
            Some(None) => true,
            Some(Some(expected)) => constant_time_ticket_eq(expected, ticket),
            None => false,
        }
    }

    fn release(&self, peer: &PeerId) {
        let mut state = self.state();
        state.active_sessions = state.active_sessions.saturating_sub(1);
        let Some(peer_active) = state.active_by_peer.get_mut(peer) else {
            return;
        };
        *peer_active = peer_active.saturating_sub(1);
        if *peer_active == 0 {
            state.active_by_peer.remove(peer);
        }
    }
}

fn constant_time_ticket_eq(left: &RpcTunnelTicket, right: &RpcTunnelTicket) -> bool {
    left.0
        .iter()
        .zip(right.0.iter())
        .fold(0_u8, |difference, (left, right)| {
            difference | (left ^ right)
        })
        == 0
}

struct AdmissionPermit {
    admission: RpcTunnelAdmission,
    peer: PeerId,
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        self.admission.release(&self.peer);
    }
}

#[derive(Debug, Clone)]
pub struct RpcTunnelWorkerConfig {
    /// Fixed process-local ggml RPC target. It is never serialized.
    pub target_addr: SocketAddr,
    pub admission: RpcTunnelAdmission,
    pub handshake_timeout: Duration,
    pub connect_timeout: Duration,
    pub max_session_duration: Duration,
}

impl RpcTunnelWorkerConfig {
    pub fn new(target_addr: SocketAddr, admission: RpcTunnelAdmission) -> Self {
        Self {
            target_addr,
            admission,
            handshake_timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(3),
            max_session_duration: Duration::from_secs(30 * 60),
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_loopback_target(self.target_addr, false)?;
        validate_timeout("handshake_timeout", self.handshake_timeout)?;
        validate_timeout("connect_timeout", self.connect_timeout)?;
        validate_timeout("max_session_duration", self.max_session_duration)
    }
}

#[derive(Debug, Clone)]
pub struct RpcTunnelProxyConfig {
    /// Must be an IPv4 loopback address. Port zero asks the OS for a port.
    pub bind_addr: SocketAddr,
    /// Exact worker identity selected by the coordinator.
    pub worker_peer_id: PeerId,
    pub ticket: RpcTunnelTicket,
    pub handshake_timeout: Duration,
    pub max_session_duration: Duration,
    pub max_connections: usize,
}

impl RpcTunnelProxyConfig {
    pub fn new(worker_peer_id: PeerId, ticket: RpcTunnelTicket) -> Self {
        Self {
            bind_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
            worker_peer_id,
            ticket,
            handshake_timeout: Duration::from_secs(10),
            max_session_duration: Duration::from_secs(30 * 60),
            max_connections: 2,
        }
    }

    pub fn validate(&self) -> Result<()> {
        validate_loopback_target(self.bind_addr, true)?;
        validate_timeout("handshake_timeout", self.handshake_timeout)?;
        validate_timeout("max_session_duration", self.max_session_duration)?;
        if self.max_connections == 0 {
            bail!("RPC tunnel proxy max_connections must be greater than zero");
        }
        Ok(())
    }
}

fn validate_loopback_target(address: SocketAddr, allow_zero_port: bool) -> Result<()> {
    if !matches!(address.ip(), IpAddr::V4(ip) if ip.is_loopback()) {
        bail!("RPC tunnel TCP address must be IPv4 loopback, got {address}");
    }
    if !allow_zero_port && address.port() == 0 {
        bail!("RPC tunnel worker target port must be non-zero");
    }
    Ok(())
}

fn validate_timeout(name: &str, duration: Duration) -> Result<()> {
    if duration.is_zero() {
        bail!("RPC tunnel {name} must be greater than zero");
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct RpcTunnelTelemetry {
    pub active_sessions: usize,
    pub accepted_sessions: u64,
    pub rejected_sessions: u64,
    pub completed_sessions: u64,
    pub failed_sessions: u64,
    pub bytes_local_to_p2p: u64,
    pub bytes_p2p_to_local: u64,
    pub last_activity_unix_ms: Option<u64>,
}

#[derive(Default)]
struct TelemetryCounters {
    active_sessions: AtomicUsize,
    accepted_sessions: AtomicU64,
    rejected_sessions: AtomicU64,
    completed_sessions: AtomicU64,
    failed_sessions: AtomicU64,
    bytes_local_to_p2p: AtomicU64,
    bytes_p2p_to_local: AtomicU64,
    last_activity_unix_ms: AtomicU64,
}

impl TelemetryCounters {
    fn snapshot(&self) -> RpcTunnelTelemetry {
        let last_activity_unix_ms = self.last_activity_unix_ms.load(Ordering::Relaxed);
        RpcTunnelTelemetry {
            active_sessions: self.active_sessions.load(Ordering::Relaxed),
            accepted_sessions: self.accepted_sessions.load(Ordering::Relaxed),
            rejected_sessions: self.rejected_sessions.load(Ordering::Relaxed),
            completed_sessions: self.completed_sessions.load(Ordering::Relaxed),
            failed_sessions: self.failed_sessions.load(Ordering::Relaxed),
            bytes_local_to_p2p: self.bytes_local_to_p2p.load(Ordering::Relaxed),
            bytes_p2p_to_local: self.bytes_p2p_to_local.load(Ordering::Relaxed),
            last_activity_unix_ms: (last_activity_unix_ms != 0).then_some(last_activity_unix_ms),
        }
    }

    fn reject(&self) {
        self.rejected_sessions.fetch_add(1, Ordering::Relaxed);
        self.touch();
    }

    fn start(self: &Arc<Self>) -> ActiveSession {
        self.accepted_sessions.fetch_add(1, Ordering::Relaxed);
        self.active_sessions.fetch_add(1, Ordering::Relaxed);
        self.touch();
        ActiveSession {
            counters: Arc::clone(self),
            finished: false,
        }
    }

    fn add_bytes(&self, direction: ByteDirection, count: u64) {
        if count == 0 {
            return;
        }
        match direction {
            ByteDirection::LocalToP2p => {
                self.bytes_local_to_p2p.fetch_add(count, Ordering::Relaxed);
            }
            ByteDirection::P2pToLocal => {
                self.bytes_p2p_to_local.fetch_add(count, Ordering::Relaxed);
            }
        }
        self.touch();
    }

    fn touch(&self) {
        self.last_activity_unix_ms
            .store(now_unix_ms(), Ordering::Relaxed);
    }
}

fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

struct ActiveSession {
    counters: Arc<TelemetryCounters>,
    finished: bool,
}

impl ActiveSession {
    fn finish(mut self, succeeded: bool) {
        if succeeded {
            self.counters
                .completed_sessions
                .fetch_add(1, Ordering::Relaxed);
        } else {
            self.counters
                .failed_sessions
                .fetch_add(1, Ordering::Relaxed);
        }
        self.finished = true;
        self.counters.touch();
    }
}

impl Drop for ActiveSession {
    fn drop(&mut self) {
        self.counters
            .active_sessions
            .fetch_sub(1, Ordering::Relaxed);
        if !self.finished {
            self.counters
                .failed_sessions
                .fetch_add(1, Ordering::Relaxed);
        }
        self.counters.touch();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum RpcTunnelRejectReason {
    Busy = 1,
    Unauthorized = 2,
    WorkerUnavailable = 3,
    ProtocolMismatch = 4,
}

impl RpcTunnelRejectReason {
    fn from_wire(value: u8) -> io::Result<Option<Self>> {
        match value {
            0 => Ok(None),
            1 => Ok(Some(Self::Busy)),
            2 => Ok(Some(Self::Unauthorized)),
            3 => Ok(Some(Self::WorkerUnavailable)),
            4 => Ok(Some(Self::ProtocolMismatch)),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown RPC tunnel acknowledgement status {value}"),
            )),
        }
    }
}

impl fmt::Display for RpcTunnelRejectReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Busy => "worker is at its RPC tunnel session limit",
            Self::Unauthorized => "worker did not authorize this peer or reservation",
            Self::WorkerUnavailable => "worker's loopback llama RPC service is unavailable",
            Self::ProtocolMismatch => "worker rejected the RPC tunnel handshake",
        })
    }
}

#[derive(Debug)]
pub enum RpcTunnelOpenError {
    TimedOut(Duration),
    Stream(OpenStreamError),
    Io(io::Error),
    Rejected(RpcTunnelRejectReason),
}

impl fmt::Display for RpcTunnelOpenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimedOut(duration) => {
                write!(
                    formatter,
                    "RPC tunnel handshake timed out after {duration:?}"
                )
            }
            Self::Stream(error) => write!(formatter, "failed to open RPC tunnel stream: {error}"),
            Self::Io(error) => write!(formatter, "RPC tunnel handshake failed: {error}"),
            Self::Rejected(reason) => write!(formatter, "RPC tunnel rejected: {reason}"),
        }
    }
}

impl Error for RpcTunnelOpenError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Stream(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::TimedOut(_) | Self::Rejected(_) => None,
        }
    }
}

/// Opens and authenticates one byte-stream tunnel to an exact worker PeerId.
/// The returned stream is positioned at the first raw ggml RPC byte.
pub async fn open_rpc_tunnel_stream(
    control: &mut Control,
    worker_peer_id: PeerId,
    session_id: Uuid,
    ticket: RpcTunnelTicket,
    handshake_timeout: Duration,
) -> std::result::Result<Stream, RpcTunnelOpenError> {
    let open = async {
        let mut stream = control
            .open_stream(worker_peer_id, RPC_TUNNEL_PROTOCOL)
            .await
            .map_err(RpcTunnelOpenError::Stream)?;
        write_open_frame(&mut stream, session_id, ticket)
            .await
            .map_err(RpcTunnelOpenError::Io)?;
        match read_ack_frame(&mut stream)
            .await
            .map_err(RpcTunnelOpenError::Io)?
        {
            None => Ok(stream),
            Some(reason) => Err(RpcTunnelOpenError::Rejected(reason)),
        }
    };

    timeout(handshake_timeout, open)
        .await
        .map_err(|_| RpcTunnelOpenError::TimedOut(handshake_timeout))?
}

/// Worker-side acceptor. The caller must keep polling the owning libp2p swarm.
pub struct RpcTunnelWorker {
    telemetry: Arc<TelemetryCounters>,
    task: Option<JoinHandle<()>>,
}

impl RpcTunnelWorker {
    pub fn spawn(incoming: IncomingStreams, config: RpcTunnelWorkerConfig) -> Result<Self> {
        config.validate()?;
        let telemetry = Arc::new(TelemetryCounters::default());
        let task = tokio::spawn(run_worker_accept_loop(
            incoming,
            config,
            Arc::clone(&telemetry),
        ));
        Ok(Self {
            telemetry,
            task: Some(task),
        })
    }

    pub fn telemetry(&self) -> RpcTunnelTelemetry {
        self.telemetry.snapshot()
    }

    pub async fn shutdown(mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
            let _ = task.await;
        }
    }
}

impl Drop for RpcTunnelWorker {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

async fn run_worker_accept_loop(
    mut incoming: IncomingStreams,
    config: RpcTunnelWorkerConfig,
    telemetry: Arc<TelemetryCounters>,
) {
    let mut sessions = JoinSet::new();
    loop {
        tokio::select! {
            incoming_stream = incoming.next() => {
                let Some((peer, stream)) = incoming_stream else {
                    break;
                };
                if sessions.len() >= config.admission.max_sessions() + MAX_PENDING_REJECTIONS {
                    telemetry.reject();
                    drop(stream);
                    continue;
                }
                match config.admission.reserve(peer) {
                    Ok(permit) => {
                        let session_config = config.clone();
                        let session_telemetry = Arc::clone(&telemetry);
                        sessions.spawn(async move {
                            handle_worker_stream(
                                peer,
                                stream,
                                permit,
                                session_config,
                                session_telemetry,
                            ).await;
                        });
                    }
                    Err(reason) => {
                        telemetry.reject();
                        let handshake_timeout = config.handshake_timeout;
                        sessions.spawn(async move {
                            reject_worker_stream(stream, reason, handshake_timeout).await;
                        });
                    }
                }
            }
            Some(_) = sessions.join_next(), if !sessions.is_empty() => {}
        }
    }
}

async fn reject_worker_stream(
    mut stream: Stream,
    reason: RpcTunnelRejectReason,
    handshake_timeout: Duration,
) {
    let rejection = match timeout(handshake_timeout, read_open_frame(&mut stream)).await {
        Ok(Ok(_)) => reason,
        Ok(Err(_)) => RpcTunnelRejectReason::ProtocolMismatch,
        Err(_) => return,
    };
    let _ = timeout(
        handshake_timeout,
        write_ack_frame(&mut stream, Some(rejection)),
    )
    .await;
}

async fn handle_worker_stream(
    peer: PeerId,
    mut stream: Stream,
    _permit: AdmissionPermit,
    config: RpcTunnelWorkerConfig,
    telemetry: Arc<TelemetryCounters>,
) {
    let open = match timeout(config.handshake_timeout, read_open_frame(&mut stream)).await {
        Ok(Ok(open)) => open,
        Ok(Err(_)) => {
            telemetry.reject();
            let _ = timeout(
                config.handshake_timeout,
                write_ack_frame(&mut stream, Some(RpcTunnelRejectReason::ProtocolMismatch)),
            )
            .await;
            return;
        }
        Err(_) => {
            telemetry.reject();
            return;
        }
    };

    if !config.admission.authorizes(&peer, &open.ticket) {
        telemetry.reject();
        let _ = timeout(
            config.handshake_timeout,
            write_ack_frame(&mut stream, Some(RpcTunnelRejectReason::Unauthorized)),
        )
        .await;
        return;
    }

    let local = match timeout(
        config.connect_timeout,
        TcpStream::connect(config.target_addr),
    )
    .await
    {
        Ok(Ok(local)) => local,
        Ok(Err(_)) | Err(_) => {
            telemetry.reject();
            let _ = timeout(
                config.handshake_timeout,
                write_ack_frame(&mut stream, Some(RpcTunnelRejectReason::WorkerUnavailable)),
            )
            .await;
            return;
        }
    };
    let _ = local.set_nodelay(true);

    if !matches!(
        timeout(config.handshake_timeout, write_ack_frame(&mut stream, None)).await,
        Ok(Ok(()))
    ) {
        telemetry.reject();
        return;
    }

    let _session_id = open.session_id;
    bridge_streams(local, stream, config.max_session_duration, telemetry).await;
}

/// Coordinator-side loopback listener passed to llama.cpp as `--rpc`.
pub struct RpcTunnelProxy {
    local_addr: SocketAddr,
    telemetry: Arc<TelemetryCounters>,
    task: Option<JoinHandle<()>>,
}

impl RpcTunnelProxy {
    pub async fn bind(control: Control, config: RpcTunnelProxyConfig) -> Result<Self> {
        config.validate()?;
        let listener = TcpListener::bind(config.bind_addr).await?;
        let local_addr = listener.local_addr()?;
        validate_loopback_target(local_addr, false)?;

        let telemetry = Arc::new(TelemetryCounters::default());
        let task = tokio::spawn(run_proxy_accept_loop(
            listener,
            control,
            config,
            Arc::clone(&telemetry),
        ));
        Ok(Self {
            local_addr,
            telemetry,
            task: Some(task),
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub fn llama_cpp_endpoint(&self) -> String {
        format!("{}:{}", self.local_addr.ip(), self.local_addr.port())
    }

    pub fn telemetry(&self) -> RpcTunnelTelemetry {
        self.telemetry.snapshot()
    }

    pub async fn shutdown(mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
            let _ = task.await;
        }
    }
}

impl Drop for RpcTunnelProxy {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

async fn run_proxy_accept_loop(
    listener: TcpListener,
    control: Control,
    config: RpcTunnelProxyConfig,
    telemetry: Arc<TelemetryCounters>,
) {
    let mut sessions = JoinSet::new();
    loop {
        tokio::select! {
            accepted = listener.accept() => {
                let Ok((local, _)) = accepted else {
                    telemetry.failed_sessions.fetch_add(1, Ordering::Relaxed);
                    telemetry.touch();
                    break;
                };
                if sessions.len() >= config.max_connections {
                    telemetry.reject();
                    drop(local);
                    continue;
                }
                let _ = local.set_nodelay(true);
                let mut session_control = control.clone();
                let session_config = config.clone();
                let session_telemetry = Arc::clone(&telemetry);
                sessions.spawn(async move {
                    let session_id = Uuid::new_v4();
                    match open_rpc_tunnel_stream(
                        &mut session_control,
                        session_config.worker_peer_id,
                        session_id,
                        session_config.ticket,
                        session_config.handshake_timeout,
                    ).await {
                        Ok(stream) => {
                            bridge_streams(
                                local,
                                stream,
                                session_config.max_session_duration,
                                session_telemetry,
                            ).await;
                        }
                        Err(_) => {
                            session_telemetry.reject();
                        }
                    }
                });
            }
            Some(_) = sessions.join_next(), if !sessions.is_empty() => {}
        }
    }
}

#[derive(Clone, Copy)]
enum ByteDirection {
    LocalToP2p,
    P2pToLocal,
}

struct MeteredIo<T> {
    inner: T,
    counters: Arc<TelemetryCounters>,
    read_direction: ByteDirection,
}

impl<T> MeteredIo<T> {
    fn new(inner: T, counters: Arc<TelemetryCounters>, read_direction: ByteDirection) -> Self {
        Self {
            inner,
            counters,
            read_direction,
        }
    }
}

impl<T: AsyncRead + Unpin> AsyncRead for MeteredIo<T> {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let before = buffer.filled().len();
        match Pin::new(&mut this.inner).poll_read(context, buffer) {
            Poll::Ready(Ok(())) => {
                let count = buffer.filled().len().saturating_sub(before) as u64;
                this.counters.add_bytes(this.read_direction, count);
                Poll::Ready(Ok(()))
            }
            other => other,
        }
    }
}

impl<T: AsyncWrite + Unpin> AsyncWrite for MeteredIo<T> {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.get_mut().inner).poll_write(context, buffer)
    }

    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_flush(context)
    }

    fn poll_shutdown(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_shutdown(context)
    }
}

async fn bridge_streams(
    local: TcpStream,
    stream: Stream,
    max_session_duration: Duration,
    telemetry: Arc<TelemetryCounters>,
) {
    let session = telemetry.start();
    let mut local = MeteredIo::new(local, Arc::clone(&telemetry), ByteDirection::LocalToP2p);
    let mut p2p = MeteredIo::new(
        stream.compat(),
        Arc::clone(&telemetry),
        ByteDirection::P2pToLocal,
    );
    let succeeded = matches!(
        timeout(
            max_session_duration,
            tokio::io::copy_bidirectional(&mut local, &mut p2p),
        )
        .await,
        Ok(Ok(_))
    );
    session.finish(succeeded);
}

struct OpenFrame {
    session_id: Uuid,
    ticket: RpcTunnelTicket,
}

async fn write_open_frame(
    stream: &mut Stream,
    session_id: Uuid,
    ticket: RpcTunnelTicket,
) -> io::Result<()> {
    let mut frame = [0_u8; OPEN_FRAME_BYTES];
    frame[..4].copy_from_slice(&OPEN_MAGIC);
    frame[4..6].copy_from_slice(&TUNNEL_WIRE_VERSION.to_be_bytes());
    frame[6..22].copy_from_slice(session_id.as_bytes());
    frame[22..].copy_from_slice(ticket.as_bytes());
    stream.write_all(&frame).await?;
    stream.flush().await
}

async fn read_open_frame(stream: &mut Stream) -> io::Result<OpenFrame> {
    let mut frame = [0_u8; OPEN_FRAME_BYTES];
    stream.read_exact(&mut frame).await?;
    if frame[..4] != OPEN_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid RPC tunnel open-frame magic",
        ));
    }
    let version = u16::from_be_bytes([frame[4], frame[5]]);
    if version != TUNNEL_WIRE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported RPC tunnel wire version {version}"),
        ));
    }
    let session_id = Uuid::from_slice(&frame[6..22]).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid RPC tunnel session id: {error}"),
        )
    })?;
    let mut ticket = [0_u8; TICKET_BYTES];
    ticket.copy_from_slice(&frame[22..]);
    Ok(OpenFrame {
        session_id,
        ticket: RpcTunnelTicket::from_bytes(ticket),
    })
}

async fn write_ack_frame(
    stream: &mut Stream,
    rejection: Option<RpcTunnelRejectReason>,
) -> io::Result<()> {
    let mut frame = [0_u8; ACK_FRAME_BYTES];
    frame[..4].copy_from_slice(&ACK_MAGIC);
    frame[4..6].copy_from_slice(&TUNNEL_WIRE_VERSION.to_be_bytes());
    frame[6] = rejection.map_or(0, |reason| reason as u8);
    stream.write_all(&frame).await?;
    stream.flush().await
}

async fn read_ack_frame(stream: &mut Stream) -> io::Result<Option<RpcTunnelRejectReason>> {
    let mut frame = [0_u8; ACK_FRAME_BYTES];
    stream.read_exact(&mut frame).await?;
    if frame[..4] != ACK_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid RPC tunnel acknowledgement magic",
        ));
    }
    let version = u16::from_be_bytes([frame[4], frame[5]]);
    if version != TUNNEL_WIRE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported RPC tunnel acknowledgement version {version}"),
        ));
    }
    RpcTunnelRejectReason::from_wire(frame[6])
}

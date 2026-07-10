use std::{
    fs,
    net::TcpListener,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

struct ChildGuard(Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn dynamic_discovery_smoke_test() {
    let binary = env!("CARGO_BIN_EXE_infernet-worker");
    let topic = format!(
        "infernet/test/{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let mut workers = Vec::new();
    let mut static_peers = Vec::new();

    for (index, layers) in ["0:3", "3:6", "6:9", "9:12"].into_iter().enumerate() {
        let port = free_tcp_port();
        let log_path = std::env::temp_dir().join(format!(
            "infernet-dynamic-worker-{port}-{}.log",
            layers.replace(':', "-")
        ));
        let stdout = fs::File::create(&log_path).unwrap();
        let child = Command::new(binary)
            .env("INFERNET_MACHINE_ID", format!("distributed-worker-{index}"))
            .args([
                "serve",
                "--model",
                "grid-demo-12",
                "--layers",
                layers,
                "--topic",
                &topic,
                "--p2p-listen",
                &format!("/ip4/127.0.0.1/tcp/{port}"),
            ])
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        static_peers.push(wait_for_static_peer(&log_path, layers));
        workers.push(ChildGuard(child));
    }

    let mut command = Command::new(binary);
    command.env("INFERNET_MACHINE_ID", "distributed-requester");
    command.args([
        "infer",
        "--model",
        "grid-demo-12",
        "--prompt",
        "hello infernet",
        "--topic",
        &topic,
        "--discovery-timeout-ms",
        "12000",
    ]);
    for peer in &static_peers {
        command.args(["--static-peer", peer]);
    }
    let output = command.output().unwrap();

    assert!(
        output.status.success(),
        "infer command failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("infernet-demo-"));
    assert!(stdout.contains("layers 0:3"));
    assert!(stdout.contains("layers 3:6"));
    assert!(stdout.contains("layers 6:9"));
    assert!(stdout.contains("layers 9:12"));
}

#[test]
fn sole_local_route_waits_for_the_discovery_window() {
    let binary = env!("CARGO_BIN_EXE_infernet-worker");
    let topic = format!(
        "infernet/test/sole-local/{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let port = free_tcp_port();
    let log_path = std::env::temp_dir().join(format!("infernet-sole-local-{port}.log"));
    let stdout = fs::File::create(&log_path).unwrap();
    let child = Command::new(binary)
        .env("INFERNET_MACHINE_ID", "sole-local-machine")
        .args([
            "serve",
            "--model",
            "grid-demo-12",
            "--layers",
            "0:12",
            "--topic",
            &topic,
            "--p2p-listen",
            &format!("/ip4/127.0.0.1/tcp/{port}"),
        ])
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let _worker = ChildGuard(child);
    let static_peer = wait_for_static_peer(&log_path, "0:12");

    let started = Instant::now();
    let output = Command::new(binary)
        .env("INFERNET_MACHINE_ID", "sole-local-machine")
        .args([
            "infer",
            "--model",
            "grid-demo-12",
            "--prompt",
            "local exception",
            "--topic",
            &topic,
            "--static-peer",
            &static_peer,
            "--discovery-timeout-ms",
            "1000",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "local inference failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        started.elapsed() >= Duration::from_millis(850),
        "sole-local inference returned before its discovery window closed"
    );
}

#[test]
fn unreachable_next_hop_reports_forwarding_error() {
    let binary = env!("CARGO_BIN_EXE_infernet-worker");
    let topic = format!(
        "infernet/test/unreachable/{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let log_path = std::env::temp_dir().join(format!(
        "infernet-unreachable-worker-{}.log",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let port = free_tcp_port();
    let stdout = fs::File::create(&log_path).unwrap();
    let child = Command::new(binary)
        .env("INFERNET_MACHINE_ID", "unreachable-live-worker")
        .args([
            "serve",
            "--model",
            "grid-demo-12",
            "--layers",
            "0:3",
            "--topic",
            &topic,
            "--p2p-listen",
            &format!("/ip4/127.0.0.1/tcp/{port}"),
        ])
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let _worker = ChildGuard(child);
    let live_peer = wait_for_static_peer(&log_path, "0:3");
    let fake_peer_id = libp2p::identity::Keypair::generate_ed25519()
        .public()
        .to_peer_id()
        .to_string();
    let fake_peer = format!(
        "{}@/ip4/127.0.0.1/tcp/9/p2p/{}#3:12#machine=fake-remote-machine",
        fake_peer_id, fake_peer_id
    );

    thread::sleep(Duration::from_secs(4));

    let output = Command::new(binary)
        .env("INFERNET_MACHINE_ID", "unreachable-requester")
        .args([
            "infer",
            "--model",
            "grid-demo-12",
            "--prompt",
            "hello infernet",
            "--topic",
            &topic,
            "--static-peer",
            &live_peer,
            "--static-peer",
            &fake_peer,
            "--discovery-timeout-ms",
            "12000",
        ])
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "infer command unexpectedly succeeded\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("forward to")
            || stderr.contains("activation request to")
            || stderr.contains("remote activation error")
            || stderr.contains("no valid execution placement"),
        "expected libp2p forwarding error, got stderr:\n{stderr}"
    );
}

#[test]
fn one_remote_machine_never_receives_the_whole_request() {
    let binary = env!("CARGO_BIN_EXE_infernet-worker");
    let topic = format!(
        "infernet/test/sole-remote/{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let port = free_tcp_port();
    let log_path = std::env::temp_dir().join(format!("infernet-sole-remote-{port}.log"));
    let stdout = fs::File::create(&log_path).unwrap();
    let child = Command::new(binary)
        .env("INFERNET_MACHINE_ID", "sole-remote-worker")
        .args([
            "serve",
            "--model",
            "grid-demo-12",
            "--layers",
            "0:12",
            "--topic",
            &topic,
            "--p2p-listen",
            &format!("/ip4/127.0.0.1/tcp/{port}"),
        ])
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let _worker = ChildGuard(child);
    let static_peer = wait_for_static_peer(&log_path, "0:12");

    let output = Command::new(binary)
        .env("INFERNET_MACHINE_ID", "sole-remote-requester")
        .args([
            "infer",
            "--model",
            "grid-demo-12",
            "--prompt",
            "this must not run remotely",
            "--topic",
            &topic,
            "--static-peer",
            &static_peer,
            "--discovery-timeout-ms",
            "1000",
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("no valid execution placement")
            || stderr.contains("second eligible physical machine"),
        "expected sole-remote placement rejection, got stderr:\n{stderr}"
    );
}

fn wait_for_static_peer(log: &std::path::Path, layers: &str) -> String {
    let deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < deadline {
        if let Ok(contents) = fs::read_to_string(log) {
            let peer_id = contents
                .lines()
                .find_map(|line| line.strip_prefix("peer_id="))
                .map(str::to_owned);
            let machine_id = contents
                .lines()
                .find_map(|line| line.strip_prefix("machine_id="))
                .map(str::to_owned);
            let address = contents
                .lines()
                .find_map(|line| line.strip_prefix("libp2p_listen=/ip4/127.0.0.1"))
                .map(|suffix| format!("/ip4/127.0.0.1{suffix}"));
            if let (Some(peer_id), Some(machine_id), Some(address)) = (peer_id, machine_id, address)
            {
                return format!("{peer_id}@{address}#{layers}#machine={machine_id}");
            }
        }
        thread::sleep(Duration::from_millis(100));
    }

    panic!("timed out waiting for worker address in {}", log.display());
}

fn free_tcp_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

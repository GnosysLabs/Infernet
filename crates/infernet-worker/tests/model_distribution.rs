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
fn shard_downloaders_become_seeders() {
    let binary = env!("CARGO_BIN_EXE_infernet-worker");
    let test_id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let topic = format!("infernet/model-test/{test_id}");
    let temp = std::env::temp_dir().join(format!("infernet-model-distribution-{test_id}"));
    let seed_cache = temp.join("seed-cache");
    let mirror_cache = temp.join("mirror-cache");
    let third_cache = temp.join("third-cache");
    let seed_file = temp.join("seed.shard");
    let seed_log = temp.join("seed.stdout");
    let mirror_log = temp.join("mirror.stdout");
    fs::create_dir_all(&temp).unwrap();
    fs::write(&seed_file, b"infernet model shard payload").unwrap();

    let import_output = Command::new(binary)
        .args([
            "model",
            "import",
            "--cache-dir",
            seed_cache.to_str().unwrap(),
            "--model",
            "grid-demo-12",
            "--layers",
            "0:3",
            "--file",
            seed_file.to_str().unwrap(),
            "--version",
            "v1",
        ])
        .output()
        .unwrap();
    assert!(
        import_output.status.success(),
        "import failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&import_output.stdout),
        String::from_utf8_lossy(&import_output.stderr)
    );
    let checksum = parse_checksum(&String::from_utf8_lossy(&import_output.stdout));
    let shard_size = fs::metadata(&seed_file).unwrap().len();
    let seed_port = free_tcp_port();

    let seed_stdout = fs::File::create(&seed_log).unwrap();
    let seed = Command::new(binary)
        .args([
            "model",
            "serve",
            "--cache-dir",
            seed_cache.to_str().unwrap(),
            "--topic",
            &topic,
            "--p2p-listen",
            &format!("/ip4/127.0.0.1/tcp/{seed_port}"),
        ])
        .stdout(Stdio::from(seed_stdout))
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut seed = ChildGuard(seed);
    let seed_peer = wait_for_peer_address(&seed_log);
    let seed_static_peer = format!(
        "{}@{}#grid-demo-12:0:3:{}:{}:v1",
        seed_peer.peer_id, seed_peer.address, checksum, shard_size
    );

    let mirror_stdout = fs::File::create(&mirror_log).unwrap();
    let mirror_stderr = fs::File::create(temp.join("mirror.stderr")).unwrap();
    let mirror = Command::new(binary)
        .args([
            "model",
            "mirror",
            "--cache-dir",
            mirror_cache.to_str().unwrap(),
            "--model",
            "grid-demo-12",
            "--layers",
            "0:3",
            "--checksum",
            &checksum,
            "--version",
            "v1",
            "--topic",
            &topic,
            "--static-peer",
            &seed_static_peer,
            "--discovery-timeout-ms",
            "20000",
        ])
        .stdout(Stdio::from(mirror_stdout))
        .stderr(Stdio::from(mirror_stderr))
        .spawn()
        .unwrap();
    let mut mirror = ChildGuard(mirror);

    wait_for_cached_shard(&mirror_cache, &mut mirror);
    let mirror_peer = wait_for_mirror_peer_address(&mirror_log);
    let mirror_static_peer = format!(
        "{}@{}#grid-demo-12:0:3:{}:{}:v1",
        mirror_peer.peer_id, mirror_peer.address, checksum, shard_size
    );
    let _ = seed.0.kill();
    let _ = seed.0.wait();
    thread::sleep(Duration::from_millis(500));

    let third_output = Command::new(binary)
        .args([
            "model",
            "fetch",
            "--cache-dir",
            third_cache.to_str().unwrap(),
            "--model",
            "grid-demo-12",
            "--layers",
            "0:3",
            "--checksum",
            &checksum,
            "--version",
            "v1",
            "--topic",
            &topic,
            "--static-peer",
            &mirror_static_peer,
            "--discovery-timeout-ms",
            "15000",
        ])
        .output()
        .unwrap();

    assert!(
        third_output.status.success(),
        "third fetch failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&third_output.stdout),
        String::from_utf8_lossy(&third_output.stderr)
    );
    let stdout = String::from_utf8_lossy(&third_output.stdout);
    assert!(stdout.contains("downloaded_from="));
    assert!(stdout.contains(&checksum));
    assert!(cache_has_meta(&third_cache));

    let _ = fs::remove_dir_all(temp);
}

fn parse_checksum(stdout: &str) -> String {
    stdout
        .split_whitespace()
        .find_map(|part| part.strip_prefix("checksum="))
        .expect("import output should include checksum")
        .to_owned()
}

struct PeerAddress {
    peer_id: String,
    address: String,
}

fn wait_for_peer_address(log: &std::path::Path) -> PeerAddress {
    let deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < deadline {
        if let Ok(contents) = fs::read_to_string(log) {
            let peer_id = contents
                .lines()
                .find_map(|line| line.strip_prefix("peer_id="))
                .map(str::to_owned);
            let address = contents
                .lines()
                .find_map(|line| line.strip_prefix("libp2p_listen=/ip4/127.0.0.1"))
                .map(|suffix| format!("/ip4/127.0.0.1{suffix}"));

            if let (Some(peer_id), Some(address)) = (peer_id, address) {
                return PeerAddress { peer_id, address };
            }
        }
        thread::sleep(Duration::from_millis(100));
    }

    panic!("timed out waiting for peer address in {}", log.display());
}

fn wait_for_mirror_peer_address(log: &std::path::Path) -> PeerAddress {
    let deadline = Instant::now() + Duration::from_secs(8);
    while Instant::now() < deadline {
        if let Ok(contents) = fs::read_to_string(log) {
            let peer_id = contents
                .lines()
                .rev()
                .find_map(|line| line.strip_prefix("peer_id="))
                .map(str::to_owned);
            let mut seen_mirroring = false;
            let mut address = None;

            for line in contents.lines() {
                if line == "mirroring=true" {
                    seen_mirroring = true;
                    continue;
                }

                if seen_mirroring {
                    if let Some(suffix) = line.strip_prefix("libp2p_listen=/ip4/127.0.0.1") {
                        address = Some(format!("/ip4/127.0.0.1{suffix}"));
                    }
                }
            }

            if let (Some(peer_id), Some(address)) = (peer_id, address) {
                return PeerAddress { peer_id, address };
            }
        }
        thread::sleep(Duration::from_millis(100));
    }

    panic!(
        "timed out waiting for mirror peer address in {}",
        log.display()
    );
}

fn free_tcp_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.local_addr().unwrap().port()
}

fn wait_for_cached_shard(cache: &std::path::Path, child: &mut ChildGuard) {
    let deadline = Instant::now() + Duration::from_secs(25);
    while Instant::now() < deadline {
        if cache_has_meta(cache) {
            return;
        }
        if let Some(status) = child.0.try_wait().unwrap() {
            let root = cache.parent().unwrap_or(cache);
            let stdout = fs::read_to_string(root.join("mirror.stdout")).unwrap_or_default();
            let stderr = fs::read_to_string(root.join("mirror.stderr")).unwrap_or_default();
            panic!(
                "mirror exited before caching shard: {status}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
        }
        thread::sleep(Duration::from_millis(250));
    }

    panic!("timed out waiting for mirror cache at {}", cache.display());
}

fn cache_has_meta(cache: &std::path::Path) -> bool {
    let meta = cache.join("meta");
    fs::read_dir(meta)
        .ok()
        .map(|mut entries| entries.any(|entry| entry.is_ok()))
        .unwrap_or(false)
}

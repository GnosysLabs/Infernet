use std::{
    fs,
    io::{self, Seek, Write},
    net::TcpListener,
    path::Path,
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
    let seed_file = temp.join("seed.gguf");
    let seed_log = temp.join("seed.stdout");
    let mirror_log = temp.join("mirror.stdout");
    fs::create_dir_all(&temp).unwrap();
    write_test_gguf(&seed_file).unwrap();

    let import_output = Command::new(binary)
        .args([
            "model",
            "add-local",
            "--cache-dir",
            seed_cache.to_str().unwrap(),
            "--model",
            "grid-demo-12",
            "--gguf",
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
    let (checksum, shard_size) =
        parse_model_shard_info(&String::from_utf8_lossy(&import_output.stdout), "0:3");
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

fn parse_model_shard_info(stdout: &str, layers: &str) -> (String, u64) {
    for line in stdout.lines() {
        if !line.starts_with("model_shard ") || !line.contains(&format!("layers={layers}")) {
            continue;
        }

        let checksum = line
            .split_whitespace()
            .find_map(|part| part.strip_prefix("checksum="))
            .expect("model_shard output should include checksum")
            .to_owned();
        let size = line
            .split_whitespace()
            .find_map(|part| part.strip_prefix("size="))
            .expect("model_shard output should include size")
            .parse::<u64>()
            .expect("model_shard size should be numeric");
        return (checksum, size);
    }

    panic!("model_shard output should include layers={layers}\n{stdout}");
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

fn write_test_gguf(path: &Path) -> io::Result<()> {
    let mut output = fs::File::create(path)?;
    output.write_all(b"GGUF")?;
    write_u32(&mut output, 3)?;
    write_u64(&mut output, 14)?;
    write_u64(&mut output, 4)?;

    write_string(&mut output, "general.architecture")?;
    write_u32(&mut output, 8)?;
    write_string(&mut output, "demo-transformer")?;
    write_string(&mut output, "demo-transformer.block_count")?;
    write_u32(&mut output, 4)?;
    write_u32(&mut output, 12)?;
    write_string(&mut output, "demo-transformer.embedding_length")?;
    write_u32(&mut output, 4)?;
    write_u32(&mut output, 16)?;
    write_string(&mut output, "general.alignment")?;
    write_u32(&mut output, 4)?;
    write_u32(&mut output, 32)?;

    let mut tensor_index = 0_u64;
    write_test_tensor_info(&mut output, "token_embd.weight", tensor_index * 32)?;
    tensor_index += 1;
    for layer in 0..12 {
        write_test_tensor_info(
            &mut output,
            &format!("blk.{layer}.attn_norm.weight"),
            tensor_index * 32,
        )?;
        tensor_index += 1;
    }
    write_test_tensor_info(&mut output, "output_norm.weight", tensor_index * 32)?;

    let header_end = output.stream_position()?;
    let data_start = align_up(header_end, 32);
    write_zero_padding(&mut output, data_start - header_end)?;
    for value in 0_u8..14 {
        output.write_all(&[value; 4])?;
        write_zero_padding(&mut output, 28)?;
    }

    Ok(())
}

fn write_test_tensor_info(output: &mut impl Write, name: &str, offset: u64) -> io::Result<()> {
    write_string(output, name)?;
    write_u32(output, 1)?;
    write_u64(output, 1)?;
    write_u32(output, 0)?;
    write_u64(output, offset)
}

fn write_string(output: &mut impl Write, value: &str) -> io::Result<()> {
    write_u64(output, value.len() as u64)?;
    output.write_all(value.as_bytes())
}

fn write_u32(output: &mut impl Write, value: u32) -> io::Result<()> {
    output.write_all(&value.to_le_bytes())
}

fn write_u64(output: &mut impl Write, value: u64) -> io::Result<()> {
    output.write_all(&value.to_le_bytes())
}

fn write_zero_padding(output: &mut impl Write, len: u64) -> io::Result<()> {
    output.write_all(&vec![0; len as usize])
}

fn align_up(value: u64, alignment: u64) -> u64 {
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value + (alignment - remainder)
    }
}

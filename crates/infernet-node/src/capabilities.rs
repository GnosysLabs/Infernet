use std::process::Command;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
use std::fs;

use infernet_protocol::NodeCapabilities;

const KIBIBYTE: u64 = 1024;
const MEBIBYTE: u64 = 1024 * KIBIBYTE;
const AVAILABLE_MEMORY_REFRESH_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, PartialEq, Eq)]
struct MemoryStats {
    total_bytes: u64,
    available_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NvidiaDevice {
    name: String,
    total_memory_bytes: u64,
    available_memory_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AvailableMemory {
    ram_bytes: u64,
    accelerator_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct AvailableMemoryProbe {
    ram_bytes: Option<u64>,
    accelerator_bytes: Option<u64>,
}

#[derive(Debug)]
struct AvailableMemoryCache {
    value: AvailableMemory,
    last_refresh: Instant,
}

impl AvailableMemoryCache {
    fn refresh_if_due(
        &mut self,
        now: Instant,
        refresh_interval: Duration,
        probe: impl FnOnce() -> AvailableMemoryProbe,
    ) -> AvailableMemory {
        if now.saturating_duration_since(self.last_refresh) < refresh_interval {
            return self.value;
        }

        let update = probe();
        if let Some(ram_bytes) = update.ram_bytes {
            self.value.ram_bytes = ram_bytes;
        }
        if let Some(accelerator_bytes) = update.accelerator_bytes {
            self.value.accelerator_bytes = accelerator_bytes;
        }
        self.last_refresh = now;
        self.value
    }
}

static DETECTED_HARDWARE: OnceLock<NodeCapabilities> = OnceLock::new();
static AVAILABLE_MEMORY: OnceLock<Mutex<AvailableMemoryCache>> = OnceLock::new();

/// Detects the hardware resources this node can offer without assuming any
/// optional system command is installed. Unknown values are reported as zero.
pub fn detect_node_capabilities() -> NodeCapabilities {
    let mut capabilities = DETECTED_HARDWARE
        .get_or_init(detect_static_hardware)
        .clone();
    let available_memory = detect_available_memory(&capabilities);
    capabilities.available_ram_bytes = available_memory.ram_bytes.min(capabilities.total_ram_bytes);
    capabilities.available_accelerator_memory_bytes = available_memory
        .accelerator_bytes
        .min(capabilities.total_accelerator_memory_bytes);
    capabilities.max_sessions = configured_u32("INFERNET_MAX_SESSIONS")
        .unwrap_or(capabilities.max_sessions)
        .max(1);
    capabilities.active_sessions = configured_u32("INFERNET_ACTIVE_SESSIONS")
        .unwrap_or(0)
        .min(capabilities.max_sessions);
    capabilities.queue_depth = configured_u32("INFERNET_QUEUE_DEPTH").unwrap_or(0);
    capabilities
}

fn detect_available_memory(capabilities: &NodeCapabilities) -> AvailableMemory {
    let initial = AvailableMemory {
        ram_bytes: capabilities.available_ram_bytes,
        accelerator_bytes: capabilities.available_accelerator_memory_bytes,
    };
    let cache = AVAILABLE_MEMORY.get_or_init(|| {
        Mutex::new(AvailableMemoryCache {
            value: initial,
            last_refresh: Instant::now(),
        })
    });
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.refresh_if_due(Instant::now(), AVAILABLE_MEMORY_REFRESH_INTERVAL, || {
        probe_available_memory(capabilities)
    })
}

fn probe_available_memory(capabilities: &NodeCapabilities) -> AvailableMemoryProbe {
    let ram_bytes = detect_available_ram_bytes();
    let accelerator_bytes = match capabilities.compute_backend.as_str() {
        "cuda" => detect_nvidia_available_memory(&capabilities.device_name),
        "metal" if capabilities.unified_memory => ram_bytes,
        "cpu" => Some(0),
        _ => None,
    };
    AvailableMemoryProbe {
        ram_bytes,
        accelerator_bytes,
    }
}

fn detect_static_hardware() -> NodeCapabilities {
    let os = std::env::consts::OS.to_owned();
    let arch = std::env::consts::ARCH.to_owned();
    let logical_cpu_cores = std::thread::available_parallelism()
        .map(|count| count.get() as u32)
        .unwrap_or(1);
    let memory = detect_system_memory().unwrap_or(MemoryStats {
        total_bytes: 0,
        available_bytes: 0,
    });
    let cpu_name = detect_cpu_name().unwrap_or_else(|| format!("{arch} CPU"));
    let mut capabilities = NodeCapabilities {
        os,
        arch,
        compute_backend: "cpu".to_owned(),
        device_name: cpu_name,
        logical_cpu_cores,
        total_ram_bytes: memory.total_bytes,
        available_ram_bytes: memory.available_bytes,
        total_accelerator_memory_bytes: 0,
        available_accelerator_memory_bytes: 0,
        unified_memory: false,
        max_sessions: 1,
        active_sessions: 0,
        measured_prefill_tokens_per_second: None,
        measured_decode_tokens_per_second: None,
        queue_depth: 0,
    };

    if let Some(device) = detect_nvidia_device() {
        capabilities.compute_backend = "cuda".to_owned();
        capabilities.device_name = device.name;
        capabilities.total_accelerator_memory_bytes = device.total_memory_bytes;
        capabilities.available_accelerator_memory_bytes = device.available_memory_bytes;
    } else if capabilities.os == "macos"
        && matches!(capabilities.arch.as_str(), "aarch64" | "arm64")
    {
        capabilities.compute_backend = "metal".to_owned();
        capabilities.unified_memory = true;
        capabilities.total_accelerator_memory_bytes = capabilities.total_ram_bytes;
        capabilities.available_accelerator_memory_bytes = capabilities.available_ram_bytes;
    }

    capabilities
}

fn configured_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok()?.trim().parse().ok()
}

fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }

    String::from_utf8(output.stdout).ok()
}

fn detect_system_memory() -> Option<MemoryStats> {
    #[cfg(target_os = "linux")]
    {
        return fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|input| parse_proc_meminfo(&input));
    }

    #[cfg(target_os = "macos")]
    {
        let total_bytes = command_stdout("sysctl", &["-n", "hw.memsize"])?
            .trim()
            .parse::<u64>()
            .ok()?;
        let available_bytes = command_stdout("vm_stat", &[])
            .and_then(|input| parse_vm_stat_available_bytes(&input))
            .unwrap_or(0)
            .min(total_bytes);
        return Some(MemoryStats {
            total_bytes,
            available_bytes,
        });
    }

    #[allow(unreachable_code)]
    None
}

fn detect_available_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        return fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|input| parse_proc_meminfo(&input))
            .map(|memory| memory.available_bytes);
    }

    #[cfg(target_os = "macos")]
    {
        return command_stdout("vm_stat", &[])
            .and_then(|input| parse_vm_stat_available_bytes(&input));
    }

    #[allow(unreachable_code)]
    None
}

fn detect_cpu_name() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
        return parse_linux_cpu_name(&cpuinfo);
    }

    #[cfg(target_os = "macos")]
    {
        return command_stdout("sysctl", &["-n", "machdep.cpu.brand_string"])
            .or_else(|| command_stdout("sysctl", &["-n", "hw.model"]))
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
    }

    #[allow(unreachable_code)]
    None
}

fn detect_nvidia_device() -> Option<NvidiaDevice> {
    // A worker currently targets one accelerator. Report the device with the
    // most immediately available memory instead of overstating usable memory
    // by summing GPUs that may not support a single shared allocation.
    query_nvidia_devices()
        .into_iter()
        .max_by_key(|device| device.available_memory_bytes)
}

fn detect_nvidia_available_memory(device_name: &str) -> Option<u64> {
    query_nvidia_devices()
        .into_iter()
        .filter(|device| device.name == device_name)
        .map(|device| device.available_memory_bytes)
        .max()
}

fn query_nvidia_devices() -> Vec<NvidiaDevice> {
    let output = command_stdout(
        "nvidia-smi",
        &[
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
    );

    output
        .as_deref()
        .map(parse_nvidia_smi_csv)
        .unwrap_or_default()
}

#[cfg(any(target_os = "linux", test))]
fn parse_proc_meminfo(input: &str) -> Option<MemoryStats> {
    let mut total_kib = None;
    let mut available_kib = None;
    let mut free_kib = 0_u64;
    let mut buffers_kib = 0_u64;
    let mut cached_kib = 0_u64;
    let mut reclaimable_kib = 0_u64;
    let mut shared_kib = 0_u64;

    for line in input.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let Some(kib) = value.split_whitespace().next().and_then(|v| v.parse().ok()) else {
            continue;
        };

        match key {
            "MemTotal" => total_kib = Some(kib),
            "MemAvailable" => available_kib = Some(kib),
            "MemFree" => free_kib = kib,
            "Buffers" => buffers_kib = kib,
            "Cached" => cached_kib = kib,
            "SReclaimable" => reclaimable_kib = kib,
            "Shmem" => shared_kib = kib,
            _ => {}
        }
    }

    let total_kib: u64 = total_kib?;
    let available_kib = available_kib.unwrap_or_else(|| {
        free_kib
            .saturating_add(buffers_kib)
            .saturating_add(cached_kib)
            .saturating_add(reclaimable_kib)
            .saturating_sub(shared_kib)
    });

    Some(MemoryStats {
        total_bytes: total_kib.saturating_mul(KIBIBYTE),
        available_bytes: available_kib.min(total_kib).saturating_mul(KIBIBYTE),
    })
}

#[cfg(any(target_os = "macos", test))]
fn parse_vm_stat_available_bytes(input: &str) -> Option<u64> {
    let first_line = input.lines().next()?;
    let page_size = first_line
        .split_whitespace()
        .find_map(|word| word.parse::<u64>().ok())?;
    let mut available_pages = 0_u64;

    for line in input.lines().skip(1) {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if !matches!(
            key.trim(),
            "Pages free" | "Pages inactive" | "Pages speculative" | "Pages purgeable"
        ) {
            continue;
        }

        if let Ok(pages) = value.trim().trim_end_matches('.').parse::<u64>() {
            available_pages = available_pages.saturating_add(pages);
        }
    }

    Some(available_pages.saturating_mul(page_size))
}

#[cfg(any(target_os = "linux", test))]
fn parse_linux_cpu_name(input: &str) -> Option<String> {
    input.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        matches!(key.trim(), "model name" | "Hardware")
            .then(|| value.trim().to_owned())
            .filter(|value| !value.is_empty())
    })
}

fn parse_nvidia_smi_csv(input: &str) -> Vec<NvidiaDevice> {
    input
        .lines()
        .filter_map(|line| {
            let mut fields = line.rsplitn(3, ',');
            let available_mib = fields.next()?.trim().parse::<u64>().ok()?;
            let total_mib = fields.next()?.trim().parse::<u64>().ok()?;
            let name = fields.next()?.trim();
            if name.is_empty() {
                return None;
            }

            Some(NvidiaDevice {
                name: name.to_owned(),
                total_memory_bytes: total_mib.saturating_mul(MEBIBYTE),
                available_memory_bytes: available_mib.min(total_mib).saturating_mul(MEBIBYTE),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn parses_linux_mem_available() {
        let input = "MemTotal:       32768000 kB\nMemFree:         1000000 kB\nMemAvailable:   12000000 kB\nCached:           4000000 kB\n";
        let parsed = parse_proc_meminfo(input).unwrap();

        assert_eq!(parsed.total_bytes, 32_768_000 * KIBIBYTE);
        assert_eq!(parsed.available_bytes, 12_000_000 * KIBIBYTE);
    }

    #[test]
    fn estimates_linux_available_memory_for_older_kernels() {
        let input = "MemTotal: 10000 kB\nMemFree: 1000 kB\nBuffers: 500 kB\nCached: 3000 kB\nSReclaimable: 700 kB\nShmem: 200 kB\n";
        let parsed = parse_proc_meminfo(input).unwrap();

        assert_eq!(parsed.available_bytes, 5_000 * KIBIBYTE);
    }

    #[test]
    fn parses_macos_vm_stat_with_large_pages() {
        let input = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:                               100.\nPages active:                             900.\nPages inactive:                           200.\nPages speculative:                         10.\nPages purgeable:                           20.\n";

        assert_eq!(parse_vm_stat_available_bytes(input), Some(330 * 16_384));
    }

    #[test]
    fn parses_nvidia_csv_and_ignores_malformed_devices() {
        let parsed = parse_nvidia_smi_csv(
            "NVIDIA GeForce RTX 3090, 24576, 22000\nNVIDIA GeForce RTX 4060, 8188, 7600\nbroken row\n",
        );

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "NVIDIA GeForce RTX 3090");
        assert_eq!(parsed[0].total_memory_bytes, 24_576 * MEBIBYTE);
        assert_eq!(parsed[1].available_memory_bytes, 7_600 * MEBIBYTE);
    }

    #[test]
    fn nvidia_name_may_contain_a_comma() {
        let parsed = parse_nvidia_smi_csv("NVIDIA Example, Engineering Sample, 4096, 2048\n");

        assert_eq!(parsed[0].name, "NVIDIA Example, Engineering Sample");
    }

    #[test]
    fn parses_linux_cpu_name() {
        let input = "processor : 0\nmodel name : AMD Ryzen 9 7950X\n";
        assert_eq!(
            parse_linux_cpu_name(input).as_deref(),
            Some("AMD Ryzen 9 7950X")
        );
    }

    #[test]
    fn available_memory_cache_throttles_probes_and_applies_partial_updates() {
        let started_at = Instant::now();
        let mut cache = AvailableMemoryCache {
            value: AvailableMemory {
                ram_bytes: 10_000,
                accelerator_bytes: 8_000,
            },
            last_refresh: started_at,
        };
        let probes = Cell::new(0);

        let unchanged = cache.refresh_if_due(
            started_at + Duration::from_secs(4),
            Duration::from_secs(5),
            || {
                probes.set(probes.get() + 1);
                AvailableMemoryProbe {
                    ram_bytes: Some(9_000),
                    accelerator_bytes: Some(7_000),
                }
            },
        );
        assert_eq!(
            unchanged,
            AvailableMemory {
                ram_bytes: 10_000,
                accelerator_bytes: 8_000
            }
        );
        assert_eq!(probes.get(), 0);

        let refreshed = cache.refresh_if_due(
            started_at + Duration::from_secs(5),
            Duration::from_secs(5),
            || {
                probes.set(probes.get() + 1);
                AvailableMemoryProbe {
                    ram_bytes: Some(9_000),
                    accelerator_bytes: None,
                }
            },
        );
        assert_eq!(
            refreshed,
            AvailableMemory {
                ram_bytes: 9_000,
                accelerator_bytes: 8_000
            }
        );
        assert_eq!(probes.get(), 1);

        let throttled_again = cache.refresh_if_due(
            started_at + Duration::from_secs(6),
            Duration::from_secs(5),
            || {
                probes.set(probes.get() + 1);
                AvailableMemoryProbe::default()
            },
        );
        assert_eq!(throttled_again, refreshed);
        assert_eq!(probes.get(), 1);
    }

    #[test]
    fn detects_current_machine_with_consistent_capacity() {
        let capabilities = detect_node_capabilities();

        assert!(matches!(
            capabilities.compute_backend.as_str(),
            "cuda" | "metal" | "cpu"
        ));
        assert!(!capabilities.device_name.is_empty());
        assert!(capabilities.logical_cpu_cores >= 1);
        assert!(capabilities.max_sessions >= 1);
        assert!(capabilities.active_sessions <= capabilities.max_sessions);
        assert!(capabilities.available_ram_bytes <= capabilities.total_ram_bytes);
        assert!(
            capabilities.available_accelerator_memory_bytes
                <= capabilities.total_accelerator_memory_bytes
        );
    }

    #[test]
    fn only_local_advertisements_claim_local_capabilities() {
        let remote = crate::empty_advertisement("remote".to_owned(), String::new());
        assert!(remote.capabilities.is_none());
        assert!(remote.available_ram_bytes.is_none());
        assert!(remote.available_vram_bytes.is_none());

        let local = crate::local_capability_advertisement("local".to_owned(), String::new());
        let capabilities = local.capabilities.as_ref().unwrap();
        assert_eq!(
            local.available_ram_bytes,
            (capabilities.available_ram_bytes > 0).then_some(capabilities.available_ram_bytes)
        );
        assert_eq!(
            local.available_vram_bytes,
            (capabilities.available_accelerator_memory_bytes > 0)
                .then_some(capabilities.available_accelerator_memory_bytes)
        );

        let mut static_remote = crate::empty_advertisement("remote".to_owned(), String::new());
        assert!(!crate::refresh_local_advertisement_capabilities(
            &mut static_remote,
            "local"
        ));
        assert!(static_remote.capabilities.is_none());
    }
}

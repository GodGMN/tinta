//! Thread budgets for the ONNX sessions.
//!
//! The benchmark showed that a hard-coded thread count oversubscribes the CPUs a
//! process actually has. Use the affinity mask, count distinct physical cores and
//! honour a cgroup CPU quota where one exists. The benchmark's defaults are kept:
//! ASR `min(4, max(1, physical_cores / 2))`, diarization
//! `clamp(physical_cores / 2, 2, 4)`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Threads {
    /// Logical CPUs this process may run on.
    pub available: usize,
    /// Distinct physical cores behind those CPUs, capped by the cgroup quota.
    pub physical: usize,
    /// Threads for the transcription sessions.
    pub asr: usize,
    /// Threads for the diarization sessions.
    pub diarization: usize,
}

impl Threads {
    pub fn new(physical: usize) -> Self {
        let physical = physical.max(1);
        Self {
            available: physical,
            physical,
            asr: asr_threads(physical),
            diarization: diarization_threads(physical),
        }
    }

    /// A small fixed budget for tests that never load a model.
    #[cfg(test)]
    pub fn for_tests() -> Self {
        Self::new(2)
    }
}

/// `min(4, max(1, physical_cores / 2))`, written as a clamp.
pub fn asr_threads(physical_cores: usize) -> usize {
    (physical_cores.max(1) / 2).clamp(1, 4)
}

/// `clamp(physical_cores / 2, 2, 4)`.
pub fn diarization_threads(physical_cores: usize) -> usize {
    (physical_cores.max(1) / 2).clamp(2, 4)
}

pub fn detect() -> Threads {
    let available = allowed_cpu_ids()
        .filter(|ids| !ids.is_empty())
        .map(|ids| ids.len())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1);
    let mut physical = physical_cores(available).unwrap_or(available);
    if let Some(quota) = cgroup_cpu_quota() {
        physical = physical.min(quota);
    }
    Threads {
        available,
        physical: physical.max(1),
        asr: asr_threads(physical),
        diarization: diarization_threads(physical),
    }
}

#[cfg(target_os = "linux")]
fn allowed_cpu_ids() -> Option<Vec<usize>> {
    // SAFETY: an all-zero cpu_set_t is valid, and sched_getaffinity only writes
    // into the buffer we pass with its size.
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let result =
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    if result != 0 {
        return None;
    }
    let mut ids = Vec::new();
    for id in 0..libc::CPU_SETSIZE as usize {
        // SAFETY: id is within the set's range and set is initialized.
        if unsafe { libc::CPU_ISSET(id, &set) } {
            ids.push(id);
        }
    }
    Some(ids)
}

#[cfg(not(target_os = "linux"))]
fn allowed_cpu_ids() -> Option<Vec<usize>> {
    None
}

#[cfg(target_os = "linux")]
fn physical_cores(available: usize) -> Option<usize> {
    let ids = allowed_cpu_ids()?;
    let mut pairs = std::collections::HashSet::new();
    for id in ids {
        let base = format!("/sys/devices/system/cpu/cpu{id}/topology");
        let package = std::fs::read_to_string(format!("{base}/physical_package_id")).ok()?;
        let core = std::fs::read_to_string(format!("{base}/core_id")).ok()?;
        pairs.insert((package.trim().to_string(), core.trim().to_string()));
    }
    if pairs.is_empty() {
        // The topology files can be missing in a container. Fall back to the
        // number of logical CPUs available to the process.
        return Some(available);
    }
    Some(pairs.len())
}

#[cfg(not(target_os = "linux"))]
fn physical_cores(_available: &usize) -> Option<usize> {
    None
}

#[cfg(target_os = "linux")]
fn cgroup_cpu_quota() -> Option<usize> {
    // cgroup v2: "max 100000" or "200000 100000".
    if let Ok(text) = std::fs::read_to_string("/sys/fs/cgroup/cpu.max") {
        let mut parts = text.split_whitespace();
        let quota = parts.next()?;
        let period = parts.next()?;
        if quota != "max" {
            let quota: u64 = quota.parse().ok()?;
            let period: u64 = period.parse().ok()?;
            if quota > 0 && period > 0 {
                return Some(quota.div_ceil(period) as usize);
            }
        }
        return None;
    }
    // cgroup v1.
    let quota = std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        .ok()?
        .trim()
        .parse::<i64>()
        .ok()?;
    let period = std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        .ok()?
        .trim()
        .parse::<i64>()
        .ok()?;
    if quota <= 0 || period <= 0 {
        return None;
    }
    Some((quota as u64).div_ceil(period as u64) as usize)
}

#[cfg(not(target_os = "linux"))]
fn cgroup_cpu_quota() -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_budgets_follow_the_benchmark() {
        assert_eq!(asr_threads(1), 1);
        assert_eq!(asr_threads(2), 1);
        assert_eq!(asr_threads(4), 2);
        assert_eq!(asr_threads(7), 3);
        assert_eq!(asr_threads(8), 4);
        assert_eq!(asr_threads(64), 4);

        // A single core still needs a usable diarization session.
        assert_eq!(diarization_threads(1), 2);
        assert_eq!(diarization_threads(2), 2);
        assert_eq!(diarization_threads(4), 2);
        assert_eq!(diarization_threads(7), 3);
        assert_eq!(diarization_threads(64), 4);
    }

    #[test]
    fn detection_returns_usable_numbers() {
        let threads = detect();
        assert!(threads.available >= 1);
        assert!(threads.physical >= 1);
        assert!((1..=4).contains(&threads.asr));
        assert!((2..=4).contains(&threads.diarization));
    }
}

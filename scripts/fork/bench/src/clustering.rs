//! Post-clustering for speaker diarization.
//!
//! The segmentation + embedding front end over-clusters on purpose: it is far
//! safer to split one speaker into several clusters than to merge two speakers.
//! This module turns that over-clustered partition into a usable one without
//! ever forcing a fixed speaker count:
//!
//! 1. Every cluster gets a centroid (duration-weighted mean of its unit
//!    embeddings, L2-normalized).
//! 2. Clusters are merged agglomeratively, closest centroid pair first, while
//!    the cosine distance between the two centroids stays below
//!    `merge_distance`.
//! 3. Clusters that still hold less than `min_share` of the diarized segment
//!    time are dissolved; each of their units goes to the centroid it is
//!    closest to in embedding space. That is the important part: a tiny cluster
//!    is a piece of somebody's voice, not the speaker who happens to talk next
//!    to it.
//!
//! The whole thing is a pure function over (embedding, seconds) pairs so it can
//! move into `crates/tinta-engine-rs` unchanged. The frozen settings
//! ([`Options::default`]) are tuned on CAMPPlus embeddings extracted from an
//! over-clustered partition at threshold 0.7; another embedding model needs its
//! own over-clustering threshold and its own merge distance.
//!
//! Two stopping rules from the first experiment, a loyalty guard and a
//! "protect large clusters from merging" guard, measured as useless and were
//! removed rather than carried as dead knobs.

/// One unit of speech with the embedding of its audio. A unit is a run of
/// segments that the front end already believes is one speaker turn; it is the
/// smallest thing this module may relabel.
#[derive(Clone, Debug)]
pub struct Unit {
    pub start: f32,
    pub end: f32,
    /// Cluster id from the over-clustered front end.
    pub cluster: usize,
    /// Speaker embedding of the unit audio, any length. It is normalized here.
    /// A zero or non-finite vector marks a failed extraction; such a unit is
    /// not used for any centroid and joins the largest final cluster.
    pub embedding: Vec<f32>,
}

impl Unit {
    pub fn seconds(&self) -> f32 {
        (self.end - self.start).max(0.0)
    }
}

#[derive(Clone, Debug)]
pub struct Options {
    /// Cosine distance between two centroids below which they are merged.
    pub merge_distance: f32,
    /// Clusters holding less than this share of the diarized segment time are
    /// dissolved and their units are reassigned by embedding. 0 disables the
    /// step. The share is computed over the summed unit durations, so
    /// overlapping units count twice and gaps of up to the unit-join limit
    /// inside a unit count as speech.
    pub min_share: f32,
}

impl Default for Options {
    /// The frozen recipe from the task #18 dev sweep. These are also the
    /// defaults of the `postcluster` command and the values the engine should
    /// ship. Valid for CAMPPlus at an over-clustering threshold of 0.7.
    fn default() -> Self {
        Options {
            merge_distance: 0.15,
            min_share: 0.03,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Outcome {
    /// Final cluster label for every input unit, in input order.
    pub labels: Vec<usize>,
    pub clusters_before: usize,
    pub clusters_after: usize,
    /// Number of agglomerative merges performed.
    pub merges: usize,
    /// Number of clusters dissolved by `min_share`.
    pub dissolved: usize,
    /// Number of units reassigned by embedding after the dissolve.
    pub reassigned: usize,
    /// Number of units whose embedding was empty, zero or non-finite. They
    /// were excluded from the clustering and assigned to the largest cluster.
    pub failed_units: usize,
}

/// L2-normalize in place. Returns false when the vector cannot be used: empty,
/// containing a non-finite value (a failed model call) or all zeros.
fn l2_normalize(v: &mut [f32]) -> bool {
    if v.is_empty() || v.iter().any(|x| !x.is_finite()) {
        return false;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if !norm.is_finite() || norm <= 1e-6 {
        return false;
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    true
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    // Both inputs are L2-normalized, so this is 1 - cos. A NaN from a broken
    // embedding is treated as "infinitely far apart", never as a close match.
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    if !dot.is_finite() {
        return f32::INFINITY;
    }
    (1.0 - dot).clamp(0.0, 2.0)
}

/// Duration-weighted, L2-normalized centroid of `members`.
fn centroid(units: &[Unit], members: &[usize]) -> Vec<f32> {
    // Use a member's dimension, not the first input unit's: the first unit may
    // be a failed extraction with an empty embedding.
    let dim = members
        .first()
        .map(|&i| units[i].embedding.len())
        .unwrap_or(0);
    let mut acc = vec![0.0f32; dim];
    let mut total = 0.0f32;
    for &i in members {
        let w = units[i].seconds().max(1e-3);
        total += w;
        for (a, x) in acc.iter_mut().zip(&units[i].embedding) {
            *a += x * w;
        }
    }
    if total > 0.0 {
        for a in acc.iter_mut() {
            *a /= total;
        }
    }
    l2_normalize(&mut acc);
    acc
}

struct Cluster {
    members: Vec<usize>,
    seconds: f32,
    centroid: Vec<f32>,
}

/// Over-clustered `units` -> a small, stable set of speaker labels.
pub fn recluster(units: &[Unit], opts: &Options) -> Outcome {
    if units.is_empty() {
        return Outcome::default();
    }

    // Distinct initial cluster ids, including clusters that only hold failed
    // units, so the caller can see what the front end produced.
    let mut ids: Vec<usize> = units.iter().map(|u| u.cluster).collect();
    ids.sort_unstable();
    ids.dedup();
    let clusters_before = ids.len();

    let mut normalized: Vec<Unit> = units.to_vec();
    let mut valid = vec![false; normalized.len()];
    for (i, u) in normalized.iter_mut().enumerate() {
        valid[i] = l2_normalize(&mut u.embedding);
    }
    let units = &normalized[..];
    let failed_units = valid.iter().filter(|v| !**v).count();

    // Group the usable units by the front end's cluster id.
    let mut by_cluster: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, u) in units.iter().enumerate() {
        if !valid[i] {
            continue;
        }
        match by_cluster.iter_mut().find(|(c, _)| *c == u.cluster) {
            Some((_, m)) => m.push(i),
            None => by_cluster.push((u.cluster, vec![i])),
        }
    }
    by_cluster.sort_by_key(|(c, _)| *c);

    if by_cluster.is_empty() {
        // Nothing usable: one speaker, as good a guess as any.
        return Outcome {
            labels: vec![0; units.len()],
            clusters_before,
            clusters_after: 1,
            failed_units,
            ..Outcome::default()
        };
    }

    let mut live: Vec<Cluster> = by_cluster
        .into_iter()
        .map(|(_, members)| {
            let seconds = members.iter().map(|&i| units[i].seconds()).sum();
            let centroid = centroid(units, &members);
            Cluster {
                members,
                seconds,
                centroid,
            }
        })
        .collect();

    let total: f32 = units.iter().map(|u| u.seconds()).sum::<f32>().max(1e-6);
    let n = live.len();
    let mut merges = 0usize;

    // Cached centroid distances. The merge loop reads this matrix; after a
    // merge only the surviving cluster's row/column is recomputed, so a long
    // meeting does not pay O(k^3 * d) dot products.
    let mut dist = vec![vec![f32::INFINITY; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = cosine_distance(&live[i].centroid, &live[j].centroid);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    let mut active = vec![true; n];
    let mut active_count = n;

    while active_count > 1 {
        let mut best = (0usize, 0usize, f32::INFINITY);
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                // total_cmp orders a NaN as greater than everything, so a
                // broken distance can never win the merge.
                if dist[i][j].total_cmp(&best.2) == std::cmp::Ordering::Less {
                    best = (i, j, dist[i][j]);
                }
            }
        }
        let (i, j, d) = best;
        if d.total_cmp(&opts.merge_distance) != std::cmp::Ordering::Less {
            break;
        }
        let (keep, gone) = (i.min(j), i.max(j));
        let moved = std::mem::take(&mut live[gone].members);
        live[keep].members.extend(moved);
        live[keep].members.sort_unstable();
        live[keep].seconds += live[gone].seconds;
        live[keep].centroid = centroid(units, &live[keep].members);
        active[gone] = false;
        active_count -= 1;
        for k in 0..n {
            if active[k] && k != keep {
                let d = cosine_distance(&live[keep].centroid, &live[k].centroid);
                dist[keep][k] = d;
                dist[k][keep] = d;
            }
        }
        merges += 1;
    }

    let kept: Vec<usize> = (0..n).filter(|&i| active[i]).collect();
    let mut labels = vec![usize::MAX; units.len()];
    for (label, &k) in kept.iter().enumerate() {
        for &m in &live[k].members {
            labels[m] = label;
        }
    }

    // Dissolve clusters that hold too little speech to stand for a speaker on
    // their own. Each of their units goes to the centroid it is closest to in
    // embedding space. Never by time, and the largest cluster is always kept,
    // so the step can never produce zero speakers.
    let mut dissolved = 0usize;
    let mut reassigned = 0usize;
    if kept.len() > 1 && opts.min_share > 0.0 {
        let largest = kept
            .iter()
            .max_by(|a, b| live[**a].seconds.total_cmp(&live[**b].seconds))
            .copied()
            .unwrap_or(kept[0]);
        let threshold = opts.min_share * total;
        let drop: Vec<usize> = kept
            .iter()
            .copied()
            .filter(|&i| i != largest && live[i].seconds < threshold)
            .collect();
        if !drop.is_empty() {
            dissolved = drop.len();
            let keep: Vec<usize> = kept.iter().copied().filter(|i| !drop.contains(i)).collect();
            // Rebuild the kept labels first: their compressed positions are
            // not the original live-vector positions, and the reassigned units
            // below are labelled with those compressed positions.
            for (label, &k) in keep.iter().enumerate() {
                for &m in &live[k].members {
                    labels[m] = label;
                }
            }
            for &k in &drop {
                for &m in &live[k].members {
                    let mut best = (0usize, f32::INFINITY);
                    for (label, &other) in keep.iter().enumerate() {
                        let d = cosine_distance(&units[m].embedding, &live[other].centroid);
                        if d.total_cmp(&best.1) == std::cmp::Ordering::Less {
                            best = (label, d);
                        }
                    }
                    labels[m] = best.0;
                    reassigned += 1;
                }
            }
        }
    }

    // Failed extractions carry no voice information. Give them to the largest
    // final cluster instead of letting a zero vector pick the lowest cluster id
    // (every distance to a zero vector is 1.0).
    if failed_units > 0 {
        let mut totals: Vec<(usize, f32)> = Vec::new();
        for (u, &l) in units.iter().zip(&labels) {
            if l == usize::MAX {
                continue;
            }
            match totals.iter_mut().find(|(k, _)| *k == l) {
                Some((_, t)) => *t += u.seconds(),
                None => totals.push((l, u.seconds())),
            }
        }
        let largest = totals
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
            .map(|(l, _)| *l)
            .unwrap_or(0);
        for l in labels.iter_mut() {
            if *l == usize::MAX {
                *l = largest;
            }
        }
    }

    // Renumber by speech time, largest speaker first, so the output is stable.
    let mut order: Vec<(usize, f32)> = speech_shares(units, &labels);
    order.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut remap: Vec<usize> = vec![usize::MAX; kept.len()];
    let mut rank = 0usize;
    for (label, _) in order {
        if remap[label] == usize::MAX {
            remap[label] = rank;
            rank += 1;
        }
    }
    for l in labels.iter_mut() {
        *l = remap[*l];
    }
    Outcome {
        labels,
        clusters_before,
        clusters_after: rank,
        merges,
        dissolved,
        reassigned,
        failed_units,
    }
}

/// Total speech seconds per final label, sorted smallest first. Labels must be
/// valid indices; callers renumber afterwards if they need a stable order.
pub fn speech_shares(units: &[Unit], labels: &[usize]) -> Vec<(usize, f32)> {
    let mut totals: Vec<(usize, f32)> = Vec::new();
    for (u, &l) in units.iter().zip(labels) {
        match totals.iter_mut().find(|(k, _)| *k == l) {
            Some((_, t)) => *t += u.seconds(),
            None => totals.push((l, u.seconds())),
        }
    }
    totals.sort_by(|a, b| a.1.total_cmp(&b.1));
    totals
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::collections::{HashMap, HashSet};

    fn unit(start: f32, end: f32, cluster: usize, emb: Vec<f32>) -> Unit {
        Unit {
            start,
            end,
            cluster,
            embedding: emb,
        }
    }

    #[test]
    fn defaults_are_the_frozen_recipe() {
        let o = Options::default();
        assert_eq!(o.merge_distance, 0.15);
        assert_eq!(o.min_share, 0.03);
    }

    #[test]
    fn merges_identical_centroids() {
        // Two clusters with the same direction must collapse into one.
        let units = vec![
            unit(0.0, 5.0, 0, vec![1.0, 0.0, 0.0]),
            unit(5.0, 10.0, 1, vec![1.0, 0.01, 0.0]),
        ];
        let out = recluster(&units, &Options::default());
        assert_eq!(out.clusters_after, 1);
        assert_eq!(out.merges, 1);
        assert_eq!(out.labels, vec![0, 0]);
    }

    #[test]
    fn merging_updates_the_distance_matrix() {
        // Twelve near-identical clusters: every merge changes the surviving
        // centroid, so a stale matrix row would stop short of one cluster.
        let units: Vec<Unit> = (0..12)
            .map(|i| unit(i as f32, i as f32 + 1.0, i, vec![1.0, 0.001 * i as f32, 0.0]))
            .collect();
        let out = recluster(&units, &Options::default());
        assert_eq!(out.clusters_after, 1);
        assert_eq!(out.merges, 11);
        assert!(out.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn keeps_distinct_speakers() {
        let units = vec![
            unit(0.0, 5.0, 0, vec![1.0, 0.0, 0.0]),
            unit(5.0, 10.0, 1, vec![0.0, 1.0, 0.0]),
        ];
        let out = recluster(&units, &Options::default());
        assert_eq!(out.clusters_after, 2);
        assert_eq!(out.merges, 0);
    }

    #[test]
    fn small_cluster_goes_to_nearest_voice_not_nearest_time() {
        // Cluster 2 is tiny and sits between two occurrences of cluster 0, but
        // its embedding matches cluster 1. Time-based reassignment would put it
        // in cluster 0; embedding reassignment must put it in cluster 1.
        let units = vec![
            unit(0.0, 4.0, 0, vec![1.0, 0.0, 0.0]),
            unit(4.0, 5.0, 2, vec![0.6, 0.8, 0.0]),
            unit(5.0, 9.0, 0, vec![1.0, 0.0, 0.0]),
            unit(9.0, 13.0, 1, vec![0.0, 1.0, 0.0]),
        ];
        let out = recluster(
            &units,
            &Options {
                merge_distance: 0.15,
                min_share: 0.15,
            },
        );
        assert_eq!(out.dissolved, 1);
        assert_eq!(out.reassigned, 1);
        // Units 0 and 2 are cluster 0, unit 3 is cluster 1, unit 1 joined it.
        assert_eq!(out.labels[1], out.labels[3]);
        assert_ne!(out.labels[1], out.labels[0]);
    }

    #[test]
    fn nan_embedding_does_not_panic_and_joins_largest_cluster() {
        // The NaN unit is first on purpose: the centroid dimension must come
        // from a cluster member, not from units[0].
        let units = vec![
            unit(0.0, 3.0, 2, vec![f32::NAN, 0.0]),
            unit(3.0, 23.0, 0, vec![1.0, 0.0]),
            unit(23.0, 33.0, 1, vec![0.0, 1.0]),
            unit(33.0, 36.0, 3, vec![]),
            unit(36.0, 40.0, 4, vec![0.0, 0.0]),
        ];
        let out = recluster(&units, &Options::default());
        assert_eq!(out.failed_units, 3);
        assert_eq!(out.clusters_after, 2);
        // The 20 s cluster is the largest; failed units join its label.
        assert_eq!(out.labels[1], 0);
        assert_eq!(out.labels[0], out.labels[1]);
        assert_eq!(out.labels[3], out.labels[1]);
        assert_eq!(out.labels[4], out.labels[1]);
        assert_ne!(out.labels[2], out.labels[1]);
    }

    #[test]
    fn all_failed_embeddings_give_one_cluster() {
        let units = vec![
            unit(0.0, 1.0, 0, vec![0.0, 0.0]),
            unit(1.0, 2.0, 1, vec![]),
        ];
        let out = recluster(&units, &Options::default());
        assert_eq!(out.clusters_before, 2);
        assert_eq!(out.clusters_after, 1);
        assert_eq!(out.failed_units, 2);
        assert_eq!(out.labels, vec![0, 0]);
    }

    #[test]
    fn empty_input_is_empty_output() {
        let out = recluster(&[], &Options::default());
        assert_eq!(out.clusters_after, 0);
        assert!(out.labels.is_empty());
    }

    /// Minimal standard-base64 decoder for the i8 fixtures (no new dependency).
    fn decode_i8(s: &str) -> Vec<f32> {
        let mut out = Vec::new();
        let mut buf = 0u32;
        let mut bits = 0u32;
        for c in s.bytes() {
            let v = match c {
                b'A'..=b'Z' => c - b'A',
                b'a'..=b'z' => c - b'a' + 26,
                b'0'..=b'9' => c - b'0' + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' => break,
                _ => continue,
            };
            buf = (buf << 6) | v as u32;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push(((buf >> bits) & 0xff) as u8);
            }
        }
        out.iter().map(|&b| (b as i8) as f32 / 127.0).collect()
    }

    struct Golden {
        name: String,
        units: Vec<Unit>,
        references: Vec<String>,
        expected: Value,
    }

    fn golden(raw: &str) -> Golden {
        let v: Value = serde_json::from_str(raw).expect("golden json");
        let units = v["units"]
            .as_array()
            .expect("units")
            .iter()
            .map(|u| Unit {
                start: u["start"].as_f64().unwrap() as f32,
                end: u["end"].as_f64().unwrap() as f32,
                cluster: u["cluster"].as_u64().unwrap() as usize,
                embedding: decode_i8(u["embedding"].as_str().expect("embedding")),
            })
            .collect();
        let references = v["units"]
            .as_array()
            .expect("units")
            .iter()
            .map(|u| u["reference"].as_str().unwrap_or("").to_string())
            .collect();
        Golden {
            name: v["meeting"].as_str().unwrap_or("?").to_string(),
            units,
            references,
            expected: v["expected"].clone(),
        }
    }

    fn assert_golden(raw: &str) {
        let g = golden(raw);
        let out = recluster(&g.units, &Options::default());
        let e = &g.expected;
        assert_eq!(
            out.clusters_before,
            e["clusters_before"].as_u64().unwrap() as usize,
            "{} clusters_before",
            g.name
        );
        assert_eq!(
            out.clusters_after,
            e["clusters_after"].as_u64().unwrap() as usize,
            "{} clusters_after",
            g.name
        );
        assert_eq!(
            out.merges,
            e["merges"].as_u64().unwrap() as usize,
            "{} merges",
            g.name
        );
        assert_eq!(
            out.dissolved,
            e["dissolved"].as_u64().unwrap() as usize,
            "{} dissolved",
            g.name
        );
        let labels: Vec<usize> = e["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(out.labels, labels, "{} labels", g.name);

        // The pinned labels must be a clean partition of the reference
        // speakers: one label per speaker and no label mixing two speakers.
        let mut by_label: HashMap<usize, HashSet<&str>> = HashMap::new();
        let mut by_ref: HashMap<&str, HashSet<usize>> = HashMap::new();
        for (label, reference) in out.labels.iter().zip(&g.references) {
            by_label.entry(*label).or_default().insert(reference);
            by_ref.entry(reference.as_str()).or_default().insert(*label);
        }
        assert!(
            by_label.values().all(|s| s.len() == 1),
            "{}: a label mixes reference speakers",
            g.name
        );
        assert!(
            by_ref.values().all(|s| s.len() == 1),
            "{}: a reference speaker is split across labels",
            g.name
        );
    }

    // Small extracts of the cached CAMPPlus per-unit embeddings from
    // evidence/tinta-fork/18/results. Each unit's embedding is quantized to i8
    // (scale = its max absolute value) and base64-encoded; the extract and the
    // expected output are documented in docs/fork/diarization.md.
    #[test]
    fn golden_es2004b_cached_extract() {
        assert_golden(include_str!("../tests/data/golden-ES2004b.json"));
    }

    #[test]
    fn golden_es2011c_cached_extract() {
        assert_golden(include_str!("../tests/data/golden-ES2011c.json"));
    }
}

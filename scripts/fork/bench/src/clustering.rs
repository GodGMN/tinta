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
//! 3. Clusters that still hold less than `min_share` of the speech are
//!    dissolved; each of their units goes to the centroid it is closest to in
//!    embedding space. That is the important part: a tiny cluster is a piece of
//!    somebody's voice, not the speaker who happens to talk next to it.
//!
//! The whole thing is a pure function over (embedding, seconds) pairs so it can
//! move into `crates/tinta-engine-rs` unchanged.

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
    /// Clusters holding less than this share of the speech are dissolved and
    /// their units are reassigned by embedding. 0 disables the step.
    pub min_share: f32,
    /// A cluster whose own units are nearest to another centroid more than
    /// `1 - min_loyalty` of the time is a mixed bag, not a speaker; it is
    /// dissolved and its units are reassigned by embedding. 0 disables it.
    pub min_loyalty: f32,
    /// Two clusters that both hold at least this share of the speech are never
    /// merged, whatever their distance. 0 disables the guard. This is the
    /// "never merge two people who both talk a lot" stopping rule.
    pub protect_share: f32,
}

impl Default for Options {
    fn default() -> Self {
        // Tuned on AMI dev only; see docs/fork/diarization.md.
        Options {
            merge_distance: 0.25,
            min_share: 0.02,
            min_loyalty: 0.5,
            protect_share: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
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
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    // Both inputs are L2-normalized, so this is 1 - cos.
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    (1.0 - dot).max(0.0)
}

/// Duration-weighted, L2-normalized centroid of `members`.
fn centroid(units: &[Unit], members: &[usize]) -> Vec<f32> {
    let dim = units.first().map(|u| u.embedding.len()).unwrap_or(0);
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
    let empty = Outcome {
        labels: Vec::new(),
        clusters_before: 0,
        clusters_after: 0,
        merges: 0,
        dissolved: 0,
        reassigned: 0,
    };
    if units.is_empty() {
        return empty;
    }

    let mut normalized: Vec<Unit> = units.to_vec();
    for u in normalized.iter_mut() {
        l2_normalize(&mut u.embedding);
    }
    let units = &normalized[..];

    // Group the units by the front end's cluster id.
    let mut by_cluster: Vec<(usize, Vec<usize>)> = Vec::new();
    for (i, u) in units.iter().enumerate() {
        match by_cluster.iter_mut().find(|(c, _)| *c == u.cluster) {
            Some((_, m)) => m.push(i),
            None => by_cluster.push((u.cluster, vec![i])),
        }
    }
    by_cluster.sort_by_key(|(c, _)| *c);
    let clusters_before = by_cluster.len();

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

    let total: f32 = live.iter().map(|c| c.seconds).sum::<f32>().max(1e-6);
    let mut merges = 0usize;

    // Agglomerative merge: closest centroid pair first, stop at the threshold.
    while live.len() > 1 {
        let mut best = (0usize, 0usize, f32::INFINITY);
        for i in 0..live.len() {
            for j in (i + 1)..live.len() {
                let d = cosine_distance(&live[i].centroid, &live[j].centroid);
                if d < best.2 {
                    best = (i, j, d);
                }
            }
        }
        let (i, j, d) = best;
        if d >= opts.merge_distance {
            break;
        }
        if opts.protect_share > 0.0 {
            let threshold = opts.protect_share * total;
            if live[i].seconds >= threshold && live[j].seconds >= threshold {
                break;
            }
        }
        let (keep, gone) = (i.min(j), i.max(j));
        let moved = live.remove(gone);
        live[keep].members.extend(moved.members);
        live[keep].members.sort_unstable();
        live[keep].seconds += moved.seconds;
        live[keep].centroid = centroid(units, &live[keep].members);
        merges += 1;
    }

    // Dissolve clusters that cannot stand for a speaker on their own: too small
    // a share of the speech, or so incoherent that most of their own units are
    // closer to another cluster's centroid (a mixed bag of a split). Each of
    // their units goes to the centroid it is closest to in embedding space.
    // Never by time, and the largest cluster is always kept, so the step can
    // never produce zero speakers.
    let mut dissolved = 0usize;
    let mut reassigned = 0usize;
    let mut labels = label_sequence(units, &live);
    if live.len() > 1 && (opts.min_share > 0.0 || opts.min_loyalty > 0.0) {
        let largest = live
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.seconds.partial_cmp(&b.1.seconds).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let threshold = opts.min_share * total;
        let drop: Vec<usize> = (0..live.len())
            .filter(|&i| {
                if i == largest {
                    return false;
                }
                if live[i].seconds < threshold {
                    return true;
                }
                if opts.min_loyalty > 0.0 {
                    let own = live[i]
                        .members
                        .iter()
                        .filter(|&&m| nearest_cluster(units, &live, m) == i)
                        .count();
                    (own as f32 / live[i].members.len().max(1) as f32) < opts.min_loyalty
                } else {
                    false
                }
            })
            .collect();
        if !drop.is_empty() {
            dissolved = drop.len();
            let keep: Vec<usize> = (0..live.len()).filter(|i| !drop.contains(i)).collect();
            let mut out = vec![0usize; units.len()];
            for (label, &k) in keep.iter().enumerate() {
                for &m in &live[k].members {
                    out[m] = label;
                }
            }
            for &k in &drop {
                for &m in &live[k].members {
                    let mut best = (0usize, f32::INFINITY);
                    for (label, &other) in keep.iter().enumerate() {
                        let d = cosine_distance(&units[m].embedding, &live[other].centroid);
                        if d < best.1 {
                            best = (label, d);
                        }
                    }
                    out[m] = best.0;
                    reassigned += 1;
                }
            }
            labels = out;
        }
    }

    // Renumber by speech time, largest speaker first, so the output is stable.
    let mut order: Vec<(usize, f32)> = speech_shares(units, &labels);
    order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    let mut remap: Vec<usize> = vec![usize::MAX; units.len()];
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
    }
}

fn nearest_cluster(units: &[Unit], live: &[Cluster], m: usize) -> usize {
    let mut best = (0usize, f32::INFINITY);
    for (i, c) in live.iter().enumerate() {
        let d = cosine_distance(&units[m].embedding, &c.centroid);
        if d < best.1 {
            best = (i, d);
        }
    }
    best.0
}

fn label_sequence(units: &[Unit], live: &[Cluster]) -> Vec<usize> {
    let mut out = vec![0usize; units.len()];
    for (label, c) in live.iter().enumerate() {
        for &m in &c.members {
            out[m] = label;
        }
    }
    out
}

/// Total speech seconds per final label, sorted largest first, for reporting.
pub fn speech_shares(units: &[Unit], labels: &[usize]) -> Vec<(usize, f32)> {
    let mut totals: Vec<(usize, f32)> = Vec::new();
    for (u, &l) in units.iter().zip(labels) {
        match totals.iter_mut().find(|(k, _)| *k == l) {
            Some((_, t)) => *t += u.seconds(),
            None => totals.push((l, u.seconds())),
        }
    }
    totals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    totals
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(start: f32, end: f32, cluster: usize, emb: Vec<f32>) -> Unit {
        Unit {
            start,
            end,
            cluster,
            embedding: emb,
        }
    }

    #[test]
    fn merges_identical_centroids() {
        // Two clusters with the same direction must collapse into one.
        let units = vec![
            unit(0.0, 5.0, 0, vec![1.0, 0.0, 0.0]),
            unit(5.0, 10.0, 1, vec![1.0, 0.01, 0.0]),
        ];
        let out = recluster(
            &units,
            &Options {
                merge_distance: 0.3,
                ..Options::default()
            },
        );
        assert_eq!(out.clusters_after, 1);
        assert_eq!(out.merges, 1);
        assert_eq!(out.labels, vec![0, 0]);
    }

    #[test]
    fn keeps_distinct_speakers() {
        let units = vec![
            unit(0.0, 5.0, 0, vec![1.0, 0.0, 0.0]),
            unit(5.0, 10.0, 1, vec![0.0, 1.0, 0.0]),
        ];
        let out = recluster(
            &units,
            &Options {
                merge_distance: 0.3,
                ..Options::default()
            },
        );
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
                min_loyalty: 0.0,
                ..Options::default()
            },
        );
        assert_eq!(out.dissolved, 1);
        assert_eq!(out.reassigned, 1);
        // Units 0 and 2 are cluster 0, unit 3 is cluster 1, unit 1 joined it.
        assert_eq!(out.labels[1], out.labels[3]);
        assert_ne!(out.labels[1], out.labels[0]);
    }

    #[test]
    fn never_merges_two_large_clusters_when_protected() {
        let units = vec![
            unit(0.0, 10.0, 0, vec![1.0, 0.0]),
            unit(10.0, 20.0, 1, vec![0.95, 0.31]),
        ];
        let out = recluster(
            &units,
            &Options {
                merge_distance: 0.9,
                protect_share: 0.3,
                ..Options::default()
            },
        );
        assert_eq!(out.clusters_after, 2);
        assert_eq!(out.merges, 0);
    }

    #[test]
    fn empty_input_is_empty_output() {
        let out = recluster(&[], &Options::default());
        assert_eq!(out.clusters_after, 0);
        assert!(out.labels.is_empty());
    }
}

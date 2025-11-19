use std::collections::{HashSet};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use pyo3::{prelude::*};
use rayon::prelude::*;
use rand::{Rng};
use rand_chacha::{ChaCha20Rng, ChaChaRng, rand_core::{RngCore, SeedableRng}};
use std::collections::HashMap;
use rand::prelude::IndexedRandom;

struct Edge {
    dist: f64,
    i: usize,
    j: usize,
}

// Max-heap by distance
impl Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Edge { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }
impl PartialEq for Edge { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
impl Eq for Edge {}


// Get the reconstructed edges for a given set embeddings and k nearest neighbors
// Parameters:
// embeddings - 2D numpy array of shape (num_nodes, embedding_dim)
// k - number of nearest neighbors to consider for each node
// Returns:
// A vector of tuples representing the reconstructed edges (node_i, node_j)
#[pyfunction]
fn get_reconstructed_edges(
    py: Python<'_>,
    embeddings: PyReadonlyArray2<f64>, 
    k: usize) 
-> PyResult<Vec<(usize, usize)>> {
    let num_nodes = embeddings.shape()[0];
    let dim = embeddings.shape()[1];
    let mut neighbors: Vec<(usize, usize)> = Vec::new();

    println!("Acquiring");
    let emb = embeddings.as_array();
    py.allow_threads(|| {
        let cpu_num = num_cpus::get();
        println!("Using {} threads", cpu_num);
        
        // parallel computing 
        // each thread processes a chunk of nodes
        let chunk_size = (num_nodes + cpu_num - 1) / cpu_num;
        let local_heaps = (0..cpu_num).into_par_iter().map(|thread_id| {
            let start = thread_id * chunk_size;
            let end = ((thread_id + 1) * chunk_size).min(num_nodes);
            let mut heap = BinaryHeap::with_capacity(k);
            for i in start..end {
                for j in i+1..num_nodes {
                    let mut dist = 0.0;
                    for d in 0..dim {
                        let diff = &emb[[i, d]] - &emb[[j, d]];
                        dist += &diff * &diff;
                    }

                    if heap.len() < k {
                        heap.push(Edge { dist, i, j });
                    } else if let Some(top) = heap.peek() {
                        if dist < top.dist {
                            heap.pop();
                            heap.push(Edge { dist, i, j });
                        }
                    }
                }
            }
            heap
        }).collect::<Vec<_>>();
        // Merge heaps from all threads
        let mut heap = BinaryHeap::with_capacity(k);
        for local_heap in local_heaps {
            for edge in local_heap.into_sorted_vec() {
                if heap.len() < k {
                    heap.push(edge);
                } else if let Some(top) = heap.peek() {
                    if edge.dist < top.dist {
                        heap.pop();
                        heap.push(edge);
                    }
                }
            }
        }

        neighbors = heap.into_sorted_vec().into_iter().map(|e| {
            if e.i < e.j { (e.i, e.j) } else { (e.j, e.i) }
        }).collect();
    });
    Ok(neighbors)
}

#[pyfunction]
fn get_next(r: f64, prev_neighbors: Vec<usize>, neighbors: Vec<usize>, current: usize, p: f64, q: f64, prev: Option<usize>) -> PyResult<usize> {
    if neighbors.len() == 0 {
        return Ok(current);
    }
    let mut weights = vec![0.0; neighbors.len()];
    let prev_neighbors: HashSet<usize>  = prev_neighbors.into_iter().collect();
    for (i, neighbor) in neighbors.iter().enumerate() {

        let weight = 1.0;
        if Some(*neighbor) == prev {
            weights[i] = weight / p;
        } else if prev.is_none() && prev_neighbors.contains(neighbor) {
            weights[i] = weight;
        } else {
            weights[i] = weight / q;
        }
    }
    let sum: f64 = weights.iter().sum();
    let weights: Vec<f64> = weights.iter().map(|x| x / sum).collect();
    let weights_cumsum: Vec<f64> = weights.iter().scan(0.0, |acc, x| {
        *acc += x;
        Some(*acc)
    }).collect();
    let index = weights_cumsum.iter().position(|&x| x > r).unwrap_or(weights_cumsum.len()-1);
    Ok(neighbors[index])

}


fn skipgrams(
    sequence: Vec<usize>,
    vocabulary_size: usize,
    window_size: usize,
) -> (Vec<(usize, usize)>, Vec<i32>) {
    // Positive couples/labels
    let mut couples: Vec<(usize, usize)> = Vec::new();
    // We'll store labels in a neutral enum, then convert to a Python list at the end
    let mut labels: Vec<i32> = Vec::new();

    let n = sequence.len();

    for (i, &wi) in sequence.iter().enumerate() {
        let start = i.saturating_sub(window_size);
        let end = (i + window_size + 1).min(n);
        for j in start..end {
            if j == i { continue; }
            let wj = sequence[j];
            couples.push((wi, wj));
            labels.push(1);
        }
    }

    (couples, labels)
}

fn log_uniform_candidate_sampler(
    true_class: usize,
    num_sampled: usize,
    range_max: usize,
    unique: bool,
) -> Vec<usize> {
    if num_sampled == 0 || range_max == 0 {
        return Vec::new();
    }

    let max_possible = if range_max > 0 {
        // exclude true_class if within range
        if true_class < range_max { range_max - 1 } else { range_max }
    } else { 0 };
    let target = if unique { num_sampled.min(max_possible) } else { num_sampled };

    let mut out = Vec::with_capacity(target);

    // Inverse-CDF sampling for a rough log-uniform over [0, range_max)
    // u ~ U(0, ln(range_max+1)), x = floor(exp(u)) - 1
    // Reject true_class; if unique, also reject duplicates.
    let ln_max = ((range_max as f64) + 1.0).ln();

    let mut guard = 0usize;
    let guard_limit = target * 20 + 100; // prevent pathological loops
    let mut cha_cha_rng = ChaChaRng::from_seed(rand::thread_rng().gen());

    while out.len() < target && guard < guard_limit {
        guard += 1;
        // generate a uniform f64 in [0, ln_max) using RngCore::next_u64()
        // avoid relying on Rng::gen_range which may not be implemented for ChaCha20Rng in some crate combos
        let rand_u = cha_cha_rng.next_u64();
        let fract = (rand_u as f64) / (u64::MAX as f64 + 1.0);
        let u: f64 = fract * ln_max;
        let mut x: usize = u.exp().floor() as usize;
        if x > 0 { x -= 1; }
        if x >= range_max {
            continue;
        }
        if x == true_class {
            continue;
        }
        if unique && out.contains(&x) {
            continue;
        }
        out.push(x);
    }

    out
}


#[pyfunction]
fn generate_skip_grams(
    py: Python<'_>,
    sequences: Vec<Vec<usize>>,   // Python list[list[int]]
    window_size: usize,
    num_ns: usize,
    vocab_size: usize,
) -> PyResult<Option<(Vec<usize>, Vec<Vec<usize>>, Vec<Vec<i64>>)>> {
    let mut targets: Vec<usize> = Vec::new();
    let mut contexts: Vec<Vec<usize>> = Vec::new();
    let mut labels: Vec<Vec<i64>> = Vec::new();

    for sequence in sequences {
        // positives only, no shuffle, no categorical
        let (positive_skip_grams, _py_labels) = skipgrams(
            sequence,
            vocab_size,
            window_size,
        );

        if positive_skip_grams.is_empty() {
            continue;
        }

        for (target, context_word) in positive_skip_grams {
            // follow the same guard logic as your Python version
            let negatives: Vec<usize> =
                if vocab_size < 10 || num_ns > vocab_size || num_ns == 0 {
                    Vec::new()
                } else {
                    log_uniform_candidate_sampler(
                        context_word,      // true_class
                        num_ns,            // number of negatives requested
                        vocab_size,        // [0, vocab_size)
                        true,              // unique
                    )
                };

            // contexts: [positive_context, negatives...]
            let mut ctx_row = Vec::with_capacity(1 + negatives.len());
            ctx_row.push(context_word);
            ctx_row.extend(negatives.iter().copied());

            // labels: [1, 0, 0, ...]
            let mut lbl_row = Vec::with_capacity(ctx_row.len());
            lbl_row.push(1);
            lbl_row.extend(std::iter::repeat(0).take(ctx_row.len() - 1));

            targets.push(target);
            contexts.push(ctx_row);
            labels.push(lbl_row);
        }
    }

    if targets.is_empty() {
        // Return None to Python if no targets (exact parity with your Python function)
        return Ok(None);
    }

    Ok(Some((targets, contexts, labels)))
}



// --------------- DistGER -------------------

/// Equivalent of Python's Walker class
#[derive(Debug, Clone)]
#[pyclass]
pub struct Walker {
    pub id: usize,
    pub step: usize,
    pub Hn: f64,
    pub Sn: f64,
    pub EnH: f64,
    pub EnS: f64,
    pub EnHS: f64,
    pub EnHH: f64,
    pub EnSS: f64,
    pub N: f64,
    pub R: f64,
    pub last_vertex: Option<usize>,
    pub trace_flag: bool,
}

#[pymethods]
impl Walker {
    #[new]
    pub fn new(id: usize) -> Self {
        Walker {
            id,
            step: 0,
            Hn: 0.0,
            Sn: 1.0,
            EnH: 0.0,
            EnS: 1.0,
            EnHS: 0.0,
            EnHH: 0.0,
            EnSS: 1.0,
            N: 1.0,
            R: 1.0,
            last_vertex: None,
            trace_flag: false,
        }
    }
}

/// Equivalent of Python's Graph class (but without networkx).
/// adjacency: vertex -> Vec<(neighbor, common_neighbors)>
#[derive(Debug, Clone)]
#[pyclass]
pub struct Graph {
    pub adjacency: HashMap<usize, Vec<(usize, usize)>>,
    pub out_degree: HashMap<usize, usize>,
}

#[pymethods]
impl Graph {
    #[new]
    pub fn new(adjacency: HashMap<usize, Vec<(usize, usize)>>) -> Self {
        let mut out_degree = HashMap::new();
        for (v, neigh) in &adjacency {
            out_degree.insert(*v, neigh.len());
        }
        Graph { adjacency, out_degree }
    }

    /// Equivalent of random_neighbor(self, vertex)
    /// Returns Some((dst, common_neighbors)) or None
    pub fn random_neighbor(&self, vertex: usize) -> Option<(usize, usize)> {
        let mut rng = rand::thread_rng();
        self.adjacency
            .get(&vertex)
            .and_then(|neighbors| neighbors.choose(&mut rng).copied())
    }

    /// Placeholder for is_remote (always false, as in Python)
    pub fn is_remote(&self, _vertex: usize) -> bool {
        // TODO: define proper logic
        false
    }
}

/// Equivalent of log2_safe in Python
fn log2_safe(x: f64) -> f64 {
    if x > 0.0 {
        x.log2()
    } else {
        f64::NEG_INFINITY
    }
}

/// Equivalent of iterate_stats(walker, fi, step_k=1.0)
pub fn iterate_stats(walker: &mut Walker, fi: f64, step_k: f64) {
    let Sn = walker.Sn;
    let Hn = walker.Hn;

    let Snp1 = Sn + step_k;

    // t = log2_safe((Sn / Snp1)**Sn * (1 / Snp1)**step_k)
    let base = (Sn / Snp1).powf(Sn) * (1.0 / Snp1).powf(step_k);
    let mut t = log2_safe(base);

    if fi > 0.0 {
        // fj = fi + step_k
        let fj = fi + step_k;
        // t = log2_safe((Sn / Snp1)**Sn * (1 / Snp1)**step_k *
        //               (fj / fi)**fi * fj**step_k)
        let base2 = base * (fj / fi).powf(fi) * fj.powf(step_k);
        t = log2_safe(base2);
    }

    // Hnp1 = (Sn * Hn - t) / Snp1
    let Hnp1 = (Sn * Hn - t) / Snp1;

    let Np1 = walker.N + 1.0;
    let EnH = walker.EnH + (Hnp1 - walker.EnH) / Np1;
    let EnS = walker.EnS + (Snp1 - walker.EnS) / Np1;
    let EnHS = walker.EnHS + (Hnp1 * Snp1 - walker.EnHS) / Np1;
    let EnHH = walker.EnHH + (Hnp1.powi(2) - walker.EnHH) / Np1;
    let EnSS = walker.EnSS + (Snp1.powi(2) - walker.EnSS) / Np1;

    let D_H = EnHH - EnH.powi(2);
    let D_S = EnSS - EnS.powi(2);

    let R = if D_H > 0.0 && D_S > 0.0 {
        (EnHS - EnH * EnS) / (D_H * D_S).sqrt()
    } else {
        0.0
    };

    // Update walker
    walker.Hn = Hnp1;
    walker.Sn = Snp1;
    walker.EnH = EnH;
    walker.EnS = EnS;
    walker.EnHS = EnHS;
    walker.EnHH = EnHH;
    walker.EnSS = EnSS;
    walker.N = Np1;
    walker.R = R;
    walker.step += 1;
}

#[pyfunction]
pub fn walk(
    graph: &Graph,
    walker: &mut Walker,
    threshold: f64,
    min_length: usize,
    max_length: usize,
) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let mut walker_to_path_count: HashMap<usize, usize> = HashMap::new();

    let mut path = Vec::new();

    let mut current_v = walker.last_vertex.unwrap_or(walker.id);
    walker.step = 1;

    loop {
        // yield current_v
        path.push(current_v);

        if walker.step == max_length {
            break;
        }

        let fi_u = walker_to_path_count.get(&current_v).copied().unwrap_or(0);
        walker_to_path_count.insert(current_v, fi_u + 1);

        if walker.step > 0 {
            let fi = fi_u as f64;
            iterate_stats(walker, fi, 1.0);
        }

        if walker.step > min_length
            && (walker.R.powi(2) < threshold || walker.R < 0.0)
        {
            break;
        }

        let degree = *graph.out_degree.get(&current_v).unwrap_or(&0);
        if degree == 0 {
            break;
        }

        let candidate = graph.random_neighbor(current_v);
        let (dst, common_neighbors) = match candidate {
            Some(c) => c,
            None => break,
        };

        let src_deg = *graph.out_degree.get(&current_v).unwrap_or(&1);
        let dst_deg = *graph.out_degree.get(&dst).unwrap_or(&1);

        let p_accept = if src_deg == 0 || dst_deg == 0 {
            0.0
        } else if src_deg == common_neighbors {
            0.0
        } else {
            // (1 / (src_deg - common_neighbors)) * max(src_deg, dst_deg) / min(src_deg, dst_deg)
            let src_deg_f = src_deg as f64;
            let dst_deg_f = dst_deg as f64;
            let common_f = common_neighbors as f64;
            let denom = src_deg_f - common_f;
            if denom == 0.0 {
                0.0
            } else {
                (1.0 / denom) * src_deg_f.max(dst_deg_f) / src_deg_f.min(dst_deg_f)
            }
        };

        let p_norm = (p_accept as f64).tanh();
        let p: f64 = rng.gen();

        if p_norm < p && walker.step > min_length {
            walker.trace_flag = true;
            walker.last_vertex = Some(current_v);
            break; // emit trace
        }

        if graph.is_remote(dst) {
            break; // emit remote walker
        }

        current_v = dst; // continue walking locally
    }

    path
}

/// A Python module implemented in Rust.
#[pymodule]
fn vertex_voyage_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_next, m)?)?;
    m.add_function(wrap_pyfunction!(generate_skip_grams, m)?)?;
    m.add_function(wrap_pyfunction!(get_reconstructed_edges, m)?)?;
    m.add_class::<Walker>()?;
    m.add_class::<Graph>()?;
    m.add_function(wrap_pyfunction!(walk, m)?)?;
    Ok(())
}

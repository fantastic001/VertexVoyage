use std::collections::{HashSet};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use pyo3::{prelude::*};
use rayon::prelude::*;

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







/// A Python module implemented in Rust.
#[pymodule]
fn vertex_voyage_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_next, m)?)?;
    m.add_function(wrap_pyfunction!(get_reconstructed_edges, m)?)?;
    Ok(())
}

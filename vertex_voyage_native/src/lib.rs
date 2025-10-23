use std::collections::{HashSet};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};

use pyo3::{prelude::*};

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
    let mut neighbors = Vec::new();
    let num_nodes = embeddings.shape()[0];
    let mut edge_distances: Vec<(usize, usize, f64)> = Vec::new();
    edge_distances.reserve(100000);
    // Compute pairwise distances
    for i in 0..num_nodes {
        for j in i+1..num_nodes {
        
            let dist = embeddings.as_array().row(i).to_owned() - embeddings.as_array().row(j).to_owned();
            let dist = dist.dot(&dist).sqrt();
            edge_distances.push((i, j, dist));
            
        }
    }
    edge_distances.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    for i in 0..k {
        neighbors.push((edge_distances[i].0, edge_distances[i].1));
    }
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

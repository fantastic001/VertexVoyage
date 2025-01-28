use std::collections::HashSet;

use pyo3::{prelude::*, types::PyList};
use rand::random_range;


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
    Ok(())
}

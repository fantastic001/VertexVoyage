use std::collections::{HashMap, HashSet, LinkedList};

use rand::{rngs::{StdRng, ThreadRng}, thread_rng, Rng, SeedableRng};

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






// LFM

/// Example placeholder function for `cfg.get_config_int(...)`.
/// In real code, replace this with your actual config-retrieval logic.
fn cfg_get_config_int(key: &str, default: i32, _description: &str) -> i32 {
    // For example, always return the default:
    default
}

/// Example trait for a graph structure that you can customize
/// to your real use case. It must provide neighbors of a node,
/// and also a list of node IDs.
pub trait Graph {
    fn neighbors(&self, node: usize) -> Vec<usize>;
    fn nodes(&self) -> Vec<usize>;
}

/// The Community struct from the previous translation:
pub struct Community<'a, G: Graph> {
    graph: &'a G,
    alpha: f64,
    nodes: HashSet<usize>,
    k_in: usize,
    k_out: usize,
}

impl<'a, G: Graph> Community<'a, G> {
    pub fn new(graph: &'a G, alpha: f64) -> Self {
        Community {
            graph,
            alpha,
            nodes: HashSet::new(),
            k_in: 0,
            k_out: 0,
        }
    }

    pub fn add_node(&mut self, node: usize) {
        let neighbors: HashSet<usize> = self.graph.neighbors(node).into_iter().collect();
        let node_k_in = neighbors.intersection(&self.nodes).count();
        let node_k_out = neighbors.len() - node_k_in;

        self.nodes.insert(node);

        self.k_in += 2 * node_k_in;
        self.k_out = self.k_out + node_k_out - node_k_in;
    }

    pub fn remove_node(&mut self, node: usize) {
        let neighbors: HashSet<usize> = self.graph.neighbors(node).into_iter().collect();
        let node_k_in = neighbors.intersection(&self.nodes).count();
        let node_k_out = neighbors.len() - node_k_in;

        self.nodes.remove(&node);

        self.k_in -= 2 * node_k_in;
        self.k_out = self.k_out - node_k_out + node_k_in;
    }

    pub fn cal_add_fitness(&self, node: usize) -> f64 {
        let neighbors: HashSet<usize> = self.graph.neighbors(node).into_iter().collect();
        let old_k_in = self.k_in;
        let old_k_out = self.k_out;
        let vertex_k_in = neighbors.intersection(&self.nodes).count();
        let vertex_k_out = neighbors.len() - vertex_k_in;

        let new_k_in = old_k_in + 2 * vertex_k_in;
        let new_k_out = old_k_out + vertex_k_out - vertex_k_in;

        let new_fitness = (new_k_in as f64)
            / ((new_k_in + new_k_out) as f64).powf(self.alpha);
        let old_fitness = (old_k_in as f64)
            / ((old_k_in + old_k_out) as f64).powf(self.alpha);

        new_fitness - old_fitness
    }

    pub fn cal_remove_fitness(&self, node: usize) -> f64 {
        let neighbors: HashSet<usize> = self.graph.neighbors(node).into_iter().collect();

        let new_k_in = self.k_in;
        let new_k_out = self.k_out;

        let node_k_in = neighbors.intersection(&self.nodes).count();
        let node_k_out = neighbors.len() - node_k_in;

        let old_k_in = new_k_in - 2 * node_k_in;
        let old_k_out = new_k_out - node_k_out + node_k_in;

        let old_fitness = (old_k_in as f64)
            / ((old_k_in + old_k_out) as f64).powf(self.alpha);
        let new_fitness = (new_k_in as f64)
            / ((new_k_in + new_k_out) as f64).powf(self.alpha);

        new_fitness - old_fitness
    }

    pub fn recalculate(&self) -> Option<usize> {
        for &vid in &self.nodes {
            let fitness_diff = self.cal_remove_fitness(vid);
            if fitness_diff < 0.0 {
                return Some(vid);
            }
        }
        None
    }

    pub fn get_neighbors(&self) -> HashSet<usize> {
        let mut neighbors = HashSet::new();
        for &node in &self.nodes {
            for neigh in self.graph.neighbors(node) {
                if !self.nodes.contains(&neigh) {
                    neighbors.insert(neigh);
                }
            }
        }
        neighbors
    }

    pub fn get_fitness(&self) -> f64 {
        let k_in = self.k_in as f64;
        let k_out = self.k_out as f64;
        k_in / (k_in + k_out).powf(self.alpha)
    }
}

struct PyGraph <'a> {
    pyobj: &'a Bound<'a, PyAny>,
}

impl<'a> Graph for PyGraph<'a> {
    fn neighbors(&self, node: usize) -> Vec<usize> {
        let myresult = self.pyobj.call_method1("neighbors", (node,));
        match myresult {
            Err(e) => {
                println!("Error: {:?}", e);
                Vec::new()
            },
            Ok(result) => {
                match result.extract() {
                    Err(e) => {
                        println!("Error: {:?}", e);
                        Vec::new()
                    },
                    Ok(neighbors) => {
                        neighbors
                    }
                }
            }
        }
    }

    fn nodes(&self) -> Vec<usize> {
        match self.pyobj.call_method0("nodes") {
            Err(e) => {
                println!("Error: {:?}", e);
                Vec::new()
            },
            Ok(result) => {
                match result.extract() {
                    Err(e) => {
                        println!("Error: {:?}", e);
                        Vec::new()
                    },
                    Ok(nodes) => {
                        nodes
                    }
                }
            }
        }
    }
}

/// Translates the `modified__lfm` function from Python into Rust.
/// Returns a vector of communities, each being a vector of node IDs.
pub fn modified_lfm<G: Graph>(
    graph: &G,
    partition_count: usize,
    alpha: f64,
    threshold: f64,
    pm_k: usize,
    seed: Option<u64>,
) -> Vec<Vec<usize>> {
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or_else(|| random_range(0..u64::MAX)));

    // Stores the final list of communities
    let mut communities: Vec<Vec<usize>> = Vec::new();

    // All nodes initially not in any community
    let mut node_not_include = graph.nodes();
    let node_num = node_not_include.len();

    // Main loop
    while (node_not_include.len() as f64) > (node_num as f64 * threshold)
        && (communities.len() < (pm_k as usize * partition_count))
    {
        // Create a new community
        let mut c = Community::new(graph, alpha);

        // Select a random seed node
        let seed_idx = rng.gen_range(0..node_not_include.len());
        let seed_node = node_not_include[seed_idx];
        c.add_node(seed_node);

        // Set of nodes to examine for potential inclusion
        let mut to_be_examined = c.get_neighbors();

        // Expand community
        while !to_be_examined.is_empty() {
            // Compute fitness gain for each candidate
            let mut candidate_map: HashMap<usize, f64> = HashMap::new();
            for &node in &to_be_examined {
                let fitness_gain = c.cal_add_fitness(node);
                candidate_map.insert(node, fitness_gain);
            }

            // Find the node with the largest fitness gain
            let to_be_add_option = candidate_map
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // If we cannot find any node (unlikely) or max fitness < 0.0, we stop
            let (best_node, best_fitness) = match to_be_add_option {
                Some((n, f)) if f >= 0.0 => (n, f),
                _ => {
                    // max fitness < 0 or no candidates, break expansion
                    break;
                }
            };

            // Add the best node
            c.add_node(best_node);

            // Recalculate: remove any nodes causing negative improvement
            let mut to_be_removed = c.recalculate();
            while let Some(node_to_remove) = to_be_removed {
                c.remove_node(node_to_remove);
                to_be_removed = c.recalculate();
            }

            // Update the next set of candidates
            to_be_examined = c.get_neighbors();
        }

        // Remove the new community’s nodes from the node_not_include list
        for &node in &c.nodes {
            if let Some(pos) = node_not_include.iter().position(|x| *x == node) {
                node_not_include.remove(pos);
            }
        }

        // Add this community to the list
        communities.push(c.nodes.iter().cloned().collect());
    }

    // If we have fewer than partition_count communities, pad with empty ones
    if communities.len() < partition_count {
        for _ in 0..(partition_count - communities.len()) {
            communities.push(Vec::new());
        }
    }

    // Distribute any leftover nodes that were not included
    for node in node_not_include {
        let idx = rng.gen_range(0..communities.len());
        communities[idx].push(node);
    }

    communities
}


#[pyfunction]
fn lfm<'a>(py: Python<'a>, graph: Bound<'a, PyAny>, partition_count: usize, alpha: f64, threshold: f64, pm_k: usize, seed: Option<u64>) -> PyResult<Bound<'a, PyList>> {
    let graph = PyGraph { pyobj: &graph };

    let communities = modified_lfm(&graph, partition_count, alpha, threshold, pm_k, seed);

    let py_communities = PyList::empty(py);
    for community in communities {
        let py_community = PyList::empty(py);
        for node in community {
            py_community.append(node)?;
        }
        py_communities.append(py_community)?;
    }

    Ok(py_communities)
}

/// A Python module implemented in Rust.
#[pymodule]
fn vertex_voyage_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_next, m)?)?;
    m.add_function(wrap_pyfunction!(lfm, m)?)?;
    Ok(())
}

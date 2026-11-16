# Synthetic check

[Stochastic block model](../../../doc/glossary.md#stochastic-block-model) with $8 \times 200$ nodes, $p_{\text{in}} = 0.10$, $p_{\text{out}} = 0.003$, $P = 2$, $d = 32$, $h = 64$, node2vec with $p = q = 1$ per partition, unit-normalised inputs and outputs, $m = 1$, $\alpha = 0.5$, 3 seeds.

- `leak`: probability that a node's home partition is the wrong one. `leak = 0` cuts only between communities; `leak = 0.5` is random partitioning.
- Replication: a node with a neighbour in the other partition is copied there with probability $0.3$ ($RF = 2$, about $30\%$ of the nodes replicated, $\approx 490$ anchors).
- $20\%$ of all edges are held out. Cross AUC is on held-out cut edges, within AUC on held-out covered edges; a pair is scored by negative distance between merged vectors $w_u$. Replica cosine is the mean cosine between the two aligned replicas of a node.
- The oracle embeds the unpartitioned training graph.
- Script: listed in full under [Script](#script) at the end of this section.

**Without replication ($RF = 1$)**, cross AUC (within AUC):

| Method | leak 0, $\delta = 0.10$ | leak 0.15, $\delta = 0.30$ | leak 0.5, $\delta = 0.50$ |
|---|---|---|---|
| Oracle (no partitioning) | 0.50 (0.83) | 0.80 (0.85) | 0.84 (0.85) |
| No alignment | 0.52 (0.83) | 0.51 (0.83) | 0.49 (0.83) |
| Procrustes on cut edges | 0.51 (0.83) | 0.73 (0.83) | 0.83 (0.83) |
| Residual MLP, contrastive (this work) | 0.53 (0.85) | 0.81 (0.86) | 0.85 (0.86) |
| Residual MLP, triplet (this work) | 0.54 (0.84) | 0.81 (0.86) | 0.85 (0.86) |

**With replication ($RF = 2$)**, cross AUC (within AUC) [replica cosine]:

| Method | leak 0, $\delta = 0.05$ | leak 0.15, $\delta = 0.14$ | leak 0.5, $\delta = 0.24$ |
|---|---|---|---|
| Oracle (no partitioning) | 0.51 (0.85) | 0.80 (0.85) | 0.83 (0.85) |
| No alignment | 0.47 (0.79) [-0.02] | 0.45 (0.78) [-0.04] | 0.49 (0.78) [0.01] |
| Procrustes on cut edges | 0.50 (0.79) [0.22] | 0.78 (0.84) [0.52] | 0.83 (0.85) [0.53] |
| Procrustes on replicas | 0.52 (0.84) [0.69] | 0.78 (0.85) [0.70] | 0.82 (0.85) [0.70] |
| Residual MLP, contrastive (this work) | 0.50 (0.83) [0.41] | 0.80 (0.86) [0.82] | 0.85 (0.86) [0.93] |
| Residual MLP, contrastive, plus replica term | 0.53 (0.86) [0.79] | 0.81 (0.86) [0.91] | 0.85 (0.86) [0.96] |
| Residual MLP, triplet (this work) | 0.50 (0.84) [-0.16] | 0.80 (0.86) [0.19] | 0.84 (0.86) [0.34] |
| Residual MLP, triplet, plus replica term | 0.51 (0.85) [0.44] | 0.80 (0.86) [0.51] | 0.84 (0.86) [0.57] |

Standard deviations are at most $0.04$ on AUC. Two seeds on the same unpartitioned graph differ by relative squared error $1.83$ (identity), $0.36$ (best rotation), $0.25$ (best linear map).

Reading:

- Unaligned cross-partition prediction is at chance everywhere. With replication, no alignment also lowers within AUC from $0.83$ to $0.78$, because unaligned replicas are averaged (Theorem 9).
- The approach of E10 works: with unit normalisation both losses reach the oracle on cut edges, and contrastive and triplet are tied.
- With replicas, a single rotation fitted on replica pairs is nearly as good ($0.78$ against $0.80$ at leak $0.15$; $0.82$ against $0.85$ at leak $0.5$) and restores within AUC to the oracle. It has $d(d-1)/2 = 496$ parameters, a closed form, and needs no negatives. Without replicas it is clearly worse at leak $0.15$ ($0.73$).
- Replica cosine after the best rotation is $0.70$: two partitions disagree about the same node beyond any rotation. This is the non-rigid part of Theorem 1, and it is where the MLP gains.
- The triplet loss leaves replicas far apart (cosine $0.19$ to $0.34$) at the same AUC, as expected from a loss that only orders distances. The contrastive loss pulls them together even without the replica term ($0.82$ to $0.93$), and the replica term raises that further at no AUC cost.
- The MLP's within AUC exceeds the oracle's. It is trained on edges and evaluated on edges, so it acts as a learned link-prediction encoder, not as a change of coordinates.
- At leak $0$ the oracle is at chance on cut edges: they are noise and nothing aligns them. Replica alignment still matters there, for within AUC ($0.79 \to 0.84$).

Limits: one synthetic family with well-separated communities, the easy case for an MLP; $P = 2$; small $n$; a replication rule that may differ from the VertexVoyage one; gensim printed `Exception ignored in 'our_dot_float'` warnings in this environment. Treat the numbers as a sanity check, not as a result.

## Script

The script below produced every number of this section. Save it as `replica_alignment_experiment.py` and run it with the VertexVoyage interpreter (Python 3.9 with numpy, networkx, gensim, torch, scikit-learn):

```sh
python replica_alignment_experiment.py > report.json
```

A full run takes about 10 minutes on a laptop and prints one JSON report with mean ± standard deviation over the seeds.

How it maps to the note:

| Code | Note |
|---|---|
| `member[p, u]` | $p \in \Pi(u)$ |
| `raw[p, u]` | $z_{u,p}$, unit-normalised |
| `links` with kinds `intra`, `cut`, `replica`; a row is $(p, u, q, v)$ | $E^+_{\text{intra}}$, $E^+_{\text{cut}}$, $E^+_{\text{rep}}$ |
| `ResidualAligner` | $W(u, p) = N(z_{u,p} + B \, \text{ReLU}(A_p z_{u,p} + a_p) + b)$ |
| `contrastive_loss`, `triplet_loss` | $L_m$ and $L_\alpha$ with sampled negatives, averaged instead of summed |
| `rotate_replicas` | Theorem 8: one Procrustes rotation per partition, partition $0$ fixed as reference |
| `merge_replicas` | Theorem 9: $w_u = N(\sum_p W(u, p))$ |
| `replica_cosine` | mean cosine over $E^+_{\text{rep}}$ after alignment |
| `seed_misalignment` | the $1.83$ / $0.36$ / $0.25$ relative errors (Theorem 2) |
| `ALIGNERS` | the rows of the two tables; add an entry to test a new method |

Every constant of `Config` can be overridden with an environment variable prefixed `ALIGN_`, for example `ALIGN_LEAKS=0.15 ALIGN_REPLICATIONS=0.3 ALIGN_SEEDS=0,1,2,3,4`.

```python
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score

LOGGER = logging.getLogger("replica_alignment_experiment")

Pairs = np.ndarray
Links = np.ndarray
Replicas = np.ndarray
Aligner = Callable[["Problem"], Replicas]

INTRA = "intra"
CUT = "cut"
REPLICA = "replica"
NORM_FLOOR = 1e-12


def read_env(name: str, default: str) -> str:
    return os.environ.get(f"ALIGN_{name}", default)


def read_floats(name: str, default: str) -> Tuple[float, ...]:
    return tuple(float(s) for s in read_env(name, default).split(","))


@dataclass(frozen=True)
class Config:
    communities: int = int(read_env("COMMUNITIES", "8"))
    community_size: int = int(read_env("COMMUNITY_SIZE", "200"))
    p_in: float = float(read_env("P_IN", "0.10"))
    p_out: float = float(read_env("P_OUT", "0.003"))
    partitions: int = int(read_env("PARTITIONS", "2"))
    dimension: int = int(read_env("DIMENSION", "32"))
    walks_per_node: int = int(read_env("WALKS_PER_NODE", "10"))
    walk_length: int = int(read_env("WALK_LENGTH", "40"))
    window: int = int(read_env("WINDOW", "5"))
    sgns_epochs: int = int(read_env("SGNS_EPOCHS", "5"))
    test_fraction: float = float(read_env("TEST_FRACTION", "0.2"))
    hidden_width: int = int(read_env("HIDDEN_WIDTH", "64"))
    train_steps: int = int(read_env("TRAIN_STEPS", "400"))
    learning_rate: float = float(read_env("LEARNING_RATE", "0.01"))
    contrastive_margin: float = float(read_env("CONTRASTIVE_MARGIN", "1.0"))
    triplet_margin: float = float(read_env("TRIPLET_MARGIN", "0.5"))
    procrustes_sweeps: int = int(read_env("PROCRUSTES_SWEEPS", "10"))
    seeds: Tuple[int, ...] = tuple(
        int(s) for s in read_env("SEEDS", "0,1,2").split(",")
    )
    leaks: Tuple[float, ...] = read_floats("LEAKS", "0.0,0.15,0.5")
    replications: Tuple[float, ...] = read_floats("REPLICATIONS", "0.0,0.3")


@dataclass(frozen=True)
class Problem:
    config: Config
    member: np.ndarray
    raw: Replicas
    links: Dict[str, Links]
    train_edges: Pairs
    test_edges: Pairs


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, NORM_FLOOR)


def sample_graph(config: Config, rng: np.random.Generator) -> nx.Graph:
    sizes = [config.community_size] * config.communities
    k = config.communities
    probabilities = np.full((k, k), config.p_out)
    np.fill_diagonal(probabilities, config.p_in)
    seed = int(rng.integers(2**31))
    return nx.stochastic_block_model(sizes, probabilities.tolist(), seed=seed)


def assign_home(
    config: Config, leak: float, rng: np.random.Generator
) -> np.ndarray:
    nodes = config.communities * config.community_size
    community = np.arange(nodes) // config.community_size
    home = community % config.partitions
    shift = rng.integers(1, config.partitions, size=nodes)
    leaked = rng.random(nodes) < leak
    return np.where(leaked, (home + shift) % config.partitions, home)


def replicate_boundary(
    home: np.ndarray,
    edges: Pairs,
    config: Config,
    replication: float,
    rng: np.random.Generator,
) -> np.ndarray:
    nodes = len(home)
    member = np.zeros((config.partitions, nodes), dtype=bool)
    member[home, np.arange(nodes)] = True
    both = np.vstack([edges, edges[:, ::-1]])
    foreign = home[both[:, 0]] != home[both[:, 1]]
    chosen = foreign & (rng.random(nodes) < replication)[both[:, 0]]
    member[home[both[chosen, 1]], both[chosen, 0]] = True
    return member


def split_pairs(
    pairs: Pairs, fraction: float, rng: np.random.Generator
) -> Tuple[Pairs, Pairs]:
    order = rng.permutation(len(pairs))
    cut = int(len(pairs) * fraction)
    return pairs[order[cut:]], pairs[order[:cut]]


def single_walk(
    neighbours: Dict[int, List[int]],
    start: int,
    config: Config,
    rng: np.random.Generator,
) -> List[str]:
    walk = [start]
    for _ in range(config.walk_length - 1):
        options = neighbours[walk[-1]]
        walk.append(options[int(rng.integers(len(options)))])
    return [str(v) for v in walk]


def random_walks(
    graph: nx.Graph, config: Config, rng: np.random.Generator
) -> List[List[str]]:
    neighbours = {v: list(graph.neighbors(v)) for v in graph.nodes}
    starts = [v for v in graph.nodes if neighbours[v]]
    return [
        single_walk(neighbours, start, config, rng)
        for _ in range(config.walks_per_node)
        for start in starts
    ]


def embed_graph(
    graph: nx.Graph, nodes: int, config: Config, rng: np.random.Generator
) -> np.ndarray:
    model = Word2Vec(
        random_walks(graph, config, rng),
        vector_size=config.dimension,
        window=config.window,
        sg=1,
        negative=5,
        min_count=1,
        epochs=config.sgns_epochs,
        workers=1,
        seed=int(rng.integers(2**31)),
    )
    vectors = np.zeros((nodes, config.dimension))
    for key in model.wv.index_to_key:
        vectors[int(key)] = model.wv[key]
    return unit_rows(vectors)


def graph_of(nodes: List[int], edges: Pairs) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges.tolist())
    return graph


def covered(member: np.ndarray, pairs: Pairs) -> np.ndarray:
    return (member[:, pairs[:, 0]] & member[:, pairs[:, 1]]).any(axis=0)


def embed_replicas(
    member: np.ndarray,
    edges: Pairs,
    config: Config,
    rng: np.random.Generator,
) -> Replicas:
    nodes = member.shape[1]
    raw = np.zeros((config.partitions, nodes, config.dimension))
    for p in range(config.partitions):
        inside = member[p, edges[:, 0]] & member[p, edges[:, 1]]
        graph = graph_of(np.flatnonzero(member[p]).tolist(), edges[inside])
        raw[p] = embed_graph(graph, nodes, config, rng)
    return raw


def intra_links(member: np.ndarray, edges: Pairs) -> Links:
    rows = []
    for p in range(len(member)):
        inside = edges[member[p, edges[:, 0]] & member[p, edges[:, 1]]]
        rows.append(
            np.column_stack(
                [np.full(len(inside), p), inside[:, 0],
                 np.full(len(inside), p), inside[:, 1]]
            )
        )
    return np.vstack(rows)


def cross_links(member: np.ndarray, pairs: Pairs) -> Links:
    rows = [
        (p, u, q, v)
        for u, v in pairs.tolist()
        for p in np.flatnonzero(member[:, u])
        for q in np.flatnonzero(member[:, v])
        if p != q
    ]
    return np.array(rows, dtype=int).reshape(-1, 4)


def build_links(member: np.ndarray, edges: Pairs) -> Dict[str, Links]:
    cut_edges = edges[~covered(member, edges)]
    replicated = np.flatnonzero(member.sum(axis=0) > 1)
    twins = np.column_stack([replicated, replicated])
    replica = cross_links(member, twins)
    return {
        INTRA: intra_links(member, edges),
        CUT: cross_links(member, cut_edges),
        REPLICA: replica[replica[:, 0] < replica[:, 2]],
    }


def build_problem(
    config: Config,
    leak: float,
    replication: float,
    rng: np.random.Generator,
) -> Tuple[Problem, np.ndarray]:
    graph = sample_graph(config, rng)
    edges = np.array(graph.edges)
    home = assign_home(config, leak, rng)
    member = replicate_boundary(home, edges, config, replication, rng)
    train_edges, test_edges = split_pairs(edges, config.test_fraction, rng)
    raw = embed_replicas(member, train_edges, config, rng)
    nodes = member.shape[1]
    oracle = embed_graph(
        graph_of(list(range(nodes)), train_edges), nodes, config, rng
    )
    problem = Problem(
        config,
        member,
        raw,
        build_links(member, train_edges),
        train_edges,
        test_edges,
    )
    return problem, oracle


def procrustes_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(source.T @ target)
    return u @ vt


def symmetric(links: Links) -> Links:
    return np.vstack([links, links[:, [2, 3, 0, 1]]])


def rotate_replicas(problem: Problem, kinds: Tuple[str, ...]) -> Replicas:
    links = symmetric(np.vstack([problem.links[kind] for kind in kinds]))
    aligned = problem.raw.copy()
    for _ in range(problem.config.procrustes_sweeps):
        for p in range(1, problem.config.partitions):
            mine = links[links[:, 0] == p]
            if len(mine) == 0:
                continue
            rotation = procrustes_rotation(
                problem.raw[p, mine[:, 1]], aligned[mine[:, 2], mine[:, 3]]
            )
            aligned[p] = problem.raw[p] @ rotation
    return aligned


class ResidualAligner(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.first = torch.nn.ModuleList(
            torch.nn.Linear(config.dimension, config.hidden_width)
            for _ in range(config.partitions)
        )
        self.second = torch.nn.Linear(config.hidden_width, config.dimension)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        moved = [
            z + self.second(torch.relu(layer(z)))
            for z, layer in zip(raw, self.first)
        ]
        return torch.nn.functional.normalize(torch.stack(moved), dim=-1)


def link_distance(out: torch.Tensor, links: torch.Tensor) -> torch.Tensor:
    left = out[links[:, 0], links[:, 1]]
    right = out[links[:, 2], links[:, 3]]
    return (left - right).norm(dim=1)


def random_replicas(slots: torch.Tensor, count: int) -> torch.Tensor:
    return slots[torch.randint(len(slots), (count,))]


def contrastive_loss(
    out: torch.Tensor, links: torch.Tensor, slots: torch.Tensor, config: Config
) -> torch.Tensor:
    negatives = torch.cat(
        [random_replicas(slots, len(links)), random_replicas(slots, len(links))],
        dim=1,
    )
    hinge = torch.clamp(
        config.contrastive_margin - link_distance(out, negatives), min=0
    )
    return (link_distance(out, links) ** 2).mean() + (hinge**2).mean()


def triplet_loss(
    out: torch.Tensor, links: torch.Tensor, slots: torch.Tensor, config: Config
) -> torch.Tensor:
    both = torch.cat([links, links[:, [2, 3, 0, 1]]])
    negatives = torch.cat(
        [both[:, :2], random_replicas(slots, len(both))], dim=1
    )
    gap = link_distance(out, both) - link_distance(out, negatives)
    return torch.clamp(gap + config.triplet_margin, min=0).mean()


LOSSES = {"contrastive": contrastive_loss, "triplet": triplet_loss}


def train_residual(
    problem: Problem, loss_name: str, kinds: Tuple[str, ...]
) -> Replicas:
    config = problem.config
    model = ResidualAligner(config)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    raw = torch.tensor(problem.raw, dtype=torch.float32)
    links = torch.tensor(np.vstack([problem.links[kind] for kind in kinds]))
    slots = torch.tensor(np.argwhere(problem.member))
    for _ in range(config.train_steps):
        optimiser.zero_grad()
        loss = LOSSES[loss_name](model(raw), links, slots, config)
        loss.backward()
        optimiser.step()
    return model(raw).detach().numpy().astype(float)


ALIGNERS: Dict[str, Aligner] = {
    "none": lambda problem: problem.raw.copy(),
    "procrustes_cut": lambda problem: rotate_replicas(problem, (CUT,)),
    "procrustes_replica": lambda problem: rotate_replicas(problem, (REPLICA,)),
    "procrustes_cut_replica": lambda problem: rotate_replicas(
        problem, (CUT, REPLICA)
    ),
    "mlp_contrastive": lambda problem: train_residual(
        problem, "contrastive", (INTRA, CUT)
    ),
    "mlp_contrastive_replica": lambda problem: train_residual(
        problem, "contrastive", (INTRA, CUT, REPLICA)
    ),
    "mlp_triplet": lambda problem: train_residual(
        problem, "triplet", (INTRA, CUT)
    ),
    "mlp_triplet_replica": lambda problem: train_residual(
        problem, "triplet", (INTRA, CUT, REPLICA)
    ),
}


def merge_replicas(problem: Problem, aligned: Replicas) -> np.ndarray:
    weights = problem.member[:, :, None]
    return unit_rows((aligned * weights).sum(axis=0))


def sample_non_edges(
    problem: Problem, count: int, want_covered: int, rng: np.random.Generator
) -> Pairs:
    known = {
        frozenset(e)
        for block in (problem.train_edges, problem.test_edges)
        for e in block.tolist()
    }
    nodes = problem.member.shape[1]
    chosen: List[Tuple[int, int]] = []
    while len(chosen) < count:
        u, v = (int(x) for x in rng.integers(nodes, size=2))
        shared = int((problem.member[:, u] & problem.member[:, v]).any())
        if u != v and shared == want_covered and frozenset((u, v)) not in known:
            chosen.append((u, v))
    return np.array(chosen)


def link_auc(vectors: np.ndarray, positives: Pairs, negatives: Pairs) -> float:
    pairs = np.vstack([positives, negatives])
    scores = -np.linalg.norm(vectors[pairs[:, 0]] - vectors[pairs[:, 1]], axis=1)
    labels = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    return float(roc_auc_score(labels, scores))


def replica_cosine(problem: Problem, aligned: Replicas) -> float:
    twins = problem.links[REPLICA]
    if len(twins) == 0:
        return float("nan")
    left = aligned[twins[:, 0], twins[:, 1]]
    right = aligned[twins[:, 2], twins[:, 3]]
    return float((left * right).sum(axis=1).mean())


def evaluate_nodes(
    problem: Problem, vectors: np.ndarray, rng: np.random.Generator
) -> Dict[str, float]:
    is_covered = covered(problem.member, problem.test_edges)
    cut_test = problem.test_edges[~is_covered]
    within_test = problem.test_edges[is_covered]
    cut_negatives = sample_non_edges(problem, len(cut_test), 0, rng)
    within_negatives = sample_non_edges(problem, len(within_test), 1, rng)
    return {
        "cross_auc": link_auc(vectors, cut_test, cut_negatives),
        "within_auc": link_auc(vectors, within_test, within_negatives),
    }


def describe(problem: Problem) -> Dict[str, float]:
    edges = np.vstack([problem.train_edges, problem.test_edges])
    return {
        "cut_fraction": float((~covered(problem.member, edges)).mean()),
        "replicated_fraction": float(
            (problem.member.sum(axis=0) > 1).mean()
        ),
        "replica_anchors": float(len(problem.links[REPLICA])),
    }


def run_alignment(
    config: Config, leak: float, replication: float, seed: int
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    problem, oracle = build_problem(config, leak, replication, rng)
    results = {"setting": describe(problem)}
    results["oracle"] = evaluate_nodes(
        problem, oracle, np.random.default_rng(seed)
    )
    for name, aligner in ALIGNERS.items():
        aligned = aligner(problem)
        merged = merge_replicas(problem, aligned)
        results[name] = evaluate_nodes(
            problem, merged, np.random.default_rng(seed)
        )
        results[name]["replica_cosine"] = replica_cosine(problem, aligned)
        LOGGER.info(
            "leak=%s rep=%s seed=%s %s %s",
            leak, replication, seed, name, results[name],
        )
    return results


def seed_misalignment(config: Config, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    graph = sample_graph(config, rng)
    nodes = graph.number_of_nodes()
    first = embed_graph(graph, nodes, config, rng)
    second = embed_graph(graph, nodes, config, rng)
    rotation = procrustes_rotation(first, second)
    linear = np.linalg.lstsq(first, second, rcond=None)[0]
    scale = np.linalg.norm(second) ** 2
    return {
        "identity": float(np.linalg.norm(first - second) ** 2 / scale),
        "rotation": float(
            np.linalg.norm(first @ rotation - second) ** 2 / scale
        ),
        "linear": float(np.linalg.norm(first @ linear - second) ** 2 / scale),
    }


def summarise(runs: List[Dict[str, Dict]]) -> Dict[str, Dict[str, str]]:
    summary: Dict[str, Dict[str, str]] = {}
    for method in runs[0]:
        summary[method] = {}
        for metric in runs[0][method]:
            values = np.array([run[method][metric] for run in runs])
            summary[method][metric] = f"{values.mean():.3f}±{values.std():.3f}"
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logging.getLogger("gensim").setLevel(logging.WARNING)
    config = Config()
    report: Dict[str, Dict] = {}
    report["seed_misalignment"] = summarise(
        [{"relative_error": seed_misalignment(config, s)} for s in config.seeds]
    )
    for leak in config.leaks:
        for replication in config.replications:
            runs = [
                run_alignment(config, leak, replication, seed)
                for seed in config.seeds
            ]
            report[f"leak={leak},replication={replication}"] = summarise(runs)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```


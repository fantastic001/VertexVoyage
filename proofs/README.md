# Machine-checked proofs

Lean 4 (v4.33.1) and Mathlib (v4.33.1) proofs of every theorem stated in
[`doc/`](../doc/README.md) and [`results/`](../results/README.md). No proof
uses `sorry`. The [theory index](../doc/theory/README.md) maps each English
statement to its Lean name.

| File | Formalises |
|---|---|
| `PartitionBalance.lean` | range of the balance metric, [evaluation.md](../doc/evaluation.md#2-partition-quality) |
| `PartitioningF1.lean` | within/cross decomposition of expected F1, [E08](../results/E08-f1-decomposition/README.md) |
| `PartitioningF1NodeAveraged.lean` | node-averaged (macro) F1 decomposition, E08 |
| `PartitioningF1ChanceFloor.lean` | chance floor of a global pair budget, E08 |
| `ErdosRenyi.lean`, `ErdosRenyiPartitions.lean`, `ErdosRenyiChanceFloor.lean` | $G(n,p)$ facts used by the E08 Erdős–Rényi sweep |
| `PartitionAlignment.lean` | Theorems 1–18 of the residual aligner, [theory](../doc/theory/partition-alignment.md) and [E10](../results/E10-partition-alignment/README.md) |
| `PrequentialLinkPrediction.lean` | no leakage (L2) and valid negatives (L3) of prequential link prediction, [evaluation.md](../doc/evaluation.md#prequential-link-prediction-pipeline-t) |

## Building

```bash
cd proofs
lake exe cache get   # downloads prebuilt Mathlib
lake build
```

The build is also run by the `proofs` GitHub workflow on every push that
changes `proofs/`.

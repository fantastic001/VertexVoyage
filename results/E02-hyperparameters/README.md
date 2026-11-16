# E02 — Hyperparameter tuning of the streaming pipeline

| | |
|---|---|
| **Research question** | [RQ2](../README.md#research-questions) |
| **Status** | ✅ done: selected the defaults of every later streaming experiment |
| **Period** | 2026-05 → 2026-09 |
| **Pipeline** | T, streaming (`vv temporal_test`) |
| **Datasets** | CITESEER, Cit-HepPh, AstroPh |
| **Headline** | Buffer size 1000 with walks from all affected vertices (`neighbors.all`), $\mu = 1$, $\alpha = 1$, $\epsilon = 0.1$, no decay. About 2 h per AstroPh run. |
| **Open issues** | Single runs without fixed seeds and no error bars. The buffer-size findings are summarised in [E07](../E07-buffer-size/README.md). |

**Question.** Which buffer size, balance penalty $\mu$, return parameter
$\alpha$ and replication factor $RF$ give the best reconstruction F1 at an
acceptable runtime? Output feeds the fixed hyperparameter set used by every
other experiment.

# Setup

- we validate on CITESEER and Cit-HepPh datasets.
- P = 4,8,16
- RF = 1,3
- Epochs = 5
- $\mu \in \{0, 0.5, 1, 2\}$
- $\alpha \in \{0.5, 1, 2\}$
- $\epsilon = 0.1$
- Decay is set to none

From this, we select the best hyperparameters based on the F1 score for each dataset.

Checklist:
* Implement script to run Dynnode2vec with different hyperparameters and save F1 scores for each run.
* Run the script on CITESEER and Cit-HepPh datasets with the specified hyperparameters.
* Analyze the results and select the best hyperparameters based on F1 scores for each dataset.
* Document the results and the selected hyperparameters in the README file.


# Results
```
Static node2vec astroph                        -> 72% - cca 5min

P=1 AstroPh without existing nodes buffer=100  -> 1% - small reconstruction score over whole time - cca 1h
P=1 AstroPh without existing nodes buffer=1000 -> 46% - cca 30min
        started at 17:32
        small f1 restore in the beggining
        but it got to 22% on first half
        almost to the end - 34%
P=1 AstroPh without existing nodes buffer=2000 -> 46% cca 20min
        without f1 comp
        started at 19:35
P=1 AstroPh without existing nodes buffer=10000 -> 63% - cca 1h
        we removed f1 score computations
        started at 18:08

P=1 AstroPh with existing nodes buffer=100  -> 0.8% - cca 2h
        dtarted at 10:49
P=1 AstroPh with existing nodes buffer=1000 ->  57% - cca 2h
P=1 AstroPh with existing nodes buffer=10000 -> 63% - cca 1h
        started at 01:23
```


Results:

```

AstroPh buffer=1000 RF=1
P       F1 score mu=0           F1 score mu=0.5         F1 score mu=1
1       0.57                    0.57                    0.57
2       0.57                    0.5481965981124276      0.5627710515942038
4       0.57                    0.5199974124493743      0.49924681405353183
8       0.57                    0.3393135144445034      0.4660607177637196
CITESEER buffer=1000 RF=1
P       F1 score mu=0           F1 score mu=0.5         F1 score mu=1
1       0.58                    0.58                    0.58
2       0.58                    0.41762272059740674     0.4618309398461626
4       0.58                    0.4353666681248792      0.4434664667225492
8       0.58                    0.44306591097645515     0.4909383858875828


CITESEER buffer=1000 RF=3
P       F1 score mu=0           F1 score mu=0.5         F1 score mu=1
4       0.69
```

Command used for testing:

```bash
vv temporal_test --name CITESEER  --long-run --iterations 1  --use-dataset-params  --buffer-size 1000 --replication-factor 1 --partitioner neighbors.all  --mu 0.5 --partitions 8
```

Clear choice - buffer size 1000 with random walks on all affected nodes - expected time is 2h

| Hyperparameter | Value |
|----------------|-------|
| Buffer size    | 1000  |
| $\mu$          | 1     |
| $\alpha$       | 1     |
| $\epsilon$     | 0.1   |
| Decay         | none  |
| Partitions (P) | $\{1, 2, 4, 8\}$ |
| Replication factor (RF) | $\{1, 3\}$ |
| Epochs        | Default     |


# Adding a dataset

1. Download and preprocess the dataset. Typical sources:
   - <https://snap.stanford.edu/data/>
   - <https://github.com/graphsinspace/graspe/tree/main/data>
2. Put the files into `data/`.
3. Register the dataset loader in
   [`experiments/datasets.py`](../../experiments/datasets.py) (`datasets`).
4. Add its default embedding parameters ($p$, $q$, `dim`) to
   `dataset_params` in the same file. Parameter values from the literature:
   [arXiv:2208.11986](https://arxiv.org/pdf/2208.11986) and
   [arXiv:1812.02356](https://arxiv.org/pdf/1812.02356).
5. Test the new dataset and tune the parameters if necessary:

   ```sh
   vv test --name DATASET_NAME --partitions 1 --long-run --epochs 5
   ```

   or, once the parameters are tuned:

   ```sh
   vv test --name DATASET_NAME --partitions 1 --long-run --epochs 5 --use-dataset-params
   ```

   or for temporal networks:

   ```sh
   vv temporal_test --name DATASET_NAME --long-run --iterations 1 \
     --use-dataset-params --buffer-size 50 --replication-factor 1 \
     --partitioner-name neighbors.all --mu 0.5 --partitions 1
   ```

6. Add the dataset with its citation to [datasets.md](../datasets.md).

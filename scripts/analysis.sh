#!/bin/bash 

# analyze_embeddings single_node_result multi_node_result clusters

MODES="sequential parallel"
PIPELINES=$(ls pipelines | sed 's/\.yml//g')

RESULTS_FOLDER="results/cluster_similarity/"

for PIPELINE in $PIPELINES; do
    for N_CLUSTERS in 2 3 4 5 6 7 8 9 10; do
        echo "Running pipeline $PIPELINE with $N_CLUSTERS clusters in $MODE mode"
        python -m client analyze_embeddings \
            --single-node-results results/sequential/$PIPELINE/2_process.json \
            --multi-node-results results/sequential/$PIPELINE/2_process.json \
            --clusters $N_CLUSTERS \
            --output $RESULTS_FOLDER/$PIPELINE/$N_CLUSTERS/analysis_dim.csv
        python -m client analyze_embeddings \
            --single-node-results results/sequential/$PIPELINE/3_process.json \
            --multi-node-results results/sequential/$PIPELINE/3_process.json \
            --clusters $N_CLUSTERS \
            --output $RESULTS_FOLDER/$PIPELINE/$N_CLUSTERS/analysis_n_walks.csv
    done
done
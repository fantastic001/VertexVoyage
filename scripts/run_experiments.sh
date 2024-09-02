#!/bin/bash 
MODE=$1
PIPELINES=$(ls pipelines | sed 's/\.yml//g')

if [ -z "$MODE" ]; then
    echo "Usage: $0 <mode>"
    exit 1
fi

for PIPELINE in $PIPELINES; do
    echo "Running pipeline $PIPELINE"
    python -m client execute --pipeline pipelines/$PIPELINE.yml --results-folder results/$MODE/$PIPELINE
    python -m client analyze_reconstruction \
        --vertex-result  results/$MODE/$PIPELINE/4_get_vertices.json \
        --edge-result    results/$MODE/$PIPELINE/5_get_edges.json \
        --embeddings-result results/$MODE/$PIPELINE/2_process.json \
        --output results/$MODE/$PIPELINE/f1_score_dim.csv 
    python -m client analyze_reconstruction \
        --vertex-result  results/$MODE/$PIPELINE/4_get_vertices.json \
        --edge-result    results/$MODE/$PIPELINE/5_get_edges.json \
        --embeddings-result results/$MODE/$PIPELINE/3_process.json \
        --output results/$MODE/$PIPELINE/f1_score_n_walks.csv 
done
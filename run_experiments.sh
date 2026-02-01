#!/bin/bash 

export VERTEX_VOYAGE_PLUGINS="experiments.*"

# python -m vertex_voyage.benchmark 


if [ "$DEBUG" = "1" ]; then
    echo "Running in DEBUG mode"
    set -x
fi

for dataset in zachary AstroPh CITESEER Cit-HepPh Cit-HepTh; do 
    for parts in 1 2 4 8 16; do 
        echo "----------------------------------------"
        echo "Running experiment on dataset $dataset with $parts partitions"
        python -m vertex_voyage  test \
            --name $dataset \
            --partitions $parts \
            --break-early \
            --use-dataset-params \
            --long-run \
            --use-lpa \
            --epochs 10 
        echo "----------------------------------------"
    done
done

echo "All experiments completed."
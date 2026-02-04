#!/bin/bash

# Defaults
SIMS=1024
NCAR=4
SEED=111
ITER=10000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sims)  SIMS="$2";  shift 2 ;;
        --ncar)  NCAR="$2";  shift 2 ;;
        --seed)  SEED="$2";  shift 2 ;;
        --iter)  ITER="$2";  shift 2 ;;
        -h|--help)
            echo "Usage: ./run.sh [OPTIONS]"
            echo "  --sims <n>   Number of simulations (default: 1024)"
            echo "  --ncar <n>   Cars per team (default: 4)"
            echo "  --seed <n>   Random seed (default: 111)"
            echo "  --iter <n>   Iterations (default: 10000)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

./build/supersonic "$SIMS" "$NCAR" "$SEED" "$ITER"

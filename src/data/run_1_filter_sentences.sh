#!/bin/bash -x

#SBATCH --time=2-0
#SBATCH -c68
#SBATCH --mem=350g
#SBATCH --output=1_filter_sentences_%A.out
#SBATCH --error=1_filter_sentences_%A.err

# Resources above reproduce the paper run.

# Run from the repository root
cd /path/to/repo
echo $PWD

# Activate your virtual environment (adjust the path)
. $PWD/myenv/bin/activate

# Make config/ and src/ importable
export PYTHONPATH=$PWD

python3 src/data/1_filter_sentences_by_queryTerms.py \
    --config-path config/data/configData_cs2_qt5.json \
    --num_workers 60 \
    --shards-dir ./data_s2orc/output

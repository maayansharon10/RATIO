#!/bin/bash -x

#SBATCH --time=2-0
#SBATCH -c1
#SBATCH --mem=300g
#SBATCH --output=2_create_datasets_%A.out
#SBATCH --error=2_create_datasets_%A.err

# Resources above reproduce the paper run.

set -e

# Run from the repository root
cd /path/to/repo
echo $PWD

# Activate your virtual environment (adjust the path)
. $PWD/myenv/bin/activate

# Make config/ and src/ importable
export PYTHONPATH=$PWD

python src/data/2_split_filterGroups_construct_datasets_temporal.py \
  --config-data-path config/data/configData_cs2_qt5.json \
  --split-mode temporal_cutoff_2026_valQ4 \
  --date-from 20150101 \
  --date-to   20260601

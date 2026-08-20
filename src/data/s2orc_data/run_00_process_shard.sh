#!/bin/bash -x

#SBATCH --time=7-0
#SBATCH -c63
#SBATCH --mem=400g
#SBATCH --output=s2orc_00_process_shard_%A.out
#SBATCH --error=s2orc_00_process_shard_%A.err

# Resources above reproduce the paper run.

# Run from the repository root
cd /path/to/repo
echo $PWD

# Activate your virtual environment (adjust the path)
. $PWD/myenv/bin/activate

# Semantic Scholar API key with bulk dataset access
export S2_API_KEY="your_key_here"

python src/data/s2orc_data/00_process_shard.py --phase 0 --workers 60
python src/data/s2orc_data/00_process_shard.py --phase 1 --workers 50

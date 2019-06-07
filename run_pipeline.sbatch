#!/bin/bash
#SBATCH --job-name=full_pipeline
#SBATCH --output=code/output/full_pipeline.out
#SBATCH --error=code/output/full_pipeline.err
#SBATCH --time=04:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mem-per-cpu=4000

source env/bin/activate
module load python/3.6.1+intel-16.0

python3 code/chicago-business.py

deactivate

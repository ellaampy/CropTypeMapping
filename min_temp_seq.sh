#!/bin/bash -l
#SBATCH --chdir=/share/projects/fuses1s2/CropTypeMapping/CropTypeMapping
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 4G
#SBATCH --partition shortrun
#SBATCH --output min_temp_length.out
#SBATCH --time=1-00:00:00


conda activate ee

python3 min_temp_seq.py --input /share/projects/fuses1s2/s1_data --init_len 75

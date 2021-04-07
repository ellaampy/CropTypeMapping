#!/bin/bash -l
#SBATCH --chdir=/share/projects/fuses1s2/pse_tsa_adapted
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 8G
#SBATCH --partition longrun
#SBATCH --output Brest_1_pse.out
#SBATCH --time=2-00:00:00


conda activate dfc

python3 train.py --dataset_folder /share/projects/fuses1s2/s2_data/Brest_1 --kfold 3 --epochs 10 --num_classes 15 

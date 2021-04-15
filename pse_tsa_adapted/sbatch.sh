#!/bin/bash -l
#SBATCH --chdir=/share/projects/fuses1s2/CropTypeMapping/CropTypeMapping/pse_tsa_adapted
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 8G
#SBATCH --partition shortrun
#SBATCH --output Brest_pse_S1.out
#SBATCH --time=1-00:00:00


conda activate dfc

##sentinel2
##python3 train.py --dataset_folder /share/projects/fuses1s2/s2_data/Brest --kfold 5 --epochs 50 --num_classes 13  --geomfeat 0 --lms 27  --mlp2 [128,128] --positions None  --npixel 64 --preload 

##sentinel1
python3 train.py --dataset_folder /share/projects/fuses1s2/s1_data/Brest --kfold 5 --epochs 50 --num_classes 13  --geomfeat 0 --lms 75  --mlp2 [128,128] --positions None --input_dim 2 --mlp1 [2,32,64] --preload --res_dir /share/projects/fuses1s2/CropTypeMapping/CropTypeMapping/pse_tsa_adapted/results_s1

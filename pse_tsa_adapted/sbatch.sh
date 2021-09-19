#!/bin/bash -l
#SBATCH --chdir=/share/projects/fuses1s2/CropTypeMapping/CropTypeMapping/pse_tsa_adapted
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 4G
#SBATCH --partition shortrun
#SBATCH --output /share/projects/fuses1s2/results_PseTsa/batch_size/b_256/s2_%A_%a.out
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2

conda activate dfc

SEED=(1 42 123456 100 1482)

##SINGLE LOADER
##sentinel2
##python3 train_singleloader.py --dataset_folder /share/projects/fuses1s2/s2_data/Brest --kfold 5 --epochs 100 --sensor S2 --num_classes 13  --geomfeat 0 --lms 27  --mlp2 [128,128] --positions bespoke  --npixel 64 --preload --res_dir /share/projects/fuses1s2/CropTypeMapping/CropTypeMapping/pse_tsa_adapted/results_s2_brest

##sentinel1
##python3 train_singleloader.py --dataset_folder /share/projects/fuses1s2/s1_data/Brest --kfold 5 --epochs 100 --sensor S1 --num_classes 13  --geomfeat 0 --lms 75  --mlp2 [128,128] --positions bespoke --input_dim 2 --sensor S1 --mlp1 [2,32,64] --preload --res_dir /share/projects/fuses1s2/CropTypeMapping/CropTypeMapping/pse_tsa_adapted/results_s1_brest


##TRAIN-MULTILOADER
##------------------mini test---------------------
##python3 train_multiloader.py --dataset_folder /share/projects/fuses1s2/sample_data/s1_data --val_folder /share/projects/fuses1s2/sample_data/s1_data --test_folder /share/projects/fuses1s2/sample_data/s1_data --rdm_seed ${SEED[$SLURM_ARRAY_TASK_ID]}  --epochs 5 --sensor S1 --num_classes 12  --geomfeat 0 --lms 460  --mlp2 [128,128] --npixel 64 --num_workers 8 --preload --batch_size 1024  --input_dim 2 --mlp1 [2,32,64] --gamma 2 --res_dir /share/projects/fuses1s2/results_PseTsa/test_${SEED[$SLURM_ARRAY_TASK_ID]}

##sentinel 1
##python3 train_multiloader.py --dataset_folder /share/projects/fuses1s2/s1_data/Quimper --dataset_folder2 /share/projects/fuses1s2/s1_data/Chateaulin --val_folder /share/projects/fuses1s2/s1_data/Morlaix --test_folder /share/projects/fuses1s2/s1_data/Brest --rdm_seed ${SEED[$SLURM_ARRAY_TASK_ID]} --epochs 100 --sensor S1 --num_classes 12  --geomfeat 0 --lms 460  --mlp2 [128,128] --npixel 64 --num_workers 4 --preload --batch_size 1024  --input_dim 2 --mlp1 [2,32,64] --gamma 2 --res_dir /share/projects/fuses1s2/results_PseTsa/s1_probs_${SEED[$SLURM_ARRAY_TASK_ID]}

##sentinel 2
python3 train_multiloader.py --dataset_folder /share/projects/fuses1s2/s2_data/Quimper --dataset_folder2 /share/projects/fuses1s2/s2_data/Chateaulin --val_folder /share/projects/fuses1s2/s2_data/Morlaix --test_folder /share/projects/fuses1s2/s2_data/Morlaix --epochs 100 --rdm_seed ${SEED[$SLURM_ARRAY_TASK_ID]} --sensor S2 --num_classes 12 --geomfeat 0 --lms 460  --mlp2 [128,128] --npixel 64 --num_workers 8 --batch_size 256  --input_dim 10 --mlp1 [10,32,64] --gamma 2 --res_dir /share/projects/fuses1s2/results_PseTsa/batch_size/b_256/s2_${SEED[$SLURM_ARRAY_TASK_ID]}

#!/bin/bash -l
#SBATCH --chdir=/share/projects/fuses1s2/CropTypeMapping/CropTypeMapping/pse_tsa_adapted
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 8G
#SBATCH --partition shortrun
#SBATCH --output /share/projects/fuses1s2/results_PseTsa/single_sensor_s1_3.out
#SBATCH --time=1-00:00:00


conda activate dfc

##SINGLE LOADER
##sentinel2
##python3 train_singleloader.py --dataset_folder /share/projects/fuses1s2/s2_data/Brest --kfold 5 --epochs 100 --sensor S2 --num_classes 13  --geomfeat 0 --lms 27  --mlp2 [128,128] --positions bespoke  --npixel 64 --preload --res_dir /share/projects/fuses1s2/CropTypeMapping/CropTypeMapping/pse_tsa_adapted/results_s2_brest

##sentinel1
##python3 train_singleloader.py --dataset_folder /share/projects/fuses1s2/s1_data/Brest --kfold 5 --epochs 100 --sensor S1 --num_classes 13  --geomfeat 0 --lms 75  --mlp2 [128,128] --positions bespoke --input_dim 2 --sensor S1 --mlp1 [2,32,64] --preload --res_dir /share/projects/fuses1s2/CropTypeMapping/CropTypeMapping/pse_tsa_adapted/results_s1_brest


##TRAIN-MULTILOADER
#sentinel 1
python3 train_multiloader.py --dataset_folder /share/projects/fuses1s2/s1_data/Chateaulin --dataset_folder2 /share/projects/fuses1s2/s1_data/Quimper --val_folder /share/projects/fuses1s2/s1_data/Brest --test_folder /share/projects/fuses1s2/s1_data/Morlaix --epochs 100 --sensor S1 --num_classes 12  --geomfeat 0 --lms 460  --mlp2 [128,128] --npixel 64 --num_workers 8 --preload --batch_size 1024  --input_dim 2 --mlp1 [2,32,64] --preload --gamma 2 --res_dir /share/projects/fuses1s2/results_PseTsa/results_s1_3

#sentinel 2
#python3 train_multiloader.py --dataset_folder /share/projects/fuses1s2/s2_data/Chateaulin --dataset_folder2 /share/projects/fuses1s2/s2_data/Quimper --val_folder /share/projects/fuses1s2/s2_data/Brest --test_folder /share/projects/fuses1s2/s2_data/Morlaix --epochs 100 --sensor S2 --num_classes 12  --preload --geomfeat 0 --lms 460  --mlp2 [128,128] --npixel 64 --num_workers 8 --batch_size 1024  --input_dim 10 --mlp1 [10,32,64] --gamma 2 --res_dir /share/projects/fuses1s2/results_PseTsa/results_s2_5

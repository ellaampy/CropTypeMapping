#!/bin/bash -l
#SBATCH --chdir=/share/projects/fuses1s2/CropTypeMapping/CropTypeMapping/pse_tsa_adapted
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 4G
#SBATCH --partition shortrun
#SBATCH --output /share/projects/fuses1s2/results_PseTsa/S2_test.out
#SBATCH --time=1-00:00:00


conda activate dfc

##SINGLE LOADER
##sentinel2
##python3 train_singleloader.py --dataset_folder /share/projects/fuses1s2/s2_data/Brest --kfold 5 --epochs 100 --sensor S2 --num_classes 13  --geomfeat 0 --lms 27  --mlp2 [128,128] --positions bespoke  --npixel 64 --preload --res_dir /share/projects/fuses1s2/CropTypeMapping/CropTypeMapping/pse_tsa_adapted/results_s2_brest

##sentinel1
##python3 train_singleloader.py --dataset_folder /share/projects/fuses1s2/s1_data/Brest --kfold 5 --epochs 100 --sensor S1 --num_classes 13  --geomfeat 0 --lms 75  --mlp2 [128,128] --positions bespoke --input_dim 2 --sensor S1 --mlp1 [2,32,64] --preload --res_dir /share/projects/fuses1s2/CropTypeMapping/CropTypeMapping/pse_tsa_adapted/results_s1_brest


##TRAIN-MULTILOADER
#sentinel 1
python3 train_multiloader.py --dataset_folder /share/projects/fuses1s2/sample_data/s1_data --dataset_folder2 /share/projects/fuses1s2/sample_data/s1_data --val_folder /share/projects/fuses1s2/sample_data/s1_data --test_folder /share/projects/fuses1s2/sample_data/s1_data --epochs 10 --sensor S1 --num_classes 12  --geomfeat 0 --lms 75  --mlp2 [128,128] --positions bespoke  --npixel 64 --num_workers 8 --preload --batch_size 1024  --input_dim 2 --mlp1 [2,32,64] --preload --gamma 2 --res_dir /share/projects/fuses1s2/results_PseTsa/test_s1_

#sentinel 2
python3 train_multiloader.py --dataset_folder /share/projects/fuses1s2/sample_data/s2_data --dataset_folder2 /share/projects/fuses1s2/sample_data/s2_data --val_folder /share/projects/fuses1s2/sample_data/s2_data --test_folder /share/projects/fuses1s2/sample_data/s2_data --epochs 10 --sensor S2 --num_classes 12  --geomfeat 0 --lms 27  --mlp2 [128,128] --positions bespoke  --npixel 64 --num_workers 8 --preload --batch_size 1024  --input_dim 10 --mlp1 [10,32,64] --preload --gamma 2 --res_dir /share/projects/fuses1s2/results_PseTsa/test_s2_1

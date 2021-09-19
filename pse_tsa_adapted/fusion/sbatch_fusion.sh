#!/bin/bash -l
#SBATCH --chdir=/share/projects/fuses1s2/CropTypeMapping/CropTypeMapping/pse_tsa_adapted/fusion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 8G
#SBATCH --partition shortrun
#SBATCH --output /share/projects/fuses1s2/results_PseTsa/batch_size/fusion/b_128/results_pse_1.out
#SBATCH --time=1-00:00:00


conda activate dfc

##TRAIN-MULTILOADER

##batch size
python3 train_multiloader_fusion.py --dataset_folder /share/projects/fuses1s2/s1_data/Quimper --dataset_folder2 /share/projects/fuses1s2/s1_data/Chateaulin --val_folder /share/projects/fuses1s2/s1_data/Morlaix --test_folder /share/projects/fuses1s2/s1_data/Morlaix --epochs 100 --fusion_type pse --rdm_seed 1 --num_classes 12  --gamma 2 --geomfeat 0  --mlp1 [10,32,64] --mlp2 [128,128] --mlp3 [512,128,128] --lms 460 --npixel 64 --num_workers 8 --batch_size 128 --res_dir /share/projects/fuses1s2/results_PseTsa/batch_size/fusion/b_128/results_pse1

##early fusion
##python3 train_multiloader_fusion.py --dataset_folder /share/projects/fuses1s2/s1_data/Quimper --dataset_folder2 /share/projects/fuses1s2/s1_data/Chateaulin --val_folder /share/projects/fuses1s2/s1_data/Morlaix --test_folder /share/projects/fuses1s2/s1_data/Brest --epochs 100 --fusion_type early_dates --rdm_seed 42 --num_classes 12  --gamma 2 --geomfeat 0  --mlp1 [10,32,64] --mlp2 [128,128] --mlp3 [512,128,128]  --lms 460 --npixel 64 --num_workers 8 --preload --batch_size 1024 --res_dir /share/projects/fuses1s2/results_PseTsa/fusion/results_early_42

##pse fusion
##python3 train_multiloader_fusion.py --dataset_folder /share/projects/fuses1s2/s1_data/Quimper --dataset_folder2 /share/projects/fuses1s2/s1_data/Chateaulin --val_folder /share/projects/fuses1s2/s1_data/Morlaix --test_folder /share/projects/fuses1s2/s1_data/Brest --epochs 100 --fusion_type pse --rdm_seed 1 --num_classes 12  --gamma 2 --geomfeat 0  --mlp1 [10,32,64] --mlp2 [128,128] --mlp3 [512,128,128] --lms 460 --npixel 64 --num_workers 8 --preload --batch_size 1024 --res_dir /share/projects/fuses1s2/results_PseTsa/fusion/results_pse1

##tsa fusion
##python3 train_multiloader_fusion.py --dataset_folder /share/projects/fuses1s2/s1_data/Quimper --dataset_folder2 /share/projects/fuses1s2/s1_data/Chateaulin --val_folder /share/projects/fuses1s2/s1_data/Morlaix --test_folder /share/projects/fuses1s2/s1_data/Brest --epochs 100 --fusion_type tsa --rdm_seed 100 --num_classes 12  --gamma 2 --geomfeat 0  --mlp1 [10,32,64] --mlp2 [128,128] --mlp3 [512,128,128] --lms 460 --npixel 64 --num_workers 8 --batch_size 1024 --res_dir /share/projects/fuses1s2/results_PseTsa/fusion/results_tsa_final4

##softmax averaging
##python3 train_multiloader_fusion.py --dataset_folder /share/projects/fuses1s2/s1_data/Quimper --dataset_folder2 /share/projects/fuses1s2/s1_data/Chateaulin --val_folder /share/projects/fuses1s2/s1_data/Morlaix --test_folder /share/projects/fuses1s2/s1_data/Brest --epochs 100 --fusion_type softmax --rdm_seed 1 --num_classes 12  --gamma 2 --geomfeat 0  --mlp1 [10,32,64] --mlp2 [128,128] --mlp3 [512,128,128] --lms 460 --npixel 64 --num_workers 8 --preload --batch_size 1024 --res_dir /share/projects/fuses1s2/results_PseTsa/fusion/results_softmax_avg3


##softmax normalization
##python3 train_multiloader_fusion.py --dataset_folder /share/projects/fuses1s2/s1_data/Quimper --dataset_folder2 /share/projects/fuses1s2/s1_data/Chateaulin --val_folder /share/projects/fuses1s2/s1_data/Morlaix --test_folder /share/projects/fuses1s2/s1_data/Brest --epochs 100 --fusion_type softmax_norm --rdm_seed 1 --num_classes 12  --gamma 2 --geomfeat 0  --mlp1 [10,32,64] --mlp2 [128,128] --mlp3 [512,128,128] --lms 460 --npixel 64 --num_workers 8 --preload --batch_size 1024  --input_dim 10 --res_dir /share/projects/fuses1s2/results_PseTsa/fusion/results_softmax_norm1


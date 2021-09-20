# Sentinel-1 & 2 time series for crop type mapping using PSE-TSA
This code extents the pytorch implementation of the [PSE-TSA](https://github.com/VSainteuf/pytorch-psetae) architecture to accomodate different forms of multi-sensor fusion.

## Requirements
- Pytorch + torchnet
- numpy + pandas + sklearn

## Data preparation
* follow the guideline [here](https://github.com/ellaampy/GEE-to-NPY) to download normalized parcel-level Sentinel-1 & 2 time series (independently) from Google Earth Engine
* otherwise, prepare parcel-level time series of shape (```T x C x N ```) where;
    * T --> number of acquisitions
    * C --> number of channels/bands
    * N --> number of pixels within parcel
* run ```data_preprartion/min_temp_seq.py``` to decide a minimum sampling size. eg. assuming study area is enveloped by multiple overlapping satellite resulting in varying time series length
* organize time series array into seperate folders from training, validation and testing. 

## Folder structure
The root folder should contain Sentinel-1 and Sentinel-2 directory named ```s1_data ``` and ```s2_data ```. Their sub-directories must be similar to the figure below
<img src="img/folder_structure.PNG" alt="folder structure" width="500">

## Running main experiments
```python

# single sensor (Sentinel-1)
train.py --dataset_folder /s1_data/Quimper --val_folder /s1_data/Morlaix --test_folder /s1_data/Brest --epochs 100 --rdm_seed 1 --sensor S1 --input_dim 2 --mlp1 [2,32,64] --num_classes 12 --minimum_sampling 27 --res_dir /output_dir

# multi-sensor (early fusion)
train_fusion.py --dataset_folder /s1_data/Quimper --val_folder /s1_data/Morlaix --test_folder /s1_data/Brest --fusion_type early --minimum_sampling 27 --interpolate_method nn --epochs 100 --rdm_seed 1 --input_dim 2 --mlp1 [2,32,64] --num_classes 12 --res_dir /output_dir

"""
for multi-sensor, Sentinel-1 data directory (s1_data) is modified as (s2_data) in the dataset.py script to load Sentinel-2 data. Additionally, input_dim and mlp1-4 are handled within multi_sensor/models/stclassifier_fusion.py
"""
```

Types of fusion
![fusion diagrams](img/fusion.gif)

## Results
Quantitative results from single and multi-sensor experiments are available in the `results` folder/ 

## Credits
* This research relies heavily on the [paper](https://arxiv.org/pdf/1911.07757.pdf) "Satellite Image Time Series Classification with Pixel-Set Encoders and Temporal Self-Attention" by Saint Fare Garnot et al.
* The label data originates from [Registre parcellaire graphique (RPG)](https://www.data.gouv.fr/fr/datasets/registre-parcellaire-graphique-rpg-contours-des-parcelles-et-ilots-culturaux-et-leur-groupe-de-cultures-majoritaire/) of the French National Geographic Institute (IGN)


## Reference
Please cite the following paper if you use any part of the code

```
citation goes here
```

## Contributors
 - [Dr. Charlotte Pelletier](https://sites.google.com/site/charpelletier)
 - [Dr. Stefan Lang](https://scholar.google.com/citations?user=e0X2Y0gAAAAJ&hl=en)
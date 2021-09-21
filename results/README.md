# Experimental results
All experiments leveraged complete Sentinel-1 and or Sentinel-2 time series except for early and PSE fusion where S1 is interpolated at S2 observation dates.

```results_all_experiments.xlsx``` --> contains quantitave results for single (Sentinel-1 & Sentinel-2) and multi-sensor experiments (early, PSE, TSA, decision) fusion.


# Incremental learning
Here the benefit of multi-sensor fusion for classifying crops before having a full year of data is studied. For this purpose, experiments are run with quarterly increments in data from October 2018 to December 2019.

```incremental_learning.xlsx``` --> contains quantitave results for Sentinel-1, Sentinel-2 and early fusion


# Sparse satellite image time series
In this setup, Sentinel-2 observations are gradually reduced to mimic cloudy conditions, which limits the use of optical data. A certain percentage of the minimum length of Sentinel-2 is randomly sampled at each trial of five runs. 

```sparse_time_series.xlsx``` --> contains quantitave results for Sentinel-2, early, TSA and decision fusion

All results are averaged over 5x runs.

import torch
from torch import Tensor
from torch.utils import data

import pandas as pd
import numpy as np
import datetime as dt

import os
import json
import random


class PixelSetData(data.Dataset):
    def __init__(self, folder, labels, npixel, sub_classes=None, norm=None,
                 extra_feature=None, jitter=(0.01, 0.05), sensor=None, return_id=False):
        """
        
        Args:
            folder (str): path to the main folder of the dataset, formatted as indicated in the readme
            labels (str): name of the nomenclature to use in the labels.json file
            npixel (int): Number of sampled pixels in each parcel
            sub_classes (list): If provided, only the samples from the given list of classes are considered. 
            (Can be used to remove classes with too few samples)
            norm (tuple): (mean,std) tuple to use for normalization
            extra_feature (str): name of the additional static feature file to use
            jitter (tuple): if provided (sigma, clip) values for the addition random gaussian noise
            return_id (bool): if True, the id of the yielded item is also returned (useful for inference)
        """
        super(PixelSetData, self).__init__()

        self.folder = folder
        self.data_folder = os.path.join(folder, 'DATA')
        self.meta_folder = os.path.join(folder, 'META')
        self.labels = labels
        self.npixel = npixel
        self.norm = norm

        self.extra_feature = extra_feature
        self.jitter = jitter  # (sigma , clip )
        self.sensor = sensor
        self.return_id = return_id


# -------------------------pid block-----------------------------------------
        # deactivate to use succeeding block. here only parcel ids common in s1 and s2 are referenced
        # list of pre-computed common parcel ids in s1 and s2
        with open(os.path.join(folder, 'META', 'common_pids.json'), 'r') as file:
            self.common_pids = json.loads(file.read())
        
        # list of pre-computed parcels with uneven pixel count in s1 and s2
        with open(os.path.join(folder, 'META', 'uneven_shapes_pids.json'), 'r') as file:
            self.pid_uneven_shapes = json.loads(file.read())     
        
        self.pid = list(x for x in self.common_pids if x not in self.pid_uneven_shapes)        
        self.len = len(self.pid)


        
         # activate to use all available parcels for single sensor scenario
            
#        l = [f for f in os.listdir(self.data_folder) if f.endswith('.npy')]
#        self.pid = [int(f.split('.')[0]) for f in l]
#        self.pid = list(np.sort(self.pid))
#        self.pid = list(map(str, self.pid))
#
#        # for sentinel-1 remove parcel with sequence len <75
#        #s1_parcels_less_seq contains parcel ids with temporal seq < 75
#        if self.sensor == 'S1':
#            with open(os.path.join(folder, 'META', 's1_parcels_less_seq.json'), 'r') as file:
#                ignored_pid = json.loads(file.read())
#            self.pid = list(x for x in self.pid if x not in ignored_pid)

#        self.len = len(self.pid)
#---------------------------------------------------------------------------------------------

        # Get Labels
        if sub_classes is not None:
            sub_indices = []
            num_classes = len(sub_classes)
            convert = dict((c, i) for i, c in enumerate(sub_classes))

        with open(os.path.join(folder, 'META', 'labels.json'), 'r') as file:
            d = json.loads(file.read())
            self.target = []
            for i, p in enumerate(self.pid):
                t = d[labels][p]

                # add conditional to merge permanent(18) and temporal meadow(19)
                # this will reduce number of target classes by 1
                if t == 19:
                    t = 18

                self.target.append(t)
                if sub_classes is not None:
                    if t in sub_classes:
                        sub_indices.append(i)
                        self.target[-1] = convert[self.target[-1]]
        if sub_classes is not None:
            self.pid = list(np.array(self.pid)[sub_indices])
            self.target = list(np.array(self.target)[sub_indices])
            self.len = len(sub_indices)


        with open(os.path.join(folder, 'META', 'dates.json'), 'r') as file:
            d = json.loads(file.read())

        # for sentinel 1
        if self.sensor == 'S1':
            self.dates = [d[str(i)] for i in range(len(d))]
            self.date_positions = date_positions(self.dates)

        # for sentinel 2
        elif self.sensor == 'S2':
             self.dates = [d[i] for i in self.pid]
             self.date_positions = [date_positions(i) for i in self.dates]
        


        if self.extra_feature is not None:
            with open(os.path.join(self.meta_folder, '{}.json'.format(extra_feature)), 'r') as file:
                self.extra = json.loads(file.read())

            if isinstance(self.extra[list(self.extra.keys())[0]], int):
                for k in self.extra.keys():
                    self.extra[k] = [self.extra[k]]
            df = pd.DataFrame(self.extra).transpose()
            self.extra_m, self.extra_s = np.array(df.mean(axis=0)), np.array(df.std(axis=0))

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        """
        Returns a Pixel-Set sequence tensor with its pixel mask and optional additional features.
        For each item npixel pixels are randomly dranw from the available pixels.
        If the total number of pixel is too small one arbitrary pixel is repeated. The pixel mask keeps track of true
        and repeated pixels.
        Returns:
              (Pixel-Set, Pixel-Mask) or ((Pixel-Set, Pixel-Mask), Extra-features) with:
                Pixel-Set: Sequence_length x Channels x npixel
                Pixel-Mask : Sequence_length x npixel
                Extra-features : Sequence_length x Number of additional features

        """
        x0 = np.load(os.path.join(self.folder, 'DATA', '{}.npy'.format(self.pid[item])))
        y = self.target[item]

#        # implement vh/vv ratio
#        if self.sensor == 'S1':
#            vh_vv_ratio = x0[:,1,:]/x0[:,0,:]
#            vh_vv_ratio = np.expand_dims(vh_vv_ratio, axis=1)
#            x0 = np.concatenate((x0, vh_vv_ratio), 1)

        # for Sentinel-2 use minimum sequence length, randomly selected
        if self.sensor == 'S2':
            indices = list(range(27)) 
            random.shuffle(indices)
            indices = sorted(indices)
            x0 = x0[indices, :,:]

            # get item data
            s2_item_date = self.date_positions[item]
            item_date = [s2_item_date[i] for i in indices] #subset 27 dates using same idx
            
        elif self.sensor == 'S1':
            item_date = self.date_positions
        

        if x0.shape[-1] > self.npixel:
            idx = np.random.choice(list(range(x0.shape[-1])), size=self.npixel, replace=False)
            x = x0[:, :, idx]
            mask = np.ones(self.npixel)

        elif x0.shape[-1] < self.npixel:

            if x0.shape[-1] == 0:
                x = np.zeros((*x0.shape[:2], self.npixel))
                mask = np.zeros(self.npixel)
                mask[0] = 1
            else:
                x = np.zeros((*x0.shape[:2], self.npixel))
                x[:, :, :x0.shape[-1]] = x0
                x[:, :, x0.shape[-1]:] = np.stack([x[:, :, 0] for _ in range(x0.shape[-1], x.shape[-1])], axis=-1)
                mask = np.array(
                    [1 for _ in range(x0.shape[-1])] + [0 for _ in range(x0.shape[-1], self.npixel)])
        else:
            x = x0
            mask = np.ones(self.npixel)

        if self.norm is not None:
            m, s = self.norm
            m = np.array(m)
            s = np.array(s)

            if len(m.shape) == 0:
                x = (x - m) / s
            elif len(m.shape) == 1:  # Normalise channel-wise
                x = (x.swapaxes(1, 2) - m) / s
                x = x.swapaxes(1, 2)  # Normalise channel-wise for each date
            elif len(m.shape) == 2:
                x = np.rollaxis(x, 2)  # TxCxS -> SxTxC
                x = (x - m) / s
                x = np.swapaxes((np.rollaxis(x, 1)), 1, 2)
        x = x.astype('float')

        if self.jitter is not None:
            sigma, clip = self.jitter
            x = x + np.clip(sigma * np.random.randn(*x.shape), -1 * clip, clip)

        mask = np.stack([mask for _ in range(x.shape[0])], axis=0)  # Add temporal dimension to mask
        data = (Tensor(x), Tensor(mask))

        if self.extra_feature is not None:
            ef = (self.extra[str(self.pid[item])] - self.extra_m) / self.extra_s
            ef = torch.from_numpy(ef).float()

            ef = torch.stack([ef for _ in range(data[0].shape[0])], dim=0)
            data = (data, ef)

        if self.return_id:
            return data, torch.from_numpy(np.array(y, dtype=int)), Tensor(item_date), self.pid[item] #add return date
            #return data, torch.from_numpy(np.array(y, dtype=int)), self.pid[item] # without return date
        else:
            return data, torch.from_numpy(np.array(y, dtype=int)), Tensor(item_date)
            #return data, torch.from_numpy(np.array(y, dtype=int))


class PixelSetData_preloaded(PixelSetData):
    """ Wrapper class to load all the dataset to RAM at initialization (when the hardware permits it).
    """
    def __init__(self, folder, labels, npixel, sub_classes=None, norm=None,
                 extra_feature=None, jitter=(0.01, 0.05), sensor=None, return_id=False):
        super(PixelSetData_preloaded, self).__init__(folder, labels, npixel, sub_classes, norm, extra_feature, jitter,sensor,
                                                     return_id)
        self.samples = []
        print('Loading samples to memory . . .')
        for item in range(len(self)):
            self.samples.append(super(PixelSetData_preloaded, self).__getitem__(item))
        print('Done !')

    def __getitem__(self, item):
        return self.samples[item]


def parse(date):
    d = str(date)
    return int(d[:4]), int(d[4:6]), int(d[6:])


def interval_days(date1, date2):
    return abs((dt.datetime(*parse(date1)) - dt.datetime(*parse(date2))).days)


def date_positions(dates):
    pos = []
    for d in dates:
        pos.append(interval_days(d, dates[0]))
    return pos


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:56:38 2019

@author: lhjin
"""

import os
import math
import torch
from torch.utils.data import  DataLoader
from skimage import io, transform
import numpy as np
import random 

#import sys
#path = '/home/star/0_code_lhj/DL-SIM-github/Testing_codes/UNet/'
#sys.path.append(path)

from unet_model import UNet
import warnings
warnings.filterwarnings('ignore')

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data_in = sample['image_in']
        name = sample['image_name']
        data_gt = sample['groundtruth']
        return {'image_in': torch.from_numpy(data_in),'groundtruth':data_gt,'image_name':name}

class ReconsDataset(torch.utils.data.Dataset):
     def __init__(self, all_data_path, maximum_intensity_4normalization_path, transform, training_dataset, in_size, train_in_size):
        self.all_data_path = all_data_path
        self.maximum_intensity_4normalization_path = maximum_intensity_4normalization_path
        self.transform = transform
        self.in_size = in_size
        self.training_dataset = training_dataset
        self.dirs_data = os.listdir(self.all_data_path)
        self.train_in_size = train_in_size
        self.dirs_testing  = self.dirs_data
        
     def __len__(self):
         if self.training_dataset:
             dirs = self.dirs_training 
         else:
             dirs = self.dirs_testing
         return len(dirs)          

     def __getitem__(self, idx):
         if self.training_dataset:
             self.dirs = self.dirs_training 
         else:
             self.dirs = self.dirs_testing
         
         max_intensity = np.load(self.maximum_intensity_4normalization_path,allow_pickle=True)
         max_HE_SRRF = max_intensity[0]['objValue']
         max_HE      = max_intensity[2]['objValue']
         print(max_HE)
         file_name = os.path.join(self.all_data_path, self.dirs[idx])
         data_all = np.load(file_name)
         
         data_gt = data_all[:,:,0]
         
         train_in_size = self.train_in_size
         
         data_in = np.zeros((self.in_size, self.in_size, train_in_size))
         for i in range(train_in_size):
             data_in[:,:,i] = data_all[:,:,i+2]
         data_in = data_in/max_HE
         
         sample = {'image_in': data_in, 'groundtruth':data_gt,'image_name':self.dirs[idx]}
         
         if self.transform:
             sample = self.transform(sample)
         return sample

def get_learning_rate(epoch):
    limits = [3, 8, 12]
    lrs = [1, 0.1, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
        return lrs[-1] * learning_rate


if __name__ == "__main__":
    cuda = torch.device('cuda:0')
    learning_rate = 0.001
    # momentum = 0.99
    # weight_decay = 0.0001
    batch_size = 1
    input_size= 5
    maximum_intensity_path="/home/star/0_code_lhj/DL-SIM-github/Testing_codes/UNet/Max_intensity.npy"
    SRRFDATASET = ReconsDataset(all_data_path="/home/star/0_code_lhj/DL-SIM-github/TESTING_DATA/SRRF_microtubule/",
                                maximum_intensity_4normalization_path=maximum_intensity_path,
                                transform = ToTensor(),
                                training_dataset = False,
                                in_size = 320,
                                train_in_size = input_size)   
   
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=False, pin_memory=True) # better than for loop

    model = UNet(n_channels=input_size, n_classes=1)
    
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    model.load_state_dict(torch.load("/home/star/0_code_lhj/DL-SIM-github/MODELS/UNet_SRRF_microtubule.pkl"))
    model.eval()
    max_intensity = np.load(maximum_intensity_path,allow_pickle=True)
    max_HE_SRRF = max_intensity[0]['objValue']
    max_HE      = max_intensity[2]['objValue']
    for batch_idx, items in enumerate(test_dataloader):
        
        image = items['image_in']
        image1 = items['image_in']*max_HE
        image_gt = items['groundtruth']
        image_name = items['image_name']
        print(image_name[0])
        model.train()
        
        image = np.swapaxes(image, 1,3)
        image = np.swapaxes(image, 2,3)
        image = image.float()
        image = image.cuda(cuda)  
        
        pred = model(image)
        pred = pred*max_HE_SRRF
        print(image1.shape)
#        mv = pred.flatten().min()
#        if mv < 0:
#            pred = pred + abs(mv)
        io.imsave('/home/star/0_code_lhj/DL-SIM-github/Testing_codes/UNet/prediction/' +image_name[0][:-4]+ '_in.tif',    image1[0,:,:,0].detach().cpu().numpy().astype(np.uint32))
        io.imsave('/home/star/0_code_lhj/DL-SIM-github/Testing_codes/UNet/prediction/' +image_name[0][:-4]+ '_gt.tif',    image_gt[0].detach().cpu().numpy().astype(np.uint32))
        io.imsave('/home/star/0_code_lhj/DL-SIM-github/Testing_codes/UNet/prediction/' +image_name[0][:-4] + '_pred.tif', pred.detach().cpu().numpy().astype(np.uint32))

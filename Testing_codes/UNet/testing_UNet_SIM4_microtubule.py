#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:51:14 2019

@author: lhjin
"""

import os
import math
import torch
from torch.utils.data import  DataLoader
from skimage import io, transform
import numpy as np

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
        return {'image_in': torch.from_numpy(data_in),'image_name':name}

class ReconsDataset(torch.utils.data.Dataset):
     def __init__(self, test_in_path, transform, img_type,in_size):
        self.test_in_path = test_in_path
        self.transform = transform
        self.img_type = img_type
        self.in_size = in_size
        self.dirs_in = os.listdir(self.test_in_path)
     def __len__(self):
        dirs = os.listdir(self.test_in_path)   # open the files
        return len(dirs)            # because one of the file is for groundtruth

     def __getitem__(self, idx): 
         train_in_size = 4
         
         data_in = np.zeros((self.in_size, self.in_size, train_in_size))
         filepath = os.path.join(self.test_in_path, self.dirs_in[idx])
         for i in range(train_in_size-1):
             ii=i*5
             if ii <= 9:
                 image_name = os.path.join(filepath, "HE_0"+str(ii)+"." + self.img_type)
             else:
                image_name = os.path.join(filepath, "HE_"+str(ii)+"." + self.img_type)
             image = io.imread(image_name)
             data_in[:,:,i] = image
         ii =15
         image_name = os.path.join(filepath, "HE_"+str(ii)+"." + self.img_type)
         image = io.imread(image_name)
         data_in[:,:,3] = image
         max_in = 5315.0
         data_in = data_in/max_in
         
         sample = {'image_in': data_in,'image_name':self.dirs_in[idx]}
         
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
    
    SRRFDATASET = ReconsDataset(test_in_path="/home/star/0_code_lhj/DL-SIM-github/TESTING_DATA/microtuble/HE_X2/",
                                transform = ToTensor(),
                                img_type = 'tif',
                                in_size = 256)
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=True, pin_memory=True) # better than for loop

    model = UNet(n_channels=4, n_classes=1)
    
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    model.load_state_dict(torch.load("/home/star/0_code_lhj/DL-SIM-github/MODELS/UNet_SIM4_microtubule.pkl"))
    model.eval()
    
    for batch_idx, items in enumerate(test_dataloader):
        
        image = items['image_in']
        image_name = items['image_name']
        print(image_name[0])
        model.train()
        
        image = np.swapaxes(image, 1,3)
        image = np.swapaxes(image, 2,3)
        image = image.float()
        image = image.cuda(cuda)  
        
        pred = model(image)
        max_out = 15383.0
        pred = pred*max_out
#        mv = pred.flatten().min()
#        if mv < 0:
#            pred = pred + abs(mv)
        io.imsave('/home/star/0_code_lhj/DL-SIM-github/Testing_codes/UNet/prediction/' +image_name[0] + '_pred.tif', pred.detach().cpu().numpy().astype(np.uint32))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:37:36 2019

@author: lhjin
"""
from xlwt import *
import numpy as np
import os
import torch
from torch.utils.data import  DataLoader
from skimage import io, transform
import random 

import sys
path = '/home/star/0_code_lhj/DL-SIM-github/Training_codes/UNet/'
sys.path.append(path)

from unet_model import UNet

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data_in, data_out = sample['image_in'], sample['groundtruth']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        #landmarks = landmarks.transpose((2, 0, 1))
        
        #return {'image': image, 'landmarks': torch.from_numpy(landmarks)}
        return {'image_in': torch.from_numpy(data_in),
               'groundtruth': torch.from_numpy(data_out)}

class ReconsDataset(torch.utils.data.Dataset):
     def __init__(self, all_data_path, maximum_intensity_4normalization_path, transform, training_dataset, in_size, train_in_size):
        self.all_data_path = all_data_path
        self.maximum_intensity_4normalization_path = maximum_intensity_4normalization_path
        self.transform = transform
        self.in_size = in_size
        self.training_dataset = training_dataset
        self.dirs_data = os.listdir(self.all_data_path)
        self.train_in_size = train_in_size
        
        self.dirs_data.sort()  # make sure that the filenames have a fixed order before shuffling
        random.seed(1000)
        random.shuffle(self.dirs_data) # shuffles the ordering of filenames (deterministic given the chosen seed)
        
        split_1 = int(0.9 * len(self.dirs_data))
        self.dirs_training = self.dirs_data[:split_1]
        self.dirs_testing  = self.dirs_data[split_1+1:]
        
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
         
         file_name = os.path.join(self.all_data_path, self.dirs[idx])
         data_all = np.load(file_name)
         
         data_gt = data_all[:,:,0]/max_HE_SRRF
         
         train_in_size = self.train_in_size
         
         data_in = np.zeros((self.in_size, self.in_size, train_in_size))
         for i in range(train_in_size):
             data_in[:,:,i] = data_all[:,:,i+2]
         data_in = data_in/max_HE
         
         sample = {'image_in': data_in, 'groundtruth': data_gt}
         
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

def val_during_training(dataloader):
    model.eval()
    
    loss_all = np.zeros((len(dataloader)))
    for batch_idx, items in enumerate(dataloader):
        image = items['image_in']
        gt = items['groundtruth']
        
        image = np.swapaxes(image, 1,3)
        image = np.swapaxes(image, 2,3)
        image = image.float()
        image = image.cuda(cuda) 
        
        gt = gt.squeeze()
        gt = gt.float()
        gt = gt.cuda(cuda)
        
        pred = model(image).squeeze()
        loss0 =(pred-gt).abs().mean()
       
        loss_all[batch_idx] = loss0.item()
        
    mae_m, mae_s = loss_all.mean(), loss_all.std()
    return  mae_m, mae_s


if __name__ == "__main__":
    cuda = torch.device('cuda:0')
    learning_rate = 0.001
    # momentum = 0.99
    # weight_decay = 0.0001
    batch_size = 1
    input_size = 5
    output_size = 1
    SRRFDATASET = ReconsDataset(all_data_path="/media/star/LuhongJin/UNC_data/SRRF/New_training_20190829/0NPY_Dataset/Dataset/Microtubule/",
                                maximum_intensity_4normalization_path="/home/star/0_code_lhj/DL-SIM-github/Training_codes/UNet/Max_intensity.npy",
                                transform = ToTensor(),
                                training_dataset = True,
                                in_size = 320,
                                train_in_size = input_size)
    train_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=True, pin_memory=True) # better than for loop
    
    SRRFDATASET2 = ReconsDataset(all_data_path="/media/star/LuhongJin/UNC_data/SRRF/New_training_20190829/0NPY_Dataset/Dataset/Microtubule/",
                                maximum_intensity_4normalization_path="/home/star/0_code_lhj/DL-SIM-github/Training_codes/UNetMax_intensity.npy",
                                transform = ToTensor(),
                                training_dataset = False,
                                in_size = 320,
                                train_in_size = input_size)
    validation_dataloader = torch.utils.data.DataLoader(SRRFDATASET2, batch_size=batch_size, shuffle=True, pin_memory=True) # better than for loop

    model = UNet(n_channels=input_size, n_classes=output_size)

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))

    loss_all = np.zeros((2000, 4))
    for epoch in range(2000):
        
        mae_m, mae_s = val_during_training(train_dataloader)
        loss_all[epoch,0] = mae_m
        loss_all[epoch,1] = mae_s
        mae_m, mae_s = val_during_training(validation_dataloader) 
        loss_all[epoch,2] = mae_m
        loss_all[epoch,3] = mae_s
        
        file = Workbook(encoding = 'utf-8')
        table = file.add_sheet('loss_all')
        for i,p in enumerate(loss_all):
            for j,q in enumerate(p):
                table.write(i,j,q)
        file.save('/home/star/0_code_lhj/DL-SIM-github/Training_codes/UNet/loss_UNet_SRRF_microtubule.xls')

        lr = get_learning_rate(epoch)
        for p in optimizer.param_groups:
            p['lr'] = lr
            print("learning rate = {}".format(p['lr']))
            
        for batch_idx, items in enumerate(train_dataloader):
            
            image = items['image_in']
            gt = items['groundtruth']
            
            model.train()
            
            image = np.swapaxes(image, 1,3)
            image = np.swapaxes(image, 2,3)
            image = image.float()
            image = image.cuda(cuda)    
            
            gt = gt.squeeze()
            gt = gt.float()
            gt = gt.cuda(cuda)
            
            pred = model(image).squeeze()

            loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print ("[Epoch %d] [Batch %d/%d] [loss: %f]" % (epoch, batch_idx, len(train_dataloader), loss.item()))

        if epoch % 50 == 49:
            torch.save(model.state_dict(), "/home/star/0_code_lhj/DL-SIM-github/Training_codes/UNet/UNet_SRRF_microtubule_"+str(epoch+1)+".pkl")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 19:13:11 2019

@author: lhjin
"""
from xlwt import *
import numpy as np
import os
import math
import torch
from torch.utils.data import  DataLoader
from skimage import io, transform

#import sys
#path = '/home/star/0_code_lhj/DL-SIM-github/Training_codes/UNet/'
#sys.path.append(path)
from unet_model import UNet



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data_in, data_out = sample['image_in'], sample['groundtruth']
        
        return {'image_in': torch.from_numpy(data_in),
               'groundtruth': torch.from_numpy(data_out)}

class ReconsDataset(torch.utils.data.Dataset):
     def __init__(self, data_in_path,data_gt_path, transform, img_type, in_size):
        self.data_in_path = data_in_path
        self.data_gt_path = data_gt_path
        self.transform = transform
        self.img_type = img_type
        self.in_size = in_size
        self.dirs_gt = os.listdir(self.data_gt_path)
     def __len__(self):
        dirs_gt = os.listdir(self.data_gt_path)   # open the files
        return len(dirs_gt)            # because one of the file is for groundtruth

     def __getitem__(self, idx):       
         train_in_size = 15
         data_in = np.zeros((self.in_size, self.in_size, train_in_size))
         filepath = os.path.join(self.data_in_path, self.dirs_gt[idx])
         for i in range(train_in_size):  
             if i < 10:
                 image_name = os.path.join(filepath, "LE_0"+str(i)+'.'+ self.img_type)
             else:
                 image_name = os.path.join(filepath, "LE_"+str(i)+'.'+ self.img_type)
             image = io.imread(image_name)
             data_in[:,:,i] = image
         max_in = 196.0
         data_in = data_in/max_in
         data_gt = np.zeros((self.in_size, self.in_size, train_in_size))
         filepath = os.path.join(self.data_gt_path, self.dirs_gt[idx])
         for i in range(train_in_size):    
             if i < 10:
                 image_name = os.path.join(filepath, "HE_0"+str(i)+'.'+ self.img_type)
             else:
                 image_name = os.path.join(filepath, "HE_"+str(i)+'.'+ self.img_type)
             image = io.imread(image_name)
             data_gt[:,:,i] = image
         max_out = 5315.0
         data_gt = data_gt/max_out
         sample = {'image_in': data_in, 'groundtruth': data_gt}
         
         if self.transform:
             sample = self.transform(sample)
        
         return sample

def get_learning_rate(epoch):
    limits = [2, 6, 8]
    lrs = [1, 0.1, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
        return lrs[-1] * learning_rate

def val_during_training(dataloader):
    model.eval()
    
    loss_all = 0 
    for batch_idx, items in enumerate(dataloader):
        image = items['image_in']
        gt = items['groundtruth']
        
        image = np.swapaxes(image, 1,3)
        image = np.swapaxes(image, 2,3)
        image = image.float()
        image = image.cuda(cuda) 
        
        pred = model(image)
        
        gt = np.swapaxes(gt, 1,3)
        gt = np.swapaxes(gt, 2,3)
        gt = gt.float()
        gt = gt.cuda(cuda)
        
        loss = (pred - gt).abs().mean()
        loss_all += loss.item()
        
    mae = loss_all/len(dataloader)
    return  mae

if __name__ == "__main__":
#    mydata = Datarange()
#    intensity_v = mydata.maximum_intensity()
#    print(intensity_v['max_in'])
#    print(intensity_v['max_out'])
    cuda = torch.device('cuda:0')
    learning_rate = 0.001
    # momentum = 0.99
    # weight_decay = 0.0001
    batch_size = 1
    
    SRRFDATASET = ReconsDataset(data_in_path="/media/star/LuhongJin/UNC_data/SIM/ALL_data/microtubule/Training_Testing/LE_X2/",
                                data_gt_path="/media/star/LuhongJin/UNC_data/SIM/ALL_data/microtubule/Training_Testing/HE_X2/",
                                transform = ToTensor(),
                                img_type = 'tif',
                                in_size = 256)
    train_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=True, pin_memory=True) 
    
    SRRFDATASET2 = ReconsDataset(data_in_path="/media/star/LuhongJin/UNC_data/SIM/ALL_data/microtubule/Training_Testing/testing_LE_X2/",
                                data_gt_path="/media/star/LuhongJin/UNC_data/SIM/ALL_data/microtubule/Training_Testing/testing_HE_X2/",
                                transform = ToTensor(),
                                img_type = 'tif',
                                in_size = 256)
    validation_dataloader = torch.utils.data.DataLoader(SRRFDATASET2, batch_size=batch_size, shuffle=True, pin_memory=True) # better than for loop
    
    model = UNet(n_channels=15, n_classes=15)

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,  betas=(0.9, 0.999))

    loss_all = np.zeros((2000, 2))
    for epoch in range(2000):
        mae_m = val_during_training(train_dataloader)
        loss_all[epoch,0] = mae_m
        mae_m = val_during_training(validation_dataloader) 
        loss_all[epoch,1] = mae_m
        
        file = Workbook(encoding = 'utf-8')
        table = file.add_sheet('loss_all')
        for i,p in enumerate(loss_all):
            for j,q in enumerate(p):
                table.write(i,j,q)
        file.save('/home/star/0_code_lhj/DL-SIM-github/Training_codes/UNet/loss_UNet_SNR_microtubule.xls')

        lr = get_learning_rate(epoch)
        for p in optimizer.param_groups:
            p['lr'] = lr
            print("learning rate = {}".format(p['lr']))
            
        for batch_idx, items in enumerate(train_dataloader):
            #item = iter(train_dataloader)
            #items = item.next()
            image = items['image_in']
            gt = items['groundtruth']
            
            model.train()
            
            image = np.swapaxes(image, 1,3)
            image = np.swapaxes(image, 2,3)
            #print(image.shape)
            image = image.float()
            image = image.cuda(cuda)    
    
            pred = model(image)
#            pred = pred.squeeze()
#            print(pred.shape)
            
            gt = np.swapaxes(gt, 1,3)
            gt = np.swapaxes(gt, 2,3)
            gt = gt.float()
            gt = gt.cuda(cuda)
#            print(gt.shape)
            #loss = (pred - gt).mean() + 5.0 *(((pred - gt)**2).mean())
            loss = (pred-gt).abs().mean() + 5 * ((pred-gt)**2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print ("[Epoch %d] [Batch %d/%d] [loss: %f]" % (epoch, batch_idx, len(train_dataloader), loss.item()))

        torch.save(model.state_dict(), "/home/star/0_code_lhj/DL-SIM-github/Training_codes/UNet/UNet_SNR_microtubule.pkl")

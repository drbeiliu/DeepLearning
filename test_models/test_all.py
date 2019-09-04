#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import torch
from torch.utils.data import  DataLoader
from skimage import transform,io
from PIL import Image
import numpy as np

from models.unet_model import UNet
import warnings
warnings.filterwarnings('ignore')

model_dict = {
    '''name:[".pkl name(not adding date)", data_in_normalization, data_out_normalization, train_in_size, mix_in_image(AVG or MAX)]'''
    "LE_HE":["LE_HE",206.0,8029.0,15,None],
    "LE_HE_enlarge":["LE_HE",190.0,8029.0,15,None],
    "LE_HER":["LE_X2_HER",190.0,8029.0,15,None],
    "HE_HER": ["HE_X2_HER",5542.0,8029.0,15,None],
    "HE_3_HER":["HE_X2_3f_HER",5542.0,8029.0,3,None],
    "HE_4_AVG_HER":["HE_X2_4fAVG_HER",5542.0,8029.0,4,'AVG'],
    "HE_4_MAX_HER":["HE_X2_4fMAX_HER",5542.0,8029.0,4,'MAX']
}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data_in = sample['image_in']
        name = sample['image_name']
        # pylint: disable=E1101
        return {'image_in': torch.from_numpy(data_in),'image_name':name}
        # pylint: enable=E1101

class ReconsDataset(torch.utils.data.Dataset):
    def __init__(self, img_dict, transform, normalization,in_size, img_type='.tif',train_in_size=15, mix_in=None):
        self.img_dict = img_dict
        self.transform = transform
        self.img_type = img_type
        self.in_size = in_size
        self.train_in_size = train_in_size
        self.normalization = normalization
        self.mix_in = mix_in
        self.keys = list(img_dict.keys())
        
    def __len__(self):
           # open the files
        return len(self.img_dict)            # because one of the file is for groundtruth
    def __getitem__(self, idx): 
        data_in = np.zeros((self.in_size, self.in_size, self.train_in_size))
        if self.train_in_size < 15:
            for i in range(3):
                image = self.img_dict[self.keys[idx]][i*5]
                data_in[:,:,i]=image
        else:
            for i in range(self.train_in_size):
                image = self.img_dict[self.keys[idx]][i]
                data_in[:,:,i] = image
        if self.mix_in is not None:
            if self.mix_in == "AVG":
                image = np.sum(self.img_dict[self.keys[idx]],axis=0)/len(self.img_dict[self.keys[idx]])
                data_in[:,:,self.train_in_size-1] = image
            elif self.mix_in == "MAX":
                image = np.max(self.img_dict[self.keys[idx]],axis=0)
                data_in[:,:,self.train_in_size-1] = image
        data_in = data_in/self.normalization

        sample = {'image_in': data_in,'image_name':self.keys[idx]}
         
        if self.transform:
            sample = self.transform(sample)
        
        return sample

def modelPredictHER(state_dict_path,dataloader,out_norm,n_channels,n_classes=1):
    # pylint: disable=E1101
    cuda = torch.device('cuda:0')
    # pylint: enable=E1101
    model = UNet(n_channels, n_classes)

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    results = {}
    for batch_idx, items in enumerate(dataloader):
        
        image = items['image_in']
        image_name = items['image_name']
        model.train()
        
        image = np.swapaxes(image, 1,3)
        image = np.swapaxes(image, 2,3)
        image = image.float()
        image = image.cuda(cuda)  
        
        pred = model(image)
        pred = pred*8029.0

        results[image_name[0]] = pred.detach().cpu().numpy().astype(np.uint32)
    return results

def loadDatasetDict(dir_path):
    dataset_dict={}
    for samples in os.listdir(dir_path):
        sample_path = os.path.join(dir_path, samples)
        if os.path.isdir(sample_path):
            names = os.listdir(sample_path)
            if samples not in dataset_dict:
                dataset_dict[samples] =[]
            for name in names:
                dataset_dict[samples].append(imgread(os.path.join(sample_path,name)))
    return dataset_dict

def imgread(file):
    return np.array(Image.open(file))

def HERtest(name, afterLEPred,batch_size=1):
    

    if name not in model_dict:
        print("model name not found! Please use following: ", model_dict.keys())
    else:
        stat = model_dict.get(name) #normalization=stat[1], train_in_size=stat[3],mix_in=stat[4]
        if name == "LE_HER":
            input_path = os.path.join(os.getcwd(),"dataset","LE_X2")
            dataset_dict = loadDatasetDict(input_path)
        else:
            if afterLEPred:
                dataset_dict = LE_HEtest(enlarged=True,batch_size=batch_size) 
            else:
                input_path = os.path.join(os.getcwd(),"dataset","HE_X2")
                dataset_dict = loadDatasetDict(input_path)
        SRRFDATASET = ReconsDataset(
            img_dict = dataset_dict,
            transform=ToTensor(),
            normalization=stat[1],
            in_size = 256,
            train_in_size=stat[3],
            mix_in=stat[4]
        )
        test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size,shuffle=True,pin_memory=True)
        stat_dict_path = os.path.join(os.getcwd(),"models",stat[0]+"_0825.pkl")
        results = modelPredictHER(stat_dict_path,test_dataloader,stat[2],stat[3])
        if afterLEPred:
            save_path = os.path.join(os.getcwd(),"predictions",name+"_adterLEPred")
        else:
            save_path = os.path.join(os.getcwd(),"predictions",name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for name, image in results.items():
            io.imsave(os.path.join(save_path,name+'_pred.tif'),image)


def LE_HEtest(enlarged,batch_size=1):
    if enlarged:
        dir_path = os.path.join(os.getcwd(),"dataset","LE_X2")
        state_dict_path = os.path.join(os.getcwd(),"models",model_dict["LE_HE_enlarge"][0]+"_0825.pkl")
        save_path = os.path.join(os.getcwd(),"predictions","LE_HE","pred_X2")
        in_norm = model_dict["LE_HE_enlarge"][1]
        out_norm =  model_dict["LE_HE_enlarge"][2]
        in_size = 256
    else:
        dir_path = os.path.join(os.getcwd(),"dataset","LE")
        state_dict_path = os.path.join(os.getcwd(),"models",model_dict["LE_HE"][0]+"_0825.pkl")
        save_path = os.path.join(os.getcwd(),"predictions","LE_HE","pred")
        in_norm = model_dict["LE_HE"][1]
        out_norm =  model_dict["LE_HE"][2]
        in_size = 128
    data_dic = loadDatasetDict(dir_path)
    SRRFDATASET = ReconsDataset(
        img_dict = data_dic,
        transform=ToTensor(),
        normalization=in_norm,
        in_size = in_size

    )
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size,shuffle=True,pin_memory=True)
    # pylint: disable=E1101
    cuda = torch.device('cuda:0')
    # pylint: enable=E1101
    model = UNet(n_channels=15, n_classes=15)

    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.cuda(cuda)
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()

    result_dict = {}
    
    for batch_idx, items in enumerate(test_dataloader):
        image = items['image_in']
        image_name = items['image_name']
        model.train()
        
        image = np.swapaxes(image, 1,3)
        image = np.swapaxes(image, 2,3)
        image = image.float()
        image = image.cuda(cuda)  
        
        pred = model(image)
        
        filepath = image_name[0]
        sample_path = os.path.join(save_path,filepath)
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        train_in_size = 15

        if filepath not in result_dict:
            result_dict[filepath]=[]
        
        for image_num in range(train_in_size):
            image = pred[0,image_num]*out_norm
#            mv = image.flatten().min()
#            if mv < 0:
#                image = image + abs(mv)
            image_name = os.path.join(sample_path, "HE_{:02}.tif".format(image_num))
            result_dict[filepath].append(np.reshape(image.detach().cpu().numpy(),(in_size,in_size)))
            io.imsave(image_name,image.detach().cpu().numpy().astype(np.uint32))
    return  result_dict
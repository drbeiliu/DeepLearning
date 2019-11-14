#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:37:36 2019

@author: lhjin
"""
import numpy as np
import os
import math
from skimage import io, transform
import PIL


class CropImages(object):
    """
    A class used to crop image
    """
    def __init__(self, in_size,
                 HE_in_path = '/media/lhj/My_Passport/SIM/ALL_data/microtubule/Samples/HE/',
                 LE_in_path = '/media/lhj/My_Passport/SIM/ALL_data/microtubule/Samples/LE/',
                 HER_in_path = '/media/lhj/My_Passport/SIM/ALL_data/microtubule/Samples/HER/',
                 LER_in_path = '/media/lhj/My_Passport/SIM/ALL_data/microtubule/Samples/LER/',
                 HE_crop_path="/media/lhj/My_Passport/SIM/ALL_data/microtubule/Training_Testing/HE/",
                 LE_crop_path="/media/lhj/My_Passport/SIM/ALL_data/microtubule/Training_Testing/LE/",
                 HER_crop_path="/media/lhj/My_Passport/SIM/ALL_data/microtubule/Training_Testing/HER/",
                 LER_crop_path="/media/lhj/My_Passport/SIM/ALL_data/microtubule/Training_Testing/LER/", 
                 img_type="tif"):
        
        self.in_size = in_size # 256*256
    
        file_name = os.listdir(HE_in_path)
        self.train_size = len(file_name) # the number of samples for training
        
        file_name = os.path.join(HE_in_path, "Sample_1")     
        file_name = os.listdir(file_name)   
        self.train_in_size = len(file_name) # the number of images in each sample for training
        
        self.HE_in_path    = HE_in_path
        self.LE_in_path    = LE_in_path
        self.HER_in_path   = HER_in_path
        self.LER_in_path   = LER_in_path
        self.HE_crop_path  = HE_crop_path
        self.LE_crop_path  = LE_crop_path
        self.HER_crop_path = HER_crop_path
        self.LER_crop_path = LER_crop_path
        self.img_type = img_type
        
    def doCropImages(self):
        print("Start Augmentation")
        """
        Start augmentation.....
        """
        in_size = self.in_size
        train_size = self.train_size       # the number of samples for training
        train_in_size = self.train_in_size # the number of raw images for each sample
        
        HE_in_path    = self.HE_in_path
        LE_in_path    = self.LE_in_path
        HER_in_path   = self.HER_in_path
        LER_in_path   = self.LER_in_path
        HE_crop_path  = self.HE_crop_path
        LE_crop_path  = self.LE_crop_path
        HER_crop_path = self.HER_crop_path
        LER_crop_path = self.LER_crop_path
        
        img_type = self.img_type 
        dirs_samples = os.listdir(HE_in_path) 
        
        crop_num = 1;
        for sample_num in range(train_size):
            sample_name = dirs_samples[sample_num]
            
            file_name = os.path.join(HE_in_path, sample_name)
            image_name = os.path.join(file_name, "HE_1."+img_type)
            image = io.imread(image_name)
            h,w = image.shape
            Num_x = math.ceil(h/in_size)
            Num_y = math.ceil(w/in_size)
            
            x_step = math.ceil(h/Num_x)
            y_step = math.ceil(w/Num_y)
            
            i0 = crop_num;
            # Crop HE_data
            file_name = os.path.join(HE_in_path, sample_name)
            for image_num in range(train_in_size):
                image_name = os.path.join(file_name, "HE_"+str(image_num+1)+"." + img_type)
                image = io.imread(image_name)
                crop_num = i0
                for i in range(Num_x):
                    for j in range(Num_y):
                        if i < Num_x-1:
                            i_top = i*x_step
                            i_bottom = i_top+in_size
                        else:
                            i_bottom = h
                            i_top = h-in_size
                        if j < Num_y-1:
                            j_left = j*y_step
                            j_right = j_left+in_size
                        else:
                            j_right = w
                            j_left = w-in_size  
                        I = image[i_top:i_bottom, j_left:j_right]
                        filepath = os.path.join(HE_crop_path, "Sample_"+str(crop_num))
                        if  os.path.isdir(filepath):
                            image_name = os.path.join(filepath, "HE_"+str(image_num+1)+"." + img_type)
                        else:
                            os.mkdir(filepath)
                            image_name = os.path.join(filepath, "HE_"+str(image_num+1)+"." + img_type)
                        io.imsave(image_name,I.astype(np.uint16))
                        
#                        I = transform.resize(I, (in_size*2, in_size*2))
#                        filepath = os.path.join(HE_crop_path, "Sample_"+str(crop_num))
#                        if  os.path.isdir(filepath):
#                            image_name = os.path.join(filepath, "HE_X2_"+str(image_num+1)+"." + img_type)
#                        else:
#                            os.mkdir(filepath)
#                            image_name = os.path.join(filepath, "HE_X2_"+str(image_num+1)+"." + img_type)
#                        io.imsave(image_name,I.astype(np.uint32))
                        crop_num = crop_num+1;
            
            # Crop LE_data
            file_name = os.path.join(LE_in_path, sample_name)
            for image_num in range(train_in_size):
                image_name = os.path.join(file_name, "LE_"+str(image_num+1)+"." + img_type)
                image = io.imread(image_name)
                crop_num = i0
                for i in range(Num_x):
                    for j in range(Num_y):
                        if i < Num_x-1:
                            i_top = i*x_step
                            i_bottom = i_top+in_size
                        else:
                            i_bottom = h
                            i_top = h-in_size
                        if j < Num_y-1:
                            j_left = j*y_step
                            j_right = j_left+in_size
                        else:
                            j_right = w
                            j_left = w-in_size  
                        I = image[i_top:i_bottom, j_left:j_right]
                        #I = transform.resize(I, (1280, 1280),Image.BICUBIC)
                        filepath = os.path.join(LE_crop_path, "Sample_"+str(crop_num))
                        if  os.path.isdir(filepath):
                            image_name = os.path.join(filepath, "LE_"+str(image_num+1)+"." + img_type)
                        else:
                            os.mkdir(filepath)
                            image_name = os.path.join(filepath, "LE_"+str(image_num+1)+"." + img_type)
                        io.imsave(image_name,I.astype(np.uint16))
#                        
#                        I = transform.resize(I, (in_size*2, in_size*2), 3)
#                        filepath = os.path.join(HE_crop_path, "Sample_"+str(crop_num))
#                        if  os.path.isdir(filepath):
#                            image_name = os.path.join(filepath, "LE_X2_"+str(image_num+1)+"." + img_type)
#                        else:
#                            os.mkdir(filepath)
#                            image_name = os.path.join(filepath, "LE_X2_"+str(image_num+1)+"." + img_type)
#                        io.imsave(image_name,I.astype(np.uint16))
                        crop_num = crop_num+1;               
#
            # Crop HER_data
            image_name = os.path.join(HER_in_path, sample_name+"." + img_type)
            image = io.imread(image_name)
            H,W = image.shape
            x_step = math.ceil(H/Num_x)
            y_step = math.ceil(W/Num_y)
            crop_num = i0
            for i in range(Num_x):
                for j in range(Num_y):
                    if i < Num_x-1:
                        i_top = i*x_step
                        i_bottom = i_top+in_size*2
                    else:
                        i_bottom = H
                        i_top = H-in_size*2
                    if j < Num_y-1:
                        j_left = j*y_step
                        j_right = j_left+in_size*2
                    else:
                        j_right = W
                        j_left = W-in_size*2
                    I = image[i_top:i_bottom, j_left:j_right]
                    image_name = os.path.join(HER_crop_path, "Sample_"+str(crop_num)+"." + img_type)
                    io.imsave(image_name,I.astype(np.uint16))
                    crop_num = crop_num+1;

            # Crop LER_data
            image_name = os.path.join(LER_in_path, sample_name+"." + img_type)
            image = io.imread(image_name)
            crop_num = i0
            for i in range(Num_x):
                for j in range(Num_y):
                    if i < Num_x-1:
                        i_top = i*x_step
                        i_bottom = i_top+in_size*2
                    else:
                        i_bottom = H
                        i_top = H-in_size*2
                    if j < Num_y-1:
                        j_left = j*y_step
                        j_right = j_left+in_size*2
                    else:
                        j_right = W
                        j_left = W-in_size*2
                    I = image[i_top:i_bottom, j_left:j_right]
                    image_name = os.path.join(LER_crop_path, "Sample_"+str(crop_num)+"." + img_type)
                    io.imsave(image_name,I.astype(np.uint16))
                    crop_num = crop_num+1;                    

if __name__ == "__main__":
    mycrop = CropImages(128)    # to crop big image into small pieces
    mycrop.doCropImages()

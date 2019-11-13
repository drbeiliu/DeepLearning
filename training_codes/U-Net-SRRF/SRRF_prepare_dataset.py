#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:29:40 2019

@author: zfq
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:37:36 2019

@author: lhjin
"""
import numpy as np
import os
import math
from skimage import io

class CropImages(object):
    """
    A class used to crop image
    """
    def __init__(self, in_size,train_in_size,
                 train_in_data_path = '/media/zfq/My_Passport/SIM/ALL_data/mitochondrial/Samples/',
                 data_crop_npy_path="/media/zfq/My_Passport/SIM/ALL_data/mitochondrial/Training_testing_npy/",
                 img_type="tif",
                 input_data_length=36):
        
        self.in_size = in_size # 256*256
    
        file_name = os.listdir(train_in_data_path)
        self.train_size = len(file_name)   # the number of samples for training
    
        self.train_in_size = train_in_size # the number of images in each sample for training
        
        self.train_in_data_path = train_in_data_path
        self.data_crop_npy_path = data_crop_npy_path
        self.img_type           = img_type
        self.input_data_length  = input_data_length
        
    def doCropImages(self):
        in_size = self.in_size
        train_size = self.train_size       # the number of samples for training
        train_in_size = self.train_in_size # the number of raw images for each sample
        
        train_in_data_path  = self.train_in_data_path
        data_crop_npy_path  = self.data_crop_npy_path
        
        img_type = self.img_type 
        input_data_length = self.input_data_length
        dirs_samples = os.listdir(train_in_data_path) 
        
        crop_num = 1;
        HE_SRRF_max = 0;
        LE_SRRF_max = 0;
        HE_max = 0;
        LE_max = 0;
        for sample_num in range(train_size):
            print("Sample_number: "+str(sample_num)+"/"+str(train_size-1))
            sample_name = dirs_samples[sample_num]
            
            file_name = os.path.join(train_in_data_path, sample_name)
            image_name = os.path.join(file_name, "SRRF_100ms"+"." + img_type)
            image = io.imread(image_name)
            H,W = image.shape
            
            Num_x = math.ceil(H/in_size)
            Num_y = math.ceil(W/in_size)
            
            x_step = math.ceil(H/Num_x)
            y_step = math.ceil(W/Num_y)
            
            for i in range(Num_x):
                for j in range(Num_y):
                    if i < Num_x-1:
                        i_top = i*x_step
                        i_bottom = i_top+in_size
                    else:
                        i_bottom = H
                        i_top = H-in_size
                    if j < Num_y-1:
                        j_left = j*y_step
                        j_right = j_left+in_size
                    else:
                        j_right = W
                        j_left = W-in_size  
                    I = image[i_top:i_bottom, j_left:j_right]
                    if I.flatten().max() > 500:
                        data_all = np.zeros((in_size, in_size, input_data_length))
                        data_all[:,:,0] = I
                        file_name = os.path.join(train_in_data_path, sample_name)
                        image_name = os.path.join(file_name, "SRRF_1ms"+"." + img_type)
                        image_t = io.imread(image_name)
                        I = image_t[i_top:i_bottom, j_left:j_right]
                        data_all[:,:,1] = I
                        
                        file_name = os.path.join(train_in_data_path, sample_name)
                        file_name = os.path.join(file_name, "100ms") 
                        for image_num in range(train_in_size):
                            if image_num < 10:
                                image_name = os.path.join(file_name, "HE_0"+str(image_num)+"." + img_type)
                            else:
                                image_name = os.path.join(file_name, "HE_"+str(image_num)+"." + img_type)
                            image_t = io.imread(image_name)
                            I = image_t[i_top:i_bottom, j_left:j_right]
                            data_all[:,:,image_num+2] = I
                            
                        file_name = os.path.join(train_in_data_path, sample_name)
                        file_name = os.path.join(file_name, "1ms") 
                        for image_num in range(train_in_size):
                            if image_num < 10:
                                image_name = os.path.join(file_name, "LE_0"+str(image_num)+"." + img_type)
                            else:
                                image_name = os.path.join(file_name, "LE_"+str(image_num)+"." + img_type)
                            image_t = io.imread(image_name)
                            I = image_t[i_top:i_bottom, j_left:j_right]
                            data_all[:,:,image_num+2+train_in_size] = I
                        
                        if HE_SRRF_max < data_all[:,:,0].flatten().max():
                            HE_SRRF_max = data_all[:,:,0].flatten().max()
                        if LE_SRRF_max < data_all[:,:,1].flatten().max():
                            LE_SRRF_max = data_all[:,:,1].flatten().max()
                        if HE_max < data_all[:,:,2].flatten().max():
                            HE_max = data_all[:,:,2].flatten().max()
                        if LE_max < data_all[:,:,2+train_in_size].flatten().max():
                            LE_max = data_all[:,:,2+train_in_size].flatten().max()
                        filepath = os.path.join(data_crop_npy_path, "Sample_"+str(crop_num)+".npy")
                        np.save(filepath, data_all)
                        crop_num = crop_num+1;

        dtype = np.dtype([('Name', '|S2'), ('objValue', object)])
        data = np.zeros(4, dtype)
        data[0]['objValue'] = HE_SRRF_max
        data[1]['objValue'] = LE_SRRF_max
        data[2]['objValue'] = HE_max
        data[3]['objValue'] = LE_max
        
        data[0]['Name'] = 'HE_SRRF_max'
        data[1]['Name'] = 'LE_SRRF_max'
        data[2]['Name'] = 'HE_max'
        data[3]['Name'] = 'LE_max'
        np.save('/home/zfq/0_lhjin_code/SRRF/Factin/SRRF_in20/Max_intensity.npy', data)

if __name__ == "__main__":
    mycrop = CropImages(320,15)    # to crop big image into small pieces
    mycrop.doCropImages()

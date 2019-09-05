#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:29:40 2019

@author: LHJ
"""

"""
BACKGROUN IMAGES: MAXIMUN INTENSITY < 500

./Training_testing_npy/256/Sample_x.npy = [data[0]['Name'] = 'HER_SIM_256*256'
                                           data[1]['Name'] = 'LER_SIM_256*256'
                                           data[2]['Name'] = 'HE_SIM_256*256_1'
                                                 .
                                                 .
                                                 .
                                          data[16]['Name'] = 'HE_SIM_256*256_15'
                                          data[17]['Name'] = 'HE_SIM_AVG'
                                          data[18]['Name'] = 'HE_SIM_MAP'
                                          data[19]['Name'] = 'LE_SIM_256*256_1'
                                                 .
                                                 .
                                                 .
                                          data[33]['Name'] = 'LE_SIM_256*256_15'
                                          data[34]['Name'] = 'LE_SIM_AVG'
                                          data[35]['Name'] = 'LE_SIM_MAP']

./Training_testing_npy/128/Sample_x.npy = [data[0]['Name'] = 'HE_SIM_128*128_1'
                                                  .
                                                  .
                                                  .
                                          data[14]['Name'] = 'HE_SIM_128*128_15'
                                          data[15]['Name'] = 'LE_SIM_128*128_1'
                                                 
                                                 .
                                                 .
                                          data[29]['Name'] = 'LE_SIM_128*128_15']

SIM/MITO_Max_intensity.npy = [data[0]['Name'] = 'HER_max'
                              data[1]['Name'] = 'LER_max'
                              data[2]['Name'] = 'HE_max'
                              data[3]['Name'] = 'LE_max]
"""
import numpy as np
import os
import math
from skimage import io, transform
#import cv2

class CropImages(object):
    """
    A class used to crop image
    """
    def __init__(self, in_size,train_in_size,
                 train_in_data_path = '/media/zfq/My_Passport/SIM/ALL_data/mitochondrial/Samples/',
                 data_crop_256_npy_path="/media/zfq/My_Passport/SIM/ALL_data/mitochondrial/Training_testing_npy/256/",
                 data_crop_128_npy_path="/media/zfq/My_Passport/SIM/ALL_data/mitochondrial/Training_testing_npy/128/",
                 img_type="tif",
                 input_data_length=36):
        
        self.in_size = in_size # 256*256
    
        file_name = os.listdir(train_in_data_path+'HER')
        self.train_size = len(file_name)   # the number of samples for training
    
        self.train_in_size = train_in_size # the number of images in each sample for training
        
        self.train_in_data_path = train_in_data_path
        self.data_crop_256_npy_path = data_crop_256_npy_path
        self.data_crop_128_npy_path = data_crop_128_npy_path
        self.img_type           = img_type
        self.input_data_length  = input_data_length
        
    def doCropImages(self):
        in_size = self.in_size
        train_size = self.train_size       # the number of samples for training
        train_in_size = self.train_in_size # the number of raw images for each sample
        
        train_in_data_path  = self.train_in_data_path
        data_crop_256_npy_path  = self.data_crop_256_npy_path
        data_crop_128_npy_path  = self.data_crop_128_npy_path
        
        img_type = self.img_type 
        input_data_length = self.input_data_length
        dirs_samples = os.listdir(train_in_data_path+'HER') 
        
        crop_num = 1;
        HER_max = 0;
        LER_max = 0;
        HE_max = 0;
        LE_max = 0;
        for sample_num in range(train_size):
            print("Sample_number: "+str(sample_num)+"/"+str(train_size-1))
            sample_name = dirs_samples[sample_num]
            
            file_name = os.path.join(train_in_data_path+'HER')
            image_name = os.path.join(file_name, sample_name)
            image = io.imread(image_name)
            image = image[50:-50,50:-50]
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
                        data_all_original = np.zeros((int(in_size/2), int(in_size/2), int(train_in_size*2)))
                        data_all[:,:,0] = I
                        file_name = os.path.join(train_in_data_path+'LER')
                        image_name = os.path.join(file_name, sample_name)
                        image_t = io.imread(image_name)
                        image_t = image_t[50:-50,50:-50]
                        I = image_t[i_top:i_bottom, j_left:j_right]
                        data_all[:,:,1] = I
                        
                        file_name = os.path.join(train_in_data_path+'HE')
                        file_name = os.path.join(file_name, sample_name[:-4])
                        for image_num in range(train_in_size):
                            image_name = os.path.join(file_name, "HE_"+str(image_num+1)+"." + img_type)
                            image_t = io.imread(image_name)
                            image_t = transform.resize(image_t,(image_t.shape[0]*2, image_t.shape[1]*2),preserve_range=True,order=3)
#                            image_t = cv2.resize(image_t,(in_size,in_size), interpolation = cv2.INTER_CUBIC)
                            image_t = image_t[50:-50,50:-50]
                            I = image_t[i_top:i_bottom, j_left:j_right]
                            data_all[:,:,image_num+2] = I
                            I = transform.resize(I,(int(in_size/2), int(in_size/2)),preserve_range=True,order=3)
#                            I = cv2.resize(I,(int(in_size/2),int(in_size/2)), interpolation = cv2.INTER_CUBIC)
                            data_all_original[:,:,image_num] = I
                        I = np.mean(data_all[:,:,2:train_in_size+1], axis=2)
                        data_all[:,:,train_in_size+2] = I
                        I = np.max(data_all[:,:,2:train_in_size+1], axis=2)
                        data_all[:,:,train_in_size+3] = I
                        
                        file_name = os.path.join(train_in_data_path+'LE')
                        file_name = os.path.join(file_name, sample_name[:-4])
                        for image_num in range(train_in_size):
                            image_name = os.path.join(file_name, "LE_"+str(image_num+1)+"." + img_type)
                            image_t = io.imread(image_name)
                            image_t = transform.resize(image_t,(image_t.shape[0]*2, image_t.shape[1]*2),preserve_range=True,order=3)
#                            image_t = cv2.resize(image_t,(in_size,in_size), interpolation = cv2.INTER_CUBIC)
                            image_t = image_t[50:-50,50:-50]
                            I = image_t[i_top:i_bottom, j_left:j_right]
                            data_all[:,:,image_num+4+train_in_size] = I
                            I = transform.resize(I,(int(in_size/2), int(in_size/2)),preserve_range=True,order=3)
#                            I = cv2.resize(I,(int(in_size/2),int(in_size/2)), interpolation = cv2.INTER_CUBIC)
                            data_all_original[:,:,image_num+train_in_size] = I
                        I = np.mean(data_all[:,:,train_in_size+4:train_in_size*2+3], axis=2)
                        data_all[:,:,train_in_size*2+4] = I
                        I = np.max(data_all[:,:,train_in_size+4:train_in_size*2+3], axis=2)
                        data_all[:,:,train_in_size*2+5] = I
                     
                        if HER_max < data_all[:,:,0].flatten().max():
                            HER_max = data_all[:,:,0].flatten().max()
                        if LER_max < data_all[:,:,1].flatten().max():
                            LER_max = data_all[:,:,1].flatten().max()
                        if HE_max < data_all[:,:,2].flatten().max():
                            HE_max = data_all[:,:,2].flatten().max()
                        if LE_max < data_all[:,:,4+train_in_size].flatten().max():
                            LE_max = data_all[:,:,4+train_in_size].flatten().max()
                        filepath = os.path.join(data_crop_256_npy_path, "Sample_"+str(crop_num)+".npy")
                        np.save(filepath, data_all)
                        filepath = os.path.join(data_crop_128_npy_path, "Sample_"+str(crop_num)+".npy")
                        np.save(filepath, data_all_original)
                        crop_num = crop_num+1;

        dtype = np.dtype([('Name', '|S2'), ('objValue', object)])
        data = np.zeros(4, dtype)
        data[0]['objValue'] = HER_max
        data[1]['objValue'] = LER_max
        data[2]['objValue'] = HE_max
        data[3]['objValue'] = LE_max
        
        data[0]['Name'] = 'HER_max'
        data[1]['Name'] = 'LER_max'
        data[2]['Name'] = 'HE_max'
        data[3]['Name'] = 'LE_max'
        np.save('/home/zfq/0_lhjin_code/SIM/MITO_Max_intensity.npy', data)

if __name__ == "__main__":
    mycrop = CropImages(256,15)    # to crop big image into small pieces
    mycrop.doCropImages()

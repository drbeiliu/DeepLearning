#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 13:29:40 2019

@author: LHJ
"""


import numpy as np
import os
import math
from skimage import io, transform
#import cv2

class Datarange(object):
    """
    A class used to crop image
    """
    def __init__(self, 
                 data_in_path = '/media/zfq/My_Passport/SIM/Raw_data/ALL_data/microtubule/Training_Testing/HE_X2/',
                 data_out_path= "/media/zfq/My_Passport/SIM/Raw_data/ALL_data/microtubule/Training_Testing/HER/",
                 
                 img_type="tif"):
        
        file_name = os.listdir(data_in_path)
        self.sample_number = len(file_name) # the number of samples for training        
        
        self.data_in_path = data_in_path
        self.data_out_path = data_out_path
        self.img_type = img_type
        
        self.train_data_dir = os.listdir(self.data_in_path)
    def maximum_intensity(self):
        sample_number = self.sample_number       # the number of samples for training
        data_in_path = self.data_in_path
        data_out_path = self.data_out_path
        img_type = self.img_type 
        max_in = 0
        max_out = 0
        for sample_num in range(sample_number):
            filepath = os.path.join(data_in_path, self.train_data_dir[sample_num])                 
            image_name = os.path.join(filepath, "HE_00." + img_type)
            image = io.imread(image_name)
            m_in = image.flatten().max()
            if m_in > max_in:
                max_in = m_in          
            
        for sample_num in range(sample_number):
            filepath = os.path.join(data_out_path, self.train_data_dir[sample_num])  
            image = io.imread(filepath+"."+img_type)
            m_out = image.flatten().max()
            if m_out > max_out:
                max_out = m_out            
        
        max_intenty_value = {'max_in': max_in,'max_out':max_out}
        return max_intenty_value

if __name__ == "__main__":
    mydata = Datarange()
    intensity_v = mydata.maximum_intensity()
    print(intensity_v['max_in'])
    print(intensity_v['max_out'])

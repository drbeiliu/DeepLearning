#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:36:19 2019

@author: ruiyan
"""
from xlwt import *
import numpy as np
import os
import math
from skimage import io, transform
from PIL import Image

def psnr(img1, img2):
    img1 = (img1/np.amax(img1))*255
    img2 = (img2/np.amax(img2))*255
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def nrmse(img_gt, img2, type="sd"):
    mse = np.mean( (img_gt - img2) ** 2 )
    rmse = math.sqrt(mse)
    if type == "sd":
        nrmse = rmse/np.std(img_gt)
    if type == "mean":
        nrmse = rmse/np.mean(img_gt)
    if type == "maxmin":
        nrmse = rmse/(np.max(img_gt) - np.min(img_gt))
    if type == "iq":
        nrmse = rmse/ (np.quantile(img_gt, 0.75) - np.quantile(img_gt, 0.25))
    if type not in ["mean", "sd", "maxmin", "iq"]:
        print("Wrong type!")
    return nrmse


#def ssim(img1, img2, data_range = None):
#    #if img2.min() < 0:
#    #   img2 += abs(img2.min())
#    img2 = (img2/img2.max()) * img1.max()
#    #img1 = (img1/img1.max()) * 255
#    if data_range is None:
#        score = compare_ssim(img1, img2)
#    else:
#        score = compare_ssim(img1, img2, data_range = data_range)
#    return score

if __name__ == "__main__":
    train_in_path="/.../microtubule/prediction/"
    train_gt_path="/.../microtubule/HER/"
    data_save = "/.../"
    dirs = os.listdir(train_gt_path) 
    for i in range(len(train_in_path_all)):
        train_in_path = train_in_path_all[i]
        p_value = np.zeros((len(dirs), 2))
        for idx in range(len(dirs)):
            image_name = os.path.join(train_gt_path, dirs[idx])
            data_gt = io.imread(image_name)
            if i < 5:
                image_name = os.path.join(train_in_path, dirs[idx][:-4]+'_pred.tif')
            else:
                image_name = os.path.join(train_in_path, dirs[idx][:-4]+'_pred_pred.tif')
            data_in = Image.open(image_name)
            data_in = np.array(data_in)
            
            
            min_v = np.quantile(data_gt, 0.01)
            max_v = np.quantile(data_gt, 0.998)
            data_gt = (data_gt - min_v)/(max_v - min_v)
            
            min_v = np.quantile(data_in, 0.01)
            max_v = np.quantile(data_in, 0.998)
            data_in = (data_in - min_v)/(max_v - min_v)
            
            
            p_value[idx,0] = psnr(data_gt, data_in)
            p_value[idx,1] = nrmse(data_gt, data_in)
        
        file = Workbook(encoding = 'utf-8')
        table = file.add_sheet('performance')
        for k,p in enumerate(p_value):
            for j,q in enumerate(p):
                    table.write(k,j,q)
        file.save(data_save+str(i)+'.xls')
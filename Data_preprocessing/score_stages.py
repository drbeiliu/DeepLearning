#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:36:19 2019

@author: ruiyan
"""
import os
import math
import torch
from torch.utils.data import  DataLoader
from skimage import transform,io
from PIL import Image
import numpy as np
from skimage.measure import compare_ssim
import warnings
import pandas as pd

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



def ssim(img1, img2, data_range = None):
    
    #if img2.min() < 0:
    #   img2 += abs(img2.min())
    
    img2 = (img2/img2.max()) * img1.max()
    #img1 = (img1/img1.max()) * 255
    
    if data_range is None:
        score = compare_ssim(img1, img2)
    else:
        score = compare_ssim(img1, img2, data_range = data_range)
    return score

def getFileNames(dirname,num):
    l = sorted(os.listdir("prediction"))
    file_name = sorted(l[1:],key=lambda x: int(os.path.splitext(x[num:])[0]))
    return file_name



def score4all():

    psnr_all = []
    ssim_all = []
    nrmse_all = []

    her_all = sorted (os.listdir("HER")) #get all the image names in HER
    pred_all = getFileNames("prediction",11)
    
    sample_names = [sample[:-4] for sample in her_all][1:]
    
    for pred in pred_all[:]:
        psnr_ = []
        nrmse_ = []
        ssim_ = []
        for name in her_all[1:]:
            pred_path = os.path.join("prediction",pred, name[:-4]+"_pred.tif")
            her_path = os.path.join("HER",name)
            
            #read the image
            her =  Image.open(her_path)
            her = np.array(her)
            min = np.quantile(her, 0.01)
            max = np.quantile(her, 0.998)
            her = (her - min)/(max - min)
            
            her_pred =  Image.open(pred_path)
            her_pred = np.array(her_pred)
            min = np.quantile(her_pred, 0.01)
            max = np.quantile(her_pred, 0.998)
            her_pred = (her_pred - min)/(max - min)
            
            #calculate scores
            psnr_.append(psnr(her, her_pred))
            nrmse_.append(nrmse(her, her_pred))
            ssim_.append(ssim(her, her_pred, data_range = her.max()))
            
        psnr_all.append(psnr_)
        nrmse_all.append(nrmse_)
        ssim_all.append(ssim_)

    
    return sample_names, pred_all[1:], psnr_all, nrmse_all, ssim_all



# Create a Pandas dataframe from the data.
sample_names, pred_names, psnr_all, nrmse_all, ssim_all = score4all()

dic_psnr = {}
dic_psnr["Samples"] = sample_names
for pred, psnr in zip(pred_names,psnr_all):
    dic_psnr[pred] = psnr
df_psnr = pd.DataFrame(dic_psnr)

dic_nrmse = {}
dic_nrmse["Samples"] = sample_names
for pred, nrmse in zip(pred_names,nrmse_all):
    dic_nrmse[pred] = nrmse
df_nrmse = pd.DataFrame(dic_nrmse)

dic_ssim = {}
dic_ssim["Samples"] = sample_names
for pred, ssim in zip(pred_names,ssim_all):
    dic_ssim[pred] = ssim

df_ssim = pd.DataFrame(dic_ssim)

writer = pd.ExcelWriter('stages_score3.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df_psnr.to_excel(writer, sheet_name='PSNR')
df_nrmse.to_excel(writer, sheet_name='NRMSE')
df_ssim.to_excel(writer, sheet_name='SSIM')

# Close the Pandas Excel writer and output the Excel file.
writer.save()

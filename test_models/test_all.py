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
    "HE_HER_microtubule_microtubule":["HE_X2_HER",5315.0,15383.0,15,None],
    "HE_HER_microtubule_adhesion_1":["HE_X2_HER",5542.0,8029.0,15,None],
    "HE_HER_microtubule_adhesion_0":["HE_X2_HER",5315.0,15383.0,15,None],
    "HE_HER_microtubule_factin_0":["HE_X2_HER",5315.0,15383.0,15,None],
    "HE_HER_microtubule_factin_1":["HE_X2_HER",9680.0,15383.0,15,None],
    "HE_HER_microtubule_mitotracker_0":["HE_X2_HER",5315.0,15383.0,15,None],
    "HE_HER_microtubule_mitotracker_1":["HE_X2_HER",8345.0,15383.0,15,None],
    "HE_HER_adhesion_adhesion":["HE_X2_HER",5542.0,8029.0,15,None],
    "HE_HER_mitotracker_mitotracker":["HE_X2_HER",5315.0,15383.0,15,None],
    "HE_HER_factin_factin":["HE_X2_HER",5315.0,15383.0,15,None],
    "LE_HE":["LE_HE",206.0,8029.0,15,None],
    "LE_HE_enlarge":["LE_X2_HE_X2",190.0,8029.0,15,None],
    "HE_HER_199":["LE_X2_HE_X2", 8130.0, 15383.0,15,None],
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
    print(cuda)
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
        #pred = pred*15383.0
        pred = pred*out_norm
        results[image_name[0]] = pred.detach().cpu().numpy().astype(np.uint32)
    return results

def loadDatasetDict(dir_path,enlarged):
    dataset_dict={}
    for samples in os.listdir(dir_path):
        sample_path = os.path.join(dir_path, samples)
        if samples not in dataset_dict:
            dataset_dict[samples] =[]
        if os.path.isdir(sample_path):
            names = os.listdir(sample_path)
            for name in names:
                img = imgread(os.path.join(sample_path,name))
                if enlarged:
                    if img.shape[0] == 256:
                        dataset_dict[samples].append(img)
                    elif img.shape[0] == 128:
                        img = transform.resize(img,(img.shape[0]*2, img.shape[1]*2),preserve_range=True,order=3)
                        dataset_dict[samples].append(img)
                else:
                    if img.shape[0] == 128:
                        dataset_dict[samples].append(img)
    print("length of dataset:", len(dataset_dict))
    return dataset_dict

def imgread(file):
    return np.array(Image.open(file))

def HERtest(state_dict_path,input_dir_path,name, in_norm,out_norm,train_in_size=15,mix_in=None,afterLEPred=False,le_dict=None,batch_size=1):
  

        if afterLEPred:
            dataset_dict = le_dict 
        else:
        	dataset_dict = loadDatasetDict(input_dir_path,enlarged=True)
                
        print("loading dataset....")
        SRRFDATASET = ReconsDataset(
            img_dict = dataset_dict,
            transform=ToTensor(),
            normalization=in_norm,
            in_size = 256,
            train_in_size=train_in_size,
            mix_in=mix_in
        )
        test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size,shuffle=True,pin_memory=True)
        print("finished loading dataset...")

        print("start predicting...")
        results = modelPredictHER(state_dict_path,test_dataloader,out_norm = out_norm,n_channels=train_in_size)
        print("finished predicting...")
        if afterLEPred:
            save_path = os.path.join(os.getcwd(),"predictions",name+"_adterLEPred")
        else:
            save_path = os.path.join(os.getcwd(),"predictions",name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for name, image in results.items():
            io.imsave(os.path.join(save_path,name+'_pred.tif'),image)


def LE_HEtest(state_dict_path,input_dir_path, name,in_norm,out_norm,enlarged,batch_size=1):
    if enlarged:
        save_path = os.path.join(os.getcwd(),"predictions","LE_HE","pred_X2")
        in_size = 256
    else:
        save_path = os.path.join(os.getcwd(),"predictions","LE_HE","pred")
        in_size = 128
    data_dic = loadDatasetDict(input_dir_path,enlarged=enlarged)
    SRRFDATASET = ReconsDataset(
        img_dict = data_dic,
        transform=ToTensor(),
        normalization=in_norm,
        in_size = in_size
    )
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size,shuffle=True,pin_memory=True)
    cuda = torch.device('cuda:0')
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
            image_name = os.path.join(sample_path, "HE_{:02}.tif".format(image_num))
            result_dict[filepath].append(np.reshape(image.detach().cpu().numpy(),(in_size,in_size)))
            io.imsave(image_name,image.detach().cpu().numpy().astype(np.uint32))
    return  result_dict



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

def getFileNames(dirname):
    file_name = sorted(os.listdir(dirname))
    return file_name

def score4all_HER(model_name, is_afterLEpred, is_enlarged, gt_path_her,input_path):
    if is_afterLEpred:
        folder1516 = "predictions/LE_HE/pred_X2"
        folder_her_pred = "predictions/"+model_name+"_afterLEpred"
    else:
        folder1516 = input_path
        folder_her_pred = "predictions/"+model_name


    set_names = []
    psnr1 = []
    psnr2 = []
    psnr3 = []
    nrmse1 = []
    nrmse2 = []
    nrmse3 = []
    ssim1 = []
    ssim2 = []
    ssim3 = []

    her_all = getFileNames(gt_path_her) #get all the image names in HER

    for name in her_all:
        #paths for each pair of files
        her_path = os.path.join(gt_path_her,name)
        her_pred_path = os.path.join(folder_her_pred,name[:-4]+"_pred.tif")
        img15_path = os.path.join(folder1516,name[:-4], "HE_15.tif")
        img16_path = os.path.join(folder1516,name[:-4], "HE_16.tif")
        

        #read the image
        her =  Image.open(her_path)
        her = np.array(her)
        min = np.quantile(her, 0.01)
        max = np.quantile(her, 0.998)
        her = (her - min)/(max - min)
        
        her_pred =  Image.open(her_pred_path)
        her_pred = np.array(her_pred)
        min = np.quantile(her_pred, 0.01)
        max = np.quantile(her_pred, 0.998)
        her_pred = (her_pred - min)/(max - min)
        
        img15 =  Image.open(img15_path)
        img15 = np.array(img15)
        min = np.quantile(img15, 0.01)
        max = np.quantile(img15, 0.998)
        img15 = (img15 - min)/(max - min)
        
        img16 =  Image.open(img16_path)
        img16 = np.array(img16)
        min = np.quantile(img16, 0.01)
        max = np.quantile(img16, 0.998)
        img16 = (img16 - min)/(max - min)
        
        #calculate scores
        set_names.append(name[:-4])
#       psnr1.append(psnr(her_pred, her_pred))
#       psnr2.append(psnr(img15, her_pred))
#       psnr3.append(psnr(img16, her_pred))
        psnr1.append(psnr(her, her_pred))
        psnr2.append(psnr(her, img15))
        psnr3.append(psnr(her, img16))
#       nrmse1.append(nrmse(her, her_pred))
#       nrmse2.append(nrmse(img15, her_pred))
#       nrmse3.append(nrmse(img16, her_pred))
        nrmse1.append(nrmse(her, her_pred))
        nrmse2.append(nrmse(her, img15))
        nrmse3.append(nrmse(her, img16))
#       ssim1.append(ssim(her, her_pred, data_range = her.max()))
#       ssim2.append(ssim(img15, her_pred, data_range = img15.max()))
#       ssim3.append(ssim(img16, her_pred, data_range = img16.max()))
        ssim1.append(ssim(her, her_pred, data_range = her.max()))
        ssim2.append(ssim(her, img15, data_range = her.max()))
        ssim3.append(ssim(her, img15, data_range = her.max()))
    
    return set_names, psnr1, psnr2, psnr3, nrmse1, nrmse2, nrmse3, ssim1, ssim2, ssim3



def save_as_xlsx(model_name, is_afterLEpred, is_enlarged, gt_path_her, input_path):
    # Create a Pandas dataframe from the data.
    set_names, psnr1, psnr2, psnr3, nrmse1, nrmse2, nrmse3, ssim1, ssim2, ssim3 = score4all_HER(model_name, is_afterLEpred, is_enlarged, gt_path_her, input_path)
    
    df_psnr = pd.DataFrame({'Img': set_names,
                           'With HER_pred': psnr1,
                           'With HE_AVG': psnr2,
                           'With HE_MAX': psnr3})
    df_nrmse = pd.DataFrame({'Img': set_names,
                            'With HER_pred': nrmse1,
                            'With HE_AVG': nrmse2,
                            'With HE_MAX': nrmse3})
    df_ssim = pd.DataFrame({'Img': set_names,
                           'With HER_pred': ssim1,
                           'With HE_AVG': ssim2,
                           'With HE_MAX': ssim3})
    

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    if not os.path.exists("scores"):
        os.makedirs("scores")
    if is_afterLEpred:
        xlsx_path = os.path.join("scores/"+ model_name +"_afterLEpred")
    else:
        xlsx_path = os.path.join("scores/"+ model_name)
    writer = pd.ExcelWriter(xlsx_path + '.xlsx', engine='xlsxwriter')
    

    # Convert the dataframe to an XlsxWriter Excel object.
    df_psnr.to_excel(writer, sheet_name='PSNR')
    df_nrmse.to_excel(writer, sheet_name='NRMSE')
    df_ssim.to_excel(writer, sheet_name='SSIM')
    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

def save_as_xlsx_le_he(model_name, gt_path, is_enlarged):
    

    her_all = getFileNames(gt_path)
    set_names = []
    psnr_ = []
    nrmse_ = []
    ssim_ = []
    
    for sample in her_all:
        if is_enlarged or (model_name != "LE_HE") :
            he_path = os.path.join(gt_path, sample,"HE_15.tif")
        else:
            he_path = os.path.join(gt_path, sample,"HE_15.tif")
        
        if model_name == "LE_HE":
            pred_path = "LE_HE/pred"
            name = "LE_HE"
        else:
            pred_path = "LE_HE/pred_X2"
            name = "LE_HE_enlarge"


        he_pred_path = os.path.join("predictions",pred_path, sample,"HE_15.tif")
                    
        
        he = Image.open(he_path)
        he = np.array(he)
        he_pred = Image.open(he_pred_path)
        he_pred = np.array(he_pred)
        
        set_names.append(sample)
        psnr_.append(psnr(he,he_pred))
        nrmse_.append(nrmse(he,he_pred))
        ssim_.append(ssim(he,he_pred))

    df = pd.DataFrame({'Img': set_names,
                      'PSNR': psnr_,
                      'NRMSE': nrmse_,
                      'SSIM': ssim_})
    
    writer = pd.ExcelWriter("scores/"+ name + '.xlsx', engine='xlsxwriter')
    
    
    # Convert the dataframe to an XlsxWriter Excel object.
    df.to_excel(writer, sheet_name='HE_AVG')

    
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

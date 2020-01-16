import numpy as np
import os
import math
from skimage import io, transform
from PIL import Image
from torch.utils.data import  DataLoader

from unet_model import UNet
import warnings
warnings.filterwarnings('ignore')

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data_in = sample['image_in']
        name = sample['image_name']
        return {'image_in': torch.from_numpy(data_in),'image_name':name}

class ReconsDataset(torch.utils.data.Dataset):
    def __init__(self, img_dict, transform, in_norm, img_type,in_size,train_in_size=15):
        self.img_dict = img_dict
        self.transform = transform
        self.img_type = img_type
        self.in_size = in_size
        self.train_in_size = train_in_size
        self.in_norm = in_norm
    def __len__(self):
        return len(self.img_dict)            
    def __getitem__(self, idx): 
        data_in = np.zeros((self.in_size, self.in_size, self.train_in_size))
        for i in range(self.train_in_size):
            image = self.img_dict[idx][i]
            data_in[:,:,i] = image
        data_in = data_in/self.in_norm

        sample = {'image_in': data_in,'image_name':idx}
         
        if self.transform:
            sample = self.transform(sample)
        
        return sample
def crop_prepare(img, step, out_size):
    tiles = []
    for x in range(0, img.shape[0], step):
        for y in range(0, img.shape[1], step):
            if x+out_size <= img.shape[0] and y+out_size <= img.shape[1]:
                tiles.append(img[x:x+out_size,y:y+out_size])
    return tiles
def get_division_maxtrix(img_shape, step, out_size):
    division_maxtrix = np.zeros((img_shape,img_shape))
    for x in range(0, img_shape, step):
        for y in range(0, img_shape, step):
            if x+out_size <= img_shape and y+out_size <= img_shape:
                division_maxtrix[x:x+out_size,y:y+out_size]+=np.ones((out_size,out_size))
    return division_maxtrix

def cropImage(image, h_step,w_step):
    h,w = image.shape
    tiles = [image[row:row+h_step,column:column+w_step] 
    for row in range(0, h, h_step) for column in range(0, w, w_step)]
    return tiles
def imgread(path):
    img = Image.open(path)
    return np.array(img)

if __name__ == "__main__":
    cuda = torch.device('cuda:0')
    IMG_SIZE = 128
    CROP_STEP = 100
    IMG_SHAPE = (512,512)
    TRAIN_IN_SIZE = 15 #SNR:15 Unet:15 or 3 ScUnet: 15 
    #
    #CHANGE parameters above and below to use.
    #dir_path: path of directory
    #in_norm, out_norm: please use the correct max intensities 
    #				that you used in training for your dataset
    dir_path = ""

    SNR_model_path = ""
    SIM_UNET_model_path = ""
	LE_in_norm = 196.0
	HE_in_norm = 5315.0
	out_norm = 15383.0

    LE_img = imgread(os.path.join(dir_path_LE, "LE_01.tif"))

    LE_512 = cropImage(LE_img, IMG_SHAPE[0],IMG_SHAPE[1])
    sample_le = {}
    for le_512 in LE_512:
        tiles = crop_prepare(le_512, CROP_STEP, IMG_SIZE)
        for n,img in enumerate(tiles):
            if n not in sample_le:
                sample_le[n] = []
            img = transform.resize(img,(IMG_SIZE*2, IMG_SIZE*2),preserve_range=True,order=3)
            sample_le[n].append(img)

	SNR_model = UNet(n_channels=15, n_classes=15)
	print("{} paramerters in total".format(sum(x.numel() for x in SNR_model.parameters())))
	SNR_model.cuda(cuda)
	SNR_model.load_state_dict(torch.load(SNR_model_path))
	# SNR_model.load_state_dict(torch.load(os.path.join(dir_path,"model","LE_HE_mito","LE_HE_0825.pkl")))
	SNR_model.eval()

	SIM_UNET = UNet(n_channels=15, n_classes=1)
	print("{} paramerters in total".format(sum(x.numel() for x in SIM_UNET.parameters())))
	SIM_UNET.cuda(cuda)
	SIM_UNET.load_state_dict(torch.load(SIM_UNET_model_path))
	# SIM_UNET.load_state_dict(torch.load(os.path.join(dir_path,"model","HE_HER_mito","HE_X2_HER_0825.pkl")))
	SIM_UNET.eval()

    SRRFDATASET = ReconsDataset(
    img_dict=sample_le,
    transform=ToTensor(),
    in_norm = LE_in_norm,
    img_type=".tif",
    in_size=256
    )
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=False, pin_memory=True) 
    result = np.zeros((256,256,len(SRRFDATASET),TRAIN_IN_SIZE))
    for batch_idx, items in enumerate(test_dataloader):
        image = items['image_in']
        image_idx = items['image_name']

        image = np.swapaxes(image, 1,3)
        image = np.swapaxes(image, 2,3)
        image = image.float()
        image = image.cuda(cuda)  

        pred = SNR_model(image)
        for image_num in range(TRAIN_IN_SIZE):
            image = pred[0,image_num]*out_norm
            image = image.detach().cpu().numpy()
            result[:,:,image_idx,image_num] = np.reshape(image,(256, 256))
	sample_he = {}
    for j in range(TRAIN_IN_SIZE):
        for k in range(len(SRRFDATASET)):
            img = result[:,:,k,j]
            if k not in sample_he:
                sample_he[k]=[]
            sample_he[k].append(img)
    SRRFDATASET = ReconsDataset(
    img_dict=sample_he,
    transform=ToTensor(),
    in_norm = HE_in_norm,
    # in_norm = 8345.0,
    img_type=".tif",
    in_size=256
    )
    test_dataloader = torch.utils.data.DataLoader(SRRFDATASET, batch_size=batch_size, shuffle=True, pin_memory=True) # better than for loop
    result=np.zeros((256,256,len(SRRFDATASET)))
    for batch_idx, items in enumerate(test_dataloader):
        image = items['image_in']
        image_idx = items['image_name']

        image = np.swapaxes(image, 1,3)
        image = np.swapaxes(image, 2,3)
        image = image.float()
        image = image.cuda(cuda)  

        pred = SIM_UNET(image)
        pred = pred*15383.0
        img = pred.detach().cpu().numpy()
        result[:,:,image_idx] = np.reshape(img,(256,256))
    print(result.shape)
    #combine

    result_img=np.zeros((1000,1000))
    i=0
    for x in range(0, 1000, 200):
        for y in range(0, 1000, 200):
            if x+200 <= 1000 and y+200 <= 1000:
                act_img = result[:,:,i][28:28+200,28:28+200]
                result_img[x:x+200,y:y+200]+= act_img
                i+=1

    io.imsave(os.path.join(dir_path,"Sample_01.tif"), result_img.astype(np.uint32))
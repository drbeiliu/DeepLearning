# Deep learning enables structured illumination microscopy with low light levels and enhanced speed
Structured illumination microscopy (SIM) [1](https://onlinelibrary.wiley.com/doi/full/10.1046/j.1365-2818.2000.00710.x), [2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2397368/)  is one of most popular SR techniques in biological studies. Typically, it requires 9 (2D) or 15 (3D) images to compute a high resolution image. The microscope has to be well charactorized and the parameters for reconstrution need to be fine tune to avoid artifacts during reconstrution [3](https://www.sciencedirect.com/science/article/pii/S003040181831054X?via%3Dihub). During to the complexity, there are not many ready-to-use, open-source packages serving the reconstruction purpose [4](https://academic.oup.com/bioinformatics/article/32/2/318/1744618), [5](https://ieeexplore.ieee.org/document/7400963), [6](https://www.nature.com/articles/ncomms10980). 

Super-resolution optical fluctuation imaging (SOFI) [7](https://www.ncbi.nlm.nih.gov/pubmed/20018714), [8](https://www.ncbi.nlm.nih.gov/pubmed/20940780) , Bayesian analysis of the blinking and bleaching (3B analysis)[9](https://www.ncbi.nlm.nih.gov/pubmed/22138825?dopt=Abstract&holding=npg), [10](https://www.nature.com/articles/nmeth.2342)  and super-resolution radial fluctuations (SRRF)[11](https://www.nature.com/articles/ncomms12471)  are pure computational analysis based approaches to retrieve high frequency information from the time serial image data. They are independent with the imaging platform and are compatible with most of the probes. However, they retrieve SR information by analyzing the spatial-temporal fluctuation from 200~1000 timelapse data, limiting the temporal resolution.

We adopt deep learning to serve the purpose of SIM (DL-SIM) and SRRF (DL-SRRF) reconstruction, particularly with reduced number of frames. We could also restore high resolution information from raw data with extreme low photon budgets. 

   
Specifically, we have trained 4 models with differnet input and groud truth:

|Model #             |Input                        |           Ground truth    |
|---                 |---                          |--- |
|1. U-Net-SIM15      |fifteen SIM raw data                  | single SIM reconstruction |
|2. U-Net-SIM3       |three SIM raw data                    | single SIM reconstruction |
|3. scU-Net          |fifteen SIM raw data (low light)       | single SIM reconstruction (normal light) |
|4. U-Net-SRRF5      |five TIRF images              | SRRF reconstruction from 200 frames |

## Keys

1. DL-SIM and DL-SRRF serve as an engine to perform SIM and SRRF reconstructions
2. U-Net-SIM could do SIM reconstruction with as few as 3 frames, instead of 9 or 15 frames
3. U-Net-SRRF could do SRRF with as low as 5 frames, instead of 200 frames
4. scU-Net  could recover images from raw data with extreme low photon budgets (low SNR)

All models have been trained with four different cellular structures, including **microtubules**, **mitochondrial**, **adhesion structures** and **actin filaments**. 

## Folders organization

|Folder              |Discription   |
| --- | --- |                   
|1. Data_preprocessing| Python codes to prepare datasets, calculates psnr, nrmse, etc|
|2. Fiji-scripts     | ImageJ/Fiji scripts to prepare traning datasets, calculate RSP etc.|
|3. Testing_codes| Codes for the testing of differenct networks|
|4. Testing_data| Raw data for the testing of microtubule networks|
|5. Training_data| Codes for the training of differenct networks|
|6. longleaf-instructions | useful commands to work on longleaf (a Linux-based cluster at UNC)|
|7. test_json| use .json file to configure the training/testing parameters (under construction)
|8. test_models | codes for both training and testing (under construction)|


## System Requirements
DL-SIM and DL-SRRF packages have been tested on both a cluster, a regular PC and Google Colab. 
### Hardware requirements
*Cluster*: we use the Longleaf cluster (Linux-based) on UNC-Chapel Hill campus. The detailed information can be found [here](https://its.unc.edu/research-computing/longleaf-cluster/). 

*PC*: we also tested our code on a Dell workstation (Dell Precision Tower 5810 ):
- OS: Window 10
- Processor:  Intel Xeon E5-1620 V3 @ 3.5 GHz
- Memory: 128 Gb
- Graphics card: Nvidia GeForce GTX1080 Ti, 11 Gb memory

*Google Colab*: we also tested our code on Google colabotory, which allows you to run the code in the cloud. You can access google colab for free with time limitations. [here](https://colab.research.google.com/drive/146rmMBtNqP-vQl_Z3cNaUCfvJlg5Befi#scrollTo=YPZT2dR3o02P). 

### Python Dependencies
- Anaconda3-4.7.12
- Python 3.7
- PyTorch 1.3.0
- Scikit-image 0.14.3
- Xlwt 1.3.0
- Numpy 1.15.4
- PIL 6.1.0
- Pandas 0.23.4


## 0. Installation guide

1. Install Anaconda3 follow the instructions [online](https://www.anaconda.com/distribution/). 
2. Create environment
   ~~~
   conda create -n your_env_name python=3.7
   ~~~
3. Activate the environment and install python dependencies
   
    ~~~
    Source activate your_env_name
    conda install -c pytorch pytorch
	conda install -c anaconda scikit-image
	conda install -c anaconda xlwt
	conda install -c anaconda pil
	conda install -c anaconda pandas
    ~~~
4. Download ImageJ/Fiji
  
   https://imagej.net/Fiji/Downloads

## 1. Prepare training dataset

### data augmentation

In SIM experiments, the size of the raw image stack was 512 × 512 × 15 (w x h x f, width x height x frame). To prepare the input for U-Net-SIM15, the raw stack was cropped into 128 × 128 × 15 (w x h x f) patches. For U-Net-SIM3, only the first phase of three illumination angles were used, producing 128 × 128 × 3 (w x h x f) patches. In SRRF experiment, the original input images were cropped into 64 × 64 × 5 (w x h x f) and the original ground truth images were cropped into 320 × 320 (w x h). Since U-Net requires the width and height of the input images to match the ground truth images, we resized the input dataset using the biocubic interpolation function of Fiji.

*SIM_prepare_dataset.py*: This file is used to do dataset cropping for the SIM experiment.

*SRRF_prepare_dataset.py*: This file is used to do dataset cropping for the SRRF experiment.

### data normalization
We normalized the input images to the maximum intensity (MI) of the whole input dataset and the ground truth images to the MI of the SIM reconstruction dataset. 

*datarange.py*: This file is be used to determine the intensity ranges of your own dataset. The value of max_in and max_out will be used to normalize the datasets.

###	Separate dataset
After dataset augmentation, we obtained 800-1500 samples for different structures, which were then randomly divided into training, validation and testing subsets. Detailed information about each dataset is in Supplementary Table 1 of our manuscript.

## 2. Train a network

### 2.1 U-Net
The details of each network architecture are shown in *unet_model.py* and *unet_parts.py*.

Files below are used for the training of four different networks in the paper:

    1. training_U-Net-SIM3.py;
    2. training_U-Net-SIM15.py; 
    3. training_U-Net-SNR.py; 
    4. training_U-Net-SRRF.py

Files below are used for the testing of four different networks in the paper:

    1. testing_U-Net-SIM3.py; 
    2. testing_U-Net-SIM15.py;
    3. testing_U-Net-SNR.py; 
    4. testing_U-Net-SRRF.py: 
**please modify file pathes and data ranges in the code before use**  

how to --->

 *train_U-Net-SIM3.py*, *train_U-Net-SIM15.py* and *train_U-Net-SNR.py* :
 
 In class *ReconsDataset(*torch.utils.data.Dataset*)* , change the value of max_out, max_in and train_in_size. The value of  train_in_size is the number of channels of the input. Before use *train _U-Net-SRRF.py*, please use *SRRF_prepare_dataset.py* to generate the Max_intensity.npy for your dataset. 

### 2.2 scU-Net

The details of each network architecture are shown in *unet_model.py* and *unet_parts.py*.

Files below are used for the training of scU-Net in the paper:

    1. training_scU-Net.py 
**please modify the value of max_out, max_in and train_in_size**

Files below are used for the testing of scU-Net in the paper:

    1. testing_scU-Net.py 
**please modify the value of max_out, max_in and train_in_size**

## 3.Quantification of the training performance

RSP and RSE were introduced before to assess the quality of super-resolution data and were calculated using NanoJ-NanoJ-SQUIRREL (https://bitbucket.org/rhenriqueslab/nanoj-squirrel/wiki/Home). 

The resolution of each cropped image was estimated using the ImageDecorrleationAnalysis plugin in Fiji/ImageJ with the default parameter settings. [12](https://www.nature.com/articles/s41592-019-0515-7)

*Peak signal-to-noise ratio (PSNR)*, *normalized root-mean-square error (NRMSE)* and *structural similarity index (SSIM)* were calcualted with a home-writtin script (*performance.py*). 

## 4. Run the demo

**set up the enviroment before use**

### 4.1 Run the training code
    Step 1: download the code from the folder Training_codes
    Step 2: prepare the training dataset
    Step 3: modify the file path and data range (normalization)
    Step 4: open the terminal 
    Step 5: run: source activate your_env_name
    Step 6: cd /file_path_for_training_code
    Step 7: run: python train_***.py
    Step 8: check the model

### 4.2 Run the testing cod
    Step 1: download the code from the “Testing_codes”
    Step 2: download the data from the “Testing_data”
    Step 3: modify the file path and data range (normalization)
    Step 4: open the terminal
    Step 5: run: source activate your_env_name
    Step 6: cd /file_path_for_testing_code
    Step 7: run: python testing_***.py
    Step 8: check the prediction images.
    Step 9: Expected output was prepared in the folder “Testing_results”


## 5. Time estimation
Installing and configuring the python enviroment take about 1 hour. It may vary depending on the speed of the network. 

Typically, training a model on a “normal” desktop computer takes around 2 days for 2000 epoch. It may vary depending on the sample size, batch size and the frequency to save the intermediate models. 

Reconstructing a sample on a “normal” desktop computer takes about 1 second (not counting in the time to load the model).
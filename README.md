# Deep learning enables structured illumination microscopy with low light levels and enhanced speed
Structured illumination microscopy (SIM) [1](https://onlinelibrary.wiley.com/doi/full/10.1046/j.1365-2818.2000.00710.x), [2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2397368/)  is one of most popular SR techniques in biological studies. Typically, it requires 9 (2D) or 15 (3D) images to compute a high resolution image. The microscope has to be well charactorized and the parameters for reconstrution need to be fine tune to avoid artifacts during reconstrution [3](https://www.sciencedirect.com/science/article/pii/S003040181831054X?via%3Dihub). During to the complexity, there are not many ready-to-use, open-source packages serving the reconstruction purpose [4](https://academic.oup.com/bioinformatics/article/32/2/318/1744618), [5](https://ieeexplore.ieee.org/document/7400963), [6](https://www.nature.com/articles/ncomms10980). 

Super-resolution optical fluctuation imaging (SOFI) [7](https://www.ncbi.nlm.nih.gov/pubmed/20018714), [8](https://www.ncbi.nlm.nih.gov/pubmed/20940780) , Bayesian analysis of the blinking and bleaching (3B analysis)[9](https://www.ncbi.nlm.nih.gov/pubmed/22138825?dopt=Abstract&holding=npg), [10](https://www.nature.com/articles/nmeth.2342)  and super-resolution radial fluctuations (SRRF)[11](https://www.nature.com/articles/ncomms12471)  are pure computational analysis based approaches to retrieve high frequency information from the time serial image data. They are independent with the imaging platform and are compatible with most of the probes. However, they retrieve SR information by analyzing the spatial-temporal fluctuation from 200~1000 timelapse data, limiting the temporal resolution.

We adopt UNet to serve the purpose of SIM and SRRF reconstruction, particularly with reduced number of frames. We could also restore high resolution information from raw data with extreme low photon budgets. 

1. srUNet serve as an engine to perform SIM and SRRF reconstructions
2. srUNet could do SIM reconstruction with 4 frames, instead of 9 or 15 frames
3. srUNet could do SRRF with as low as 5 frames, instead of 200 frames
4. snrUNet could recover images from raw data with extreme low photon budgets (low SNR)
5. srUNet + snrUNet could restore SR images from low SNR SI raw data or TIRF timelapse data
   
Specifically, we have trained 4 models with differnet input and groud truth:

|Model #             |Input                        |           Ground truth    |
|---                 |---                          |--- |
|1. U-Net-SIM15      |fifteen SIM raw data                  | single SIM reconstruction |
|2. U-Net-SIM3       |three SIM raw data                    | single SIM reconstruction |
|3. scU-Net          |fifteen SIM raw data (low light)       | single SIM reconstruction (normal light) |
|4. U-Net-SRRF5      |five TIRF images              | SRRF reconstruction from 200 frames |


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
|8. test_models | codes for both training and testing|





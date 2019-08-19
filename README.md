# Ultra-fast, superresolution (SR) imaging facilited by deep learning
Structured illumination microscopy (SIM) is one of most popular SR techniques in biological studies [1](https://onlinelibrary.wiley.com/doi/full/10.1046/j.1365-2818.2000.00710.x), [2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2397368/). Typically, it requires 9 (2D) or 15 (3D) images to compute a high resolution image. The microscope has to be well charactorized and the parameters for reconstrution need to be fine tune to avoid artifacts during reconstrution [3](https://www.sciencedirect.com/science/article/pii/S003040181831054X?via%3Dihub). During to the complexity, there are not many ready-to-use, open-source packages serving the reconstruction purpose [4](https://academic.oup.com/bioinformatics/article/32/2/318/1744618), [5](https://ieeexplore.ieee.org/document/7400963), [6](https://www.nature.com/articles/ncomms10980). 

Super-resolution optical fluctuation imaging (SOFI) , Bayesian analysis of the blinking and bleaching (3B analysis)  and super-resolution radial fluctuations (SRRF)  are pure computational analysis based approaches to retrieve high frequency information from the time serial image data. They are independent with the imaging platform and are compatible with most of the probes. However, they retrieve SR information by analyzing the spatial-temporal fluctuation from 200~1000 timelapse data, limiting the temporal resolution.

We adopt UNet to serve the purpose of SIM and SRRF reconstruction. Particularly, we can restore high resolution information from raw data with extreme low photon budgets. 

1. suUNet to retrieve SR images from structured illumination (SI) raw data
2. suUNet to retrieve SR images from TIRF timelapse data
3. snrUNet to recover images from raw data with extreme low photon budgets (low SNR)
4. suUNet + snrUNet to restore SR images from low SNR SI raw data or TIRF timelapse data


# Software package
request library:
- pyTorch (>=xxxx)
- a
- b
- c
- d
- 
## preprocessing
- [x] cropping
- [x] resizing (bilinear, bicubic)
- [x] normalization
- [ ] remove pure background 
## training
- [x] optimizer (Adam)

## prediction
- [x] input 64x64
- [ ] input arbitrary size image

## validataion
- [x] PSNR
- [ ] FCR
- [ ] ???
- [ ] ???



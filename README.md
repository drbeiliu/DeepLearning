# Ultra-fast, superresolution (SR) imaging facilited by deep learning

1. suUNet to retrieve SR images from structured illumination (SI) raw data
2. suUNet to retrieve SR images from TIRF timelapse data
3. snrUNet to recover images from raw data with extreme low photo budgets (low SNR)
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



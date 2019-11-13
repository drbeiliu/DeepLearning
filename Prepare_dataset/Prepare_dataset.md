## Data format
#### SIM data format
BACKGROUN IMAGES: MAXIMUN INTENSITY < 500
```python
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
                              data[3]['Name'] = 'LE_max]'
```

#### SRRF data format
BACKGROUN IMAGES: MAXIMUN INTENSITY < 500
```python
./microtubule/Sample_x.npy = [data[0]['Name'] = 'HER_SRRF_320*320'
                              data[1]['Name'] = 'LER_SRRF_320*320'
                              data[2]['Name'] = 'HE_SRRF_320*320_1'
                                                 .
                                                 .
                                                 .
                              data[21]['Name'] = 'HE_SRRF_320*320_20'
                              data[22]['Name'] = 'LE_SRRF_320*320_1'
                                                 .
                                                 .
                                                 .
                              data[41]['Name'] = 'LE_SRRF_320*320_20']
./SRRF/Microtubule_max_intensity.npy = [data[0]['Name'] = 'HER_max'
                                    data[1]['Name'] = 'LER_max'
                                    data[2]['Name'] = 'HE_max'
                                    data[3]['Name'] = 'LE_max]'
```

**Don't forget to change filepath**


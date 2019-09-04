#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from test_all import LE_HEtest,HERtest,model_dict

parser = argparse.ArgumentParser(description='Adhesion test')
parser.add_argument('--model', type=str, required=True, help='model name to use')
# parser.add_argument('--output_filename', type=str, default="result.tif", help='where to save the output image')
# parser.add_argument('--input_image', type=str, help='input image to use')
parser.add_argument('--batch_size',type=int, default=1, help='batch size')
parser.add_argument('--enlarge', dest='enlarge',action='store_true',help='with use enlarged LE_HE')
parser.add_argument('--notenlarge', dest='enlarge',action='store_false',help='with use enlarged LE_HE')
parser.add_argument('--afterLEpred',dest='afterLEPred',action='store_true',help='HE input of HER pred ')
parser.add_argument('--notafterLEpred',dest='afterLEPred',action='store_False',help='HE input of HER pred ')
parser.set_defaults(afterLEPred=True)
parser.set_defaults(enlarge=True)
opt = parser.parse_args()

name = opt.model
batch_size = opt.batch_size

if name == 'LE_HE':
    LE_HEtest(enlarged=opt.enlarge,batch_size=opt.batch_size)
else:
    if opt.model not in model_dict:
        print("Model not found. Availiable: ", model_dict.keys())
    HERtest(name,afterLEPred=opt.afterLEPred,batch_size=opt.batch_size)

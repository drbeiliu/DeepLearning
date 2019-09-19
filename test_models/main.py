#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from test_all import LE_HEtest,HERtest,model_dict

'''
Example of using 15 images HER
python3 main.py --model 'HER_15' --save_name DIR_NAME --model_dict_HE PATH --input_HE PATH --in_norm_HE '8130.0' --out_norm_HE '15383.0' --notafterLEPred
'LE_HE', 'HER_15','HER_3','HER_4_AVG','HER_4_MIX' are possible models names.

'''
parser = argparse.ArgumentParser(description='Adhesion test')
parser.add_argument('--model', type=str, required=True, help='model name to use')
parser.add_argument('--save_name', type=str, required=True, help='name to save to')
parser.add_argument('--model_dict_HE', type=str, help='path of HE_HER model to test')
parser.add_argument('--model_dict_LE', type=str, help='path of LE_HE model to test')
parser.add_argument('--input_LE',type=str, help='path of LE_HE images to test')
parser.add_argument('--input_HE',type=str, help='path of HE_HER images to test')
parser.add_argument('--in_norm_LE',type=float, help='Input normalization of LE')
parser.add_argument('--out_norm_LE',type=float, help='Output normalization of LE')
parser.add_argument('--in_norm_HE',type=float, help='Input normalization of HE')
parser.add_argument('--out_norm_HE',type=float, help='Output normalization of HE')
parser.add_argument('--batch_size',type=int, default=1, help='batch size')
parser.add_argument('--enlarge', dest='enlarge',action='store_true',help='with use enlarged LE_HE')
parser.add_argument('--notenlarge', dest='enlarge',action='store_false',help='not enlarge LE_HE predictions')
parser.add_argument('--afterLEPred',dest='afterLEPred',action='store_true',help='HE input of HER pred ')
parser.add_argument('--notafterLEPred',dest ='afterLEPred',action='store_false',help='use HE input not LEPred HE for HER pred')
parser.set_defaults(afterLEPred=True)
parser.set_defaults(enlarge=True)
opt = parser.parse_args()

name = opt.model
names = ['LE_HE', 'HER_15','HER_3','HER_4_AVG','HER_4_MIX']
batch_size = opt.batch_size

if name == 'LE_HE':
    LE_HEtest(
        state_dict_path = opt.model_dict_LE,
        input_dir_path=opt.input_LE,
        name=opt.save_name,
        in_norm = opt.in_norm_LE,
        out_norm=opt.out_norm_LE,
        enlarged=opt.enlarge,
        batch_size=opt.batch_size)
elif name[:3] == 'HER':
    par = name.split('_')
    assert (len(par)==2 or len(par)==3)
    assert par[0]=='HER' and (par[1]=='3' or par[1]=='4' or par[1]=='15')
    train_in_size = int(par[1])

    if len(par) == 3:
        assert (par[2]=='AVG'or par[2]=='MIX')
        mix_in = par[2]
    else:
        mix_in = None
    if opt.afterLEPred:
        result_dict = LE_HEtest(
            state_dict_path = opt.model_dict_LE,
            input_dir_path=opt.input_LE,
            name=opt.save_name,
            in_norm = opt.in_norm_LE,
            out_norm=opt.out_norm_LE,
            enlarged=opt.enlarge,
            batch_size=opt.batch_size)
    else:
        result_dict = None
        HERtest(
            state_dict_path=opt.model_dict_HE,
            input_dir_path = opt.input_HE,
            name=opt.save_name,
            in_norm = opt.in_norm_HE,
            out_norm=opt.out_norm_HE,
            train_in_size = train_in_size,
            mix_in = mix_in,
            afterLEPred=opt.afterLEPred,
            le_dict = result_dict,
            batch_size=opt.batch_size)
else:
    print("model name not allowed.", names)
# def HERtest(stat_dict_path,input_dir_path，name, in_norm,out_norm,train_in_size=15,mix_in=None,afterLEPred=False,le_dict=None,batch_size=1)
# def LE_HEtest(state_dict_path,input_dir_path,name,in_norm,out_norm,enlarged,batch_size=1)
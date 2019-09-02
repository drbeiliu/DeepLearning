''' 

Purpose:
    set project depedent parameters in a json file and load it while training/testing
How to use
    pythong test_json.py --config_file "path_to_the_configuration_file"

base on:
https://github.com/ksanjeevan/pytorch-project-template

'''

import json as js
from pathlib import Path
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This is a test')
    parser.add_argument('--config_file', type=str, required=True, help='configuraion file of the project')
    opt = parser.parse_args()

    #path_to_json = Path("E:/mitochondrial/Mito-SIM/LE_X2_HE_X2_Batch32/config.json")
    path_to_json = Path(opt.config_file)
    
    with open(path_to_json, "r") as proj_file:

        proj_file_data = js.load(proj_file)

        data_info = proj_file_data['data']

        for i, v in enumerate(data_info):
            print(i, v)
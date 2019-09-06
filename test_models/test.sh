#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --mem=4g
#SBATCH -n 1
#SBATCH -t 05-00:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=bowei@email.unc.edu
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

source activate pytorch


python3 main.py --model "LE_HE" --enlarge
python3 main.py --model "LE_HE" --notenlarge
python3 main.py --model "LE_HER" 
python3 main.py --model "HE_3_HER" --afterLEPred
python3 main.py --model "HE_3_HER" --notafterLEPred
python3 main.py --model "HE_HER" --afterLEPred
python3 main.py --model "HE_HER" --notafterLEPred
python3 main.py --model "HE_4_AVG_HER" --afterLEPred
python3 main.py --model "HE_4_AVG_HER" --notafterLEPred
python3 main.py --model "HE_4_MAX_HER" --afterLEPred
python3 main.py --model "HE_4_MAX_HER" --notafterLEPred

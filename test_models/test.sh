#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05-00:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=bowei@email.unc.edu
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

source activate pytorch

python3 main.py --model 'HER_15' --save_name 'plateau_0916' --model_dict_HE '/pine/scr/b/o/bowei/0916trainlive/HE_HER_Scheduler_0916.pkl' --input_HE '/pine/scr/b/o/bowei/0916trainlive/pred_X2' --in_norm_HE '8130.0' --out_norm_HE '15383.0' --notafterLEPred

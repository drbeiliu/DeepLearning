#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05-00:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=ruiyan@live.unc.edu
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

source activate pytorch

python demo/demo.py --config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
--input input1.jpg input2.jpg \
--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

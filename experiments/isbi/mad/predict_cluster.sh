#! /bin/bash
#SBATCH -A kreshuk
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 0-0:30
#SBATCH -o inference.out
#SBATCH -e inference.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=constantin.pape@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=1080Ti
#SBATCH --gres=gpu:1

module load cuDNN
/g/kreshuk/pape/Work/my_projects/inferno-experiments/experiments/isbi/mad/predict.py /g/kreshuk/pape/Work/data/networks/mad/isbi/isbi_unet_lr_v2 isbi_unet_lr_v2 --algorithm mws

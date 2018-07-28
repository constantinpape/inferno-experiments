#! /bin/bash
#SBATCH -A kreshuk
#SBATCH -N 1
#SBATCH -n 5
#SBATCH -t 1-0:00
#SBATCH -o train.out
#SBATCH -e train.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=constantin.pape@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=1080Ti
#SBATCH --gres=gpu:1

module load cuDNN
/g/kreshuk/pape/Work/my_projects/inferno-experiments/experiments/isbi/mad/train_affs.py /g/kreshuk/pape/Work/data/networks/mad/isbi/isbi_hed_ms_v1 --architecture hed

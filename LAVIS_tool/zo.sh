#!/bin/sh
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

python LAVIS_tool/zo.py --steps 8 --epsilon 0.03 --num_query 100 --output_dir zo_ii --image_dir images --target_dir target_image/samples --annotation_path annotations.txt --num_samples 1000
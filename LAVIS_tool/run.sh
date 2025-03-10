#!/bin/sh
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
CUDA_VISIBLE_DEVICES=3 python LAVIS_tool/zo.py --steps 100 --alpha 0.2 --epsilon 0.03 --sigma 0.01 --output_dir transfer_ii --image_dir images --target_dir target_image/samples/ --annotation_path annotations.txt --num_samples 1000 --method transfer_MF_ii --model_name blip2_opt --model_type pretrain_opt2.7b
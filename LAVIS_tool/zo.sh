#!/bin/sh
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1

python LAVIS_tool/transfer_and_query.py --steps 100 --alpha 1 --epsilon 0.03 --sigma 0.01 --output_dir zo_MF_ii_tt --image_dir transfer_ii_1000_100_1.0_0.03_0.01 --target_dir images --annotation_path annotations.txt --num_samples 1000 --method zo_MF_tt --model_name blip2_opt --model_type pretrain_opt2.7b
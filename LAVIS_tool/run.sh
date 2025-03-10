#!/bin/sh
#SBATCH --gres=gpu:1 # So GPU can dung
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
python LAVIS_tool/_train_adv_img_query.py --input_res 224 --epsilon 2 --steps 50  --output test_query --data_path images --text_path target_annotations.txt --num_query 500 --image_dir images --target_dir target_image/samples --annotation_file annotations.txt --num_samples 5
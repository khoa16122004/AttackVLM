import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image
from lavis.models import load_model_and_preprocess
from torch.utils.data import Dataset
from tqdm import tqdm

@torch.no_grad()
def p(model, image):
    image_ = image.clone()
    samples  = {"image": image_}
    # the input must be scaled but not normalize
    caption  = model.generate(samples, use_nucleus_sampling=True, num_captions=1)
    return caption


def main(args):

    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device="cuda")
    model.eval()
    img = Image.open(args.img_path).convert("RGB")
    img = vis_processors(img).cuda()
    
    caption = p(model, img)
    print(caption)   
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--model_name", default="blip_caption", type=str)
    parser.add_argument("--model_type", default="base_coco", type=str)

    args = parser.parse_args()
    
    main(args)
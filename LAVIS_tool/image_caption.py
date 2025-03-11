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

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    img = img / 255.0
    return img.to(dtype=torch.get_default_dtype())

@torch.no_grad()
def p(model, image):
    image_ = image.clone()
    samples  = {"image": image_}
    # the input must be scaled but not normalize
    caption  = model.generate(samples, use_nucleus_sampling=True, num_captions=1)
    return caption


def main(args):
    transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
                            torchvision.transforms.Resize(size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
                            torchvision.transforms.Lambda(lambda img: to_tensor(img))])
    
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device="cuda")
    model.eval()
    img = Image.open(args.img_path).convert("RGB")
    img = transform(img).cuda().unsqueeze(0)
    
    caption = p(model, img)
    print(caption)   
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--model_name", default="blip_caption", type=str)
    parser.add_argument("--model_type", default="base_coco", type=str)

    args = parser.parse_args()
    
    main(args)
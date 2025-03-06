import argparse
import os
import random
# import clip
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from lavis.models import load_model_and_preprocess


# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"

# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
# ------------------------------------------------------------------ #  

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path")
    parser.add_argument("--adv_path")
    parser.add_argument("--model_name", default="blip_caption", type=str)
    parser.add_argument("--model_type", default="base_coco", type=str)

    args = parser.parse_args()

    DEFAULT_RANDOM_SEED = 2023
    seedEverything()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)
    image = vis_processors["eval"](Image.open(args.img_path).convert("RGB")).cuda()
    adv_image = vis_processors["eval"](Image.open(args.adv_path).convert("RGB")).cuda()
    
    with torch.no_grad():
        samples = {"image": torch.stack([image, adv_image])}
        caption = model.generate(samples, use_nucleus_sampling=True, num_captions=1)
        print(caption)
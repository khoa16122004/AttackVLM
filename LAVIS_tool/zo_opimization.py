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

DEFAULT_RANDOM_SEED = 22520691
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





def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

class CustomDataset(Dataset):
    def __init__(self, annotations_file, image_dir, target_dir, transform=None):
        with open(annotations_file, "r") as f:
            lines = [line.strip().split("\t") for line in f.readlines()]
            self.file_names = [line[0] for line in lines]
            self.gt_txts = [line[1] for line in lines]
            self.tar_txts = [line[2] for line in lines]
            
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform = transform

        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.file_names[idx])
        gt_txt = self.gt_txts[idx]
        tar_txt = self.tar_txts[idx]
        target_path = os.path.join(self.target_dir, self.file_names[idx])

        image = Image.open(image_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        # image_processed = vis_processors["eval"](image)
        # target_image_processed = vis_processors["eval"](target_image)
        # text_processed  = txt_processors["eval"](class_text_all[original_tuple[1]])
        if self.transform:
            image = self.transform(image)
            target_image = self.transform(target_image)
        
        return image, gt_txt, image_path, target_image, tar_txt, target_path


normalize = torchvision.transforms.Compose(
    [   
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]
)
@torch.no_grad()
def p(model, image):
    image_ = image.clone()
    image_ = normalize(image_ / 255.0)
    samples  = {"image": image_}
    caption  = model.generate(samples, use_nucleus_sampling=True, num_captions=1)
    return caption

@torch.no_grad()
def clip_encode_text(txt, clip_model, detach=True):
    text_token = clip.tokenize(txt).to(device)
    target_text_features = clip_model.encode_text(text_token)
    target_text_features = target_text_features / target_text_features.norm(dim=1, keepdim=True)
    if detach == True:
        target_text_features = target_text_features.detach()
    return target_text_features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_index", type=int)
    parser.add_argument("--alpha", default=8, type=float)
    parser.add_argument("--epsilon", default=8, type=float)
    parser.add_argument("--sigma", default=16, type=float)
    parser.add_argument("--num_query", default=100, type=int)
    parser.add_argument("--output_dir", default="zo", type=str)
    parser.add_argument("--image_dir", type=str, help='The folder name contains the original image')
    parser.add_argument("--target_dir", type=str, help="The folder name contains the target image")    
    parser.add_argument("--annotation_path", type=str)
    parser.add_argument("--model_name", default="blip_caption", type=str)
    parser.add_argument("--model_type", default="base_coco", type=str)
    args = parser.parse_args()
    
    seedEverything()
    # ----------------------- Our problem ----------------------
    """
        c_tar: the target_text
        x: a image
        c = p(x): is the predicted caption of x
        g(c): is the txtembedding of c     
        L_x = g(c_tar) * g(p(x)) 
        
        estimate gradient of L_x
    """
    
    os.makedirs(args.output_dir, exist_ok=True)
    # ---------------------- Model --------------------
    clip_img_model_vitb32, _ = clip.load("ViT-L/14", device=device)
    clip_img_model_vitb32.eval()

    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)
    model.eval()
    # ---------------------- Data ---------------------    
    data = CustomDataset(args.annotation_path, args.image_dir, args.target_dir,
                         torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
                                                         torchvision.transforms.Resize(size=(384, 384), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
                                                         torchvision.transforms.Lambda(lambda img: to_tensor(img)),])
                        )
    
    image, gt_txt, image_path, target_image, tar_txt, target_path = data[args.img_index]
    
    image = image.to(device).unsqueeze(0)
    clean_txt = p(model, image)[0]
    clean_txt_embedding = clip_encode_text(clean_txt, clip_img_model_vitb32)
    print("Clean txt: ", clean_txt)
    print("target txt: ", tar_txt)
    # g(c_tar)
    target_feature = clip_encode_text(tar_txt, 
                                      clip_img_model_vitb32)
    
    # original loss
    loss = clean_txt_embedding @ target_feature.T
    print("original loss: ", loss)
    
    
    # x + sigma * noise 
    image_repeat = image.repeat(args.num_query, 1, 1, 1)
    noise = torch.randn_like(image_repeat).sign()
    perturbed_image_repeat = torch.clamp(image_repeat + (args.sigma * noise), 0.0, 255.0)    
    
    # c = p(x + sigma * noise)
    pertubed_txt = p(model, perturbed_image_repeat)
    pertubed_txt_embedding = clip_encode_text(pertubed_txt, clip_img_model_vitb32)
    
    # [g(p(x + sigma * noise)) - g(p(x))] * g(c_tar)
    coefficient = pertubed_txt_embedding - clean_txt_embedding # num_query x 512
    coefficient = (coefficient @ target_feature.T)    # num_query x 1

    pseudo_gradient = coefficient.view(args.num_query, 1, 1, 1) * noise    
    pseudo_gradient = torch.sum(pseudo_gradient, dim=0) / (args.num_query * args.sigma)
    print("Coefficient mean:", coefficient.mean().item(), "std:", coefficient.std().item())
    delta = torch.clamp(args.alpha * pseudo_gradient.sign(), -args.epsilon, args.epsilon)
    print("Delta mean:", delta.mean().item(), "std:", delta.std().item())

    # x + delta
    img_adv = image + delta
    img_adv = torch.clamp(img_adv, 0.0, 255.0)
    print("Image difference:", torch.abs(img_adv - image).mean().item())

    adv_text_feature = clip_encode_text(p(model, img_adv), clip_img_model_vitb32)
    loss = adv_text_feature @ target_feature.T
    
    print("adv Loss: ", loss)
    print("adv txt: ", p(model, img_adv))
    
    # save_image
    basename = os.path.basename(image_path)
    torchvision.utils.save_image(img_adv / 255.0, os.path.join(args.output_dir, basename))


    
if __name__ == "__main__":
    main()
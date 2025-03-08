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
    def __init__(self, annotations_file, image_dir, target_dir, vis_processors):
        with open(annotations_file, "r") as f:
            lines = [line.strip().split("\t") for line in f.readlines()]
            self.file_names = [line[0] for line in lines]
            self.gt_txts = [line[1] for line in lines]
            self.tar_txts = [line[2] for line in lines]
            
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.vis_processors = vis_processors

        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.file_names[idx])
        gt_txt = self.gt_txts[idx]
        tar_txt = self.tar_txts[idx]
        target_path = os.path.join(self.target_dir, self.file_names[idx])

        image = Image.open(image_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")
        image = self.vis_processors["eval"](image)
        target_image = self.vis_processors["eval"](target_image)
       
        
        return image, gt_txt, image_path, target_image, tar_txt, target_path



@torch.no_grad()
def p(model, image):
    image_ = image.clone()
    # image_ = normalize(image_ / 255.0)
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

def FO_Attack(args, image, image_tar, model):
    image_adv = image.clone().detach()
    image_tar_ = image_tar.clone().detach()
    image_adv.requires_grad = True

    for i in tqdm(range(args.steps)):
        image_feauture = blip_image_encoder(image_adv, model)
        image_tar_feauture = blip_image_encoder(image_tar_, model)
        loss = torch.sum(image_feauture * image_tar_feauture)
        loss.backward()

        gradient = image_adv.grad.detach()
        pertubtation = torch.clamp(args.alpha * torch.sign(gradient), -args.epsilon, args.epsilon)

        image_adv.data = torch.clamp(image_adv + pertubtation, 0, 1)

        image_adv.grad.zero_()

    return image_adv, gradient


def ZO_Attack(args, image, image_tar, model):
    image_adv = image.clone().detach()
    image_tar_ = image_tar.clone().detach()

    for i in tqdm(range(args.steps)):
        image_feature = blip_image_encoder(image_adv, model)
        image_tar_feature = blip_image_encoder(image_tar_, model)
        
        image_repeat = image_adv.repeat(args.num_query, 1, 1, 1)
        noise = torch.randn_like(image_repeat) * args.sigma  # Điều chỉnh nhiễu theo sigma
        image_pertubed = torch.clamp(image_repeat + noise, 0, 1)
        image_pertubed_feature = blip_image_encoder(image_pertubed, model)
        
        coeficient = (image_pertubed_feature - image_feature).mean(dim=0)  # Trung bình trên batch
        coeficient = coeficient @ image_tar_feature.T
        gradient = (coeficient.view(1, 1, 1, 1) * noise).mean(dim=0)  # Trung bình để giảm nhiễu
        delta = torch.clamp(gradient, -args.epsilon, args.epsilon)
        image_adv = torch.clamp(image_adv + args.alpha * delta, 0, 1)  # Thêm learning rate để kiểm soát cập nhật

    return image_adv, gradient

def blip_image_encoder(image, model, gradient=True):
    if gradient == True:
        image_feauture = model.forward_encoder({"image": image})[:,0,:]
        image_feauture = image_feauture / image_feauture.norm(dim=1, keepdim=True)
    else:
        with torch.no_grad():
            image_ = image.clone().detach()
            image_feauture = model.forward_encoder({"image": image_})[:,0,:]
            image_feauture = image_feauture / image_feauture.norm(dim=1, keepdim=True)
    return image_feauture

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_index", type=int)
    parser.add_argument("--steps", default=8, type=int)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--epsilon", default=0.5, type=float)
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

    
    os.makedirs(args.output_dir, exist_ok=True)
    # ---------------------- Model --------------------

    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)
    model.eval()
    # ---------------------- Data ---------------------    
    data = CustomDataset(args.annotation_path, args.image_dir, args.target_dir, vis_processors)
    
    image, gt_txt, image_path, target_image, tar_txt, target_path = data[args.img_index]
    basename = os.path.basename(image_path)
    
    image = image.to(device).unsqueeze(0)
    target_image = target_image.to(device).unsqueeze(0)
    print(image.shape, target_image.shape)
    
    print("oriignal loss: ", blip_image_encoder(image, model) @ blip_image_encoder(target_image, model).T)

    # ----------------- FO attack -------------------
    image_adv, fo_gradient = FO_Attack(args, image, target_image, model)
    fo_adv_cap = p(model, image_adv)
    print("Fo adv cap: ", fo_adv_cap)
    print("FO loss: ", blip_image_encoder(image_adv, model) @ blip_image_encoder(target_image, model).T)
    print("FO difference: ", (image_adv - image).mean())
    torchvision.utils.save_image(image_adv, os.path.join(args.output_dir, basename))
    torchvision.utils.save_image(image, os.path.join(args.output_dir, "ori_" + basename))
    torchvision.utils.save_image(target_image, os.path.join(args.output_dir, "tar_" + basename))


    # ------------------- ZO attack -------------------
    image_adv, zo_gradient = ZO_Attack(args, image, target_image, model)
    zo_adv_cap = p(model, image_adv)
    print("Zo adv cap: ", zo_adv_cap)
    print("ZO loss: ", blip_image_encoder(image_adv, model) @ blip_image_encoder(target_image, model).T)
    print("ZO difference: ", (image_adv - image).mean())
    torchvision.utils.save_image(image_adv, os.path.join(args.output_dir, "zo_" + basename))

    print("Differecen perutbation: ", (fo_gradient - zo_gradient).mean())
    print("FO gradient mean: ", fo_gradient.abs().mean().item())
    print("ZO gradient mean: ", zo_gradient.abs().mean().item())

if __name__ == "__main__":
    main()
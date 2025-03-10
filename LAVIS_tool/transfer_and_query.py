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

class CustomDataset(Dataset):
    def __init__(self, annotations_file, image_dir, target_dir, transform=None, num_sample=1000):
        with open(annotations_file, "r") as f:
            lines = [line.strip().split("\t") for line in f.readlines()]
            self.file_names = [line[0] for line in lines][:num_sample]
            self.gt_txts = [line[1] for line in lines][:num_sample]
            self.tar_txts = [line[2] for line in lines][:num_sample]
        
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



def clip_encode_text(txt, clip_model, gradient=False, detach=True):
    text_token = clip.tokenize(txt).cuda()
    if gradient == False:
        with torch.no_grad():
            target_text_features = clip_model.encode_text(text_token)
    else:
        target_text_features = clip_model.encode_text(text_token)
        
    target_text_features = target_text_features / target_text_features.norm(dim=1, keepdim=True)
    if detach == True:
        target_text_features = target_text_features.detach()
    return target_text_features

def clip_encode_image(image, clip_model, gradient=False,detach=True):
        
    if gradient == False:
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
    else:
        image_features = clip_model.encode_image(image)
        
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    if detach == True:
        image_features = image_features.detach()
    return image_features


normalize = torchvision.transforms.Compose(
    [   
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]
)

inverse_normalize = torchvision.transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])


@torch.no_grad()
def p(model, image):
    image_ = image.clone()
    samples  = {"image": image_}
    # the input must be scaled but not normalize
    caption  = model.generate(samples, use_nucleus_sampling=True, num_captions=1)
    return caption

def tt_zo(image, c_clean, c_tar, model, clip_img_model_vitb32, num_query, steps, alpha, epsilon, sigma):
    c_tar_embedding = clip_encode_text(c_tar, clip_img_model_vitb32)

    img_adv = image.clone()
    adv_cap = c_clean
    
    for step in range(steps):
        clean_txt_embedding = clip_encode_text(adv_cap, clip_img_model_vitb32)
        image_repeat = img_adv.repeat(num_query, 1, 1, 1)
        noise = torch.randn_like(image_repeat) * sigma
        perturbed_image_repeat = torch.clamp(image_repeat + noise, 0.0, 1.0)    
        
        pertubed_txt = p(model, perturbed_image_repeat)
        pertubed_txt_embedding = clip_encode_text(pertubed_txt, clip_img_model_vitb32)
        
        coefficient = pertubed_txt_embedding - clean_txt_embedding # num_query x 512
        coefficient = torch.sum(coefficient * c_tar_embedding, dim=1)
        pseudo_gradient = (coefficient.view(num_query, 1, 1, 1) * noise).mean(dim=0) # num_query x 3 x 384 x 384 
        delta = torch.clamp(alpha * pseudo_gradient.sign(), -epsilon, epsilon)
        img_adv = img_adv + delta
        img_adv = torch.clamp(img_adv, 0.0, 1.0)
        adv_cap = p(model, img_adv)        
    
    return inverse_normalize(img_adv), adv_cap[0], c_tar_embedding

def ii_fo(image, tar_image, tar_txt, model, clip_img_model_vitb32, steps, alpha, epsilon):
    tar_txt_embedding = clip_encode_text(tar_txt, clip_img_model_vitb32)
    image_ = image.clone()
    delta = torch.zeros_like(image_, requires_grad=True)
    
    for step in range(steps):
        image_adv = torch.clamp(image + delta, 0., 1.)
        clean_image_embedding = clip_encode_image(image_adv, clip_img_model_vitb32, True, False)
        tar_image_embedding = clip_encode_image(tar_image, clip_img_model_vitb32, True, False)
        loss = torch.sum(clean_image_embedding * tar_image_embedding, dim=1)
        loss.backward()
        gradient = delta.grad.detach()
        delta_data = torch.clamp(delta + alpha * torch.sign(gradient), -epsilon, epsilon)
        delta.data = delta_data
        delta.grad.zero_()
    
    image_adv = inverse_normalize(torch.clamp(image + delta, 0., 1.))

    adv_cap = p(model, image_adv)
    
    return image_adv, adv_cap[0], tar_txt_embedding

def it_fo(image, tar_image, tar_txt, model, clip_img_model_vitb32, steps, alpha, epsilon):
    tar_txt_embedding = clip_encode_text(tar_txt, clip_img_model_vitb32)
    image_ = image.clone()    
    delta = torch.zeros_like(image_adv, requires_grad=True)
    for step in range(steps):
        image_adv = torch.clamp(image_ + delta, 0., 1.)
        clean_image_embedding = clip_encode_image(image_adv, clip_img_model_vitb32, True, False)
        loss = torch.sum(clean_image_embedding * tar_txt_embedding, dim=1)
        loss.backward()
        gradient = delta.grad.detach()
        delta_data = torch.clamp(delta +alpha * torch.sign(gradient), -epsilon, epsilon)
        delta.data = delta_data
        delta.grad.zero_()
    
    
    adv_cap = p(model, inverse_normalize(image_adv))
    return image_adv, adv_cap[0], tar_txt_embedding

# CLIP -> BLIP-2 opt 224
# BLIP -> BLIP-2 caption 384
def main(args):
    output_dir = f"{args.output_dir}_{args.num_query}_{args.steps}_{args.alpha}_{args.epsilon}_{args.sigma}"
    os.makedirs(output_dir, exist_ok=True)
    
    clip_img_model_vitb32, preprocess = clip.load("ViT-B/32", device="cuda")
    clip_img_model_vitb32.eval()
    
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device="cuda")
    model.eval()
    
    alpha, epsilon, sigma = args.alpha, args.epsilon, args.sigma

    if args.method == "zo_MF_tt" or args.method == "clean_image":
        data = CustomDataset(args.annotation_path, args.image_dir, args.target_dir,
                             torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
                                                            torchvision.transforms.Resize(size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
                                                            torchvision.transforms.Lambda(lambda img: to_tensor(img))]),
                             args.num_samples)

    elif args.method == "transfer_MF_ii" or args.method == "transfer_MF_it":
        data = CustomDataset(args.annotation_path, args.image_dir, args.target_dir, preprocess, args.num_samples)
            
    clip_scores = 0
    with open(f"{output_dir}.txt", "w") as f:
        for i in tqdm(range(args.num_samples)):
            image, gt_txt, image_path, target_image, tar_txt, target_path = data[i]
            basename = os.path.basename(image_path)
            torchvision.utils.save_image(inverse_normalize(image), "image.png")

            image = image.cuda()
            image = image.unsqueeze(0)
            target_image = target_image.cuda()
            target_image = target_image.unsqueeze(0)
            
            if args.method == "zo_MF_tt": 
                c_clean = p(model, image)[0]
                image_adv, adv_cap, c_tar_embedding = tt_zo(image, c_clean, tar_txt, model, clip_img_model_vitb32, args.num_query, args.steps, alpha, epsilon, sigma)
            
            elif args.method == "transfer_MF_ii":
                c_clean = p(model, inverse_normalize(image))[0]
                image_adv, adv_cap, c_tar_embedding = ii_fo(image, target_image, tar_txt, model, clip_img_model_vitb32, args.steps, alpha, epsilon)

            elif args.method == "transfer_MF_it":
                c_clean = p(model, inverse_normalize(image))[0]
                image_adv, adv_cap, c_tar_embedding = it_fo(image, target_image, tar_txt, model, clip_img_model_vitb32, args.steps, alpha, epsilon)
                
            elif args.method == "clean_image":
                image_adv = image.clone()
                adv_cap = p(model, image_adv)[0]
                c_tar_embedding = clip_encode_text(tar_txt, clip_img_model_vitb32)


            c_adv_embedding = clip_encode_text(adv_cap, clip_img_model_vitb32)
            clip_score = torch.sum(c_tar_embedding * c_adv_embedding, dim=1)
            clip_scores += clip_score
            if args.method != "clean_image":
                torchvision.utils.save_image(image_adv, os.path.join(output_dir, basename))
                f.write(f"{basename}\t{c_clean}\t{tar_txt}\t{adv_cap}\n")
            break            
    clip_scores = clip_scores / args.num_samples
    print(f"Average clip score: {clip_scores}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=['transfer_MF_ii', 'transfer_MF_it', 'zo_MF_tt', "clean_image"])
    parser.add_argument("--steps", default=8, type=int)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--epsilon", default=0.001, type=float)
    parser.add_argument("--sigma", default=0.01, type=float)
    parser.add_argument("--num_query", default=1000, type=int)
    parser.add_argument("--output_dir", default="zo", type=str)
    parser.add_argument("--image_dir", type=str, help='The folder name contains the original image')
    parser.add_argument("--target_dir", type=str, help="The folder name contains the target image")    
    parser.add_argument("--annotation_path", type=str)
    parser.add_argument("--model_name", default="blip_caption", type=str)
    parser.add_argument("--model_type", default="base_coco", type=str)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()
    
    main(args)
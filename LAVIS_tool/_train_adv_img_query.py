import argparse
import os
import random
import clip
import numpy as np
import torch
import torchvision
from PIL import Image
import wandb
from torch.utils.data import Dataset
from tqdm import tqdm
from lavis.common.gradcam import getAttMap
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


# to replace vis_processor
transform_a = torchvision.transforms.Compose(
    [   
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.Resize(size=(384, 384), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
        torchvision.transforms.Lambda(lambda img: to_tensor(img)),
        # torchvision.transforms.ToTensor(),
    ]
)
transform_b = torchvision.transforms.Compose(
    [   
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.Resize(size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
        torchvision.transforms.Lambda(lambda img: to_tensor(img)),
        # torchvision.transforms.ToTensor(),
    ]
)
normalize = torchvision.transforms.Compose(
    [   
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]
)


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


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
        image_processed = self.transform(image)
        target_image_processed = self.transform(target_image)
        
        return image_processed, gt_txt, image_path, target_image_processed, tar_txt, target_path


def _i2t(args, txt_processors, model, image):
    
    # normalize image here
    image = normalize(image / 255.0)
    
    # generate caption
    if args.model_name == "img2prompt_vqa":
        question = "what is the content of this image?"
        question = txt_processors["eval"](question)
        samples  = {"image": image, "text_input": [question] * (image.size()[0])}
        
        # obtain gradcam and update dict
        samples  = model.forward_itm(samples=samples)
        samples  = model.forward_cap(samples=samples, num_captions=1, num_patches=20)
        caption  = samples['captions']
        for cap_idx, cap in enumerate(caption):
            if cap_idx == 0:
                caption_merged = cap
            else:
                caption_merged = caption_merged + cap
    else:
        samples  = {"image": image}
        # print(samples["image"].shape)
        caption  = model.generate(samples, use_nucleus_sampling=True, num_captions=1)
        caption_merged = caption
    
    return caption_merged




if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()
    # load models for i2t
    parser.add_argument("--model_name", default="blip_caption", type=str)
    parser.add_argument("--model_type", default="base_coco", type=str)
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_samples", default=5, type=int)
    parser.add_argument("--input_res", default='224', type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=1, type=int)
    parser.add_argument("--output", default="temp", type=str)
    parser.add_argument("--data_path", default="temp", type=str)
    parser.add_argument("--text_path", default="temp.txt", type=str)
    parser.add_argument("--image_dir", type=str, help='The folder name contains the original image')
    parser.add_argument("--target_dir", type=str, help="The folder name contains the target image")    
    parser.add_argument("--delta", default="normal", type=str)
    parser.add_argument("--num_query", default=20, type=int)
    parser.add_argument("--num_sub_query", default=5, type=int)
    parser.add_argument("--sigma", default=16, type=float)
    parser.add_argument("--annotation_file", type=str)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default='temp_proj')
    parser.add_argument("--wandb_run_name", type=str, default='temp_run')
    
    args = parser.parse_args()

    # ---------------------- #
    print(f"Loading LAVIS models: {args.model_name}, model_type: {args.model_type}...")
    # load models for i2t
    model, vis_processors, txt_processors = load_model_and_preprocess(name=args.model_name, model_type=args.model_type, is_eval=True, device=device)
    
    # use clip text coder for attack
    clip_img_model_vitb32, _ = clip.load("ViT-B/32", device=device, jit=False)
    print("Done")
    os.makedirs(args.output, exist_ok=True)
    
    # ---------------------- #

    # load clip_model params
    num_sub_query, num_query, sigma = args.num_sub_query, args.num_query, args.sigma
    batch_size    = args.batch_size
    alpha         = args.alpha
    epsilon       = args.epsilon
    
    if args.model_name == 'blip2_opt':
        args.input_res = 224
    else:
        args.input_res = 384

    if args.input_res == 384:
        data = CustomDataset(args.annotation_file, args.image_dir, args.target_dir, transform_a)

    else:
        data = CustomDataset(args.annotation_file, args.image_dir, args.target_dir, transform_b)

    
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=24)

    # # org text/features
    # adv_vit_text_path = args.text_path
    # with open(os.path.join(adv_vit_text_path), 'r') as f:
    #     lavis_text_of_adv_vit  = f.readlines()[:args.num_samples] # num_samples
    #     print("org text: ", lavis_text_of_adv_vit)
    #     f.close()
    
    # # adv_vit_text_feautes: 
    # with torch.no_grad():
    #     adv_vit_text_token    = clip.tokenize(lavis_text_of_adv_vit).to(device)
    #     adv_vit_text_features = clip_img_model_vitb32.encode_text(adv_vit_text_token)
    #     adv_vit_text_features = adv_vit_text_features / adv_vit_text_features.norm(dim=1, keepdim=True)
    #     adv_vit_text_features = adv_vit_text_features.detach() # z_clean = g(c_clean)
    #     # print("Text groundtruth shape: ", adv_vit_text_features.shape) # num_samples x 512

    # tgt text/features
    tgt_text_path = 'target_annotations.txt'
    with open(os.path.join(tgt_text_path), 'r') as f:
        tgt_text  = f.readlines()[:args.num_samples] # num_samples
        f.close()
        # print("target text: ", tgt_text)
    
    tgt_text = "A dog playing with cat"
    
    # clip text features of the target
    with torch.no_grad():
        target_text_token    = clip.tokenize(tgt_text).to(device)
        target_text_features = clip_img_model_vitb32.encode_text(target_text_token)
        target_text_features = target_text_features / target_text_features.norm(dim=1, keepdim=True)
        target_text_features = target_text_features.detach() # z_tar = g(c_tar)
        # print("Text target shape: ", target_text_features.shape) # num_samples x 512

    
    if args.wandb:
        run = wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, reinit=True)
    
    for i, (image_clean, gt_txt, gt_path, image , tar_txt, path) in tqdm(enumerate(data_loader)):
        # print("Target Image: ", image.shape)
        # print("Image clean: ", image_clean.shape)
        
        image = image.to(device)  # size=(10, 3, 224, 224)
        image_clean = image_clean.to(device)  # size=(10, 3, 224, 224)
        
        # batch_size == num_samples
        # obtain all text features (via CLIP text encoder)
        # adv_text_features = adv_vit_text_features[batch_size * (i): batch_size * (i+1)] # z_clean = g(c_clean)     
        tgt_text_features = target_text_features[batch_size * (i): batch_size * (i+1)] # z_tar = g(c_tar)
        
        # ------------------- random gradient-free method
        # print("init delta with diff(adv-clean)")
        delta = torch.tensor((image - image_clean))
        torch.cuda.empty_cache()
        
        better_flag = 0
        adv_image_in_current_step = image.clone()
        for step_idx in range(args.steps):
            # print(f"{i}-th image - {step_idx}-th step")
            # step 1. obtain purturbed images
  
            image_repeat = adv_image_in_current_step.repeat(num_query, 1, 1, 1)             
            lavis_text_of_adv_image_in_current_step = _i2t(args, txt_processors, model, image=adv_image_in_current_step) # c = p(x)
            adv_vit_text_token_in_current_step      = clip.tokenize(lavis_text_of_adv_image_in_current_step).to(device) # 
            adv_vit_text_features_in_current_step   = clip_img_model_vitb32.encode_text(adv_vit_text_token_in_current_step) # z = g(c_)
            adv_vit_text_features_in_current_step   = adv_vit_text_features_in_current_step / adv_vit_text_features_in_current_step.norm(dim=1, keepdim=True)
            adv_vit_text_features_in_current_step   = adv_vit_text_features_in_current_step.detach()                
            adv_text_features                       = adv_vit_text_features_in_current_step #  z = [g(c)]
            torch.cuda.empty_cache()
                
            # print("image_repeat shape: ", image_repeat.shape)
                
            query_noise = torch.randn_like(image_repeat).sign() # Rademacher noise
            perturbed_image_repeat = torch.clamp(image_repeat + (sigma * query_noise), 0.0, 255.0)  # x + sigma * noise
            
            # num_query is obtained via serveral iterations
            text_of_perturbed_imgs = []
            # for query_idx in range(num_query//num_sub_query):
            print("estimate grad...")
            for query_idx in tqdm(range(num_query)):
                sub_perturbed_image_repeat = perturbed_image_repeat[batch_size * (query_idx) : batch_size * (query_idx + 1)]
                # print("Sub_pertubed image repeat shape: ", sub_perturbed_image_repeat.shape)
                if args.model_name == 'img2prompt_vqa':
                    text_of_sub_perturbed_imgs = _i2t(args, txt_processors, model, image=sub_perturbed_image_repeat) # c_ = p(x + sigma * noise)
                else:
                    with torch.no_grad():
                        text_of_sub_perturbed_imgs = _i2t(args, txt_processors, model, image=sub_perturbed_image_repeat) # c_ =p(x + sigma * noise)
                text_of_perturbed_imgs.extend(text_of_sub_perturbed_imgs) # [c_ ] has len = num_query
            
            print("Text_of_pertubed: ", text_of_perturbed_imgs)
            
            # step 2. estimate grad => z_^T * g(c_tar) - z^T * g(c_tar)
            with torch.no_grad():
                perturb_text_token    = clip.tokenize(text_of_perturbed_imgs).to(device) # [c_ ] has len = num_query
                perturb_text_features = clip_img_model_vitb32.encode_text(perturb_text_token) # z_ = g(c_)
                perturb_text_features = perturb_text_features / perturb_text_features.norm(dim=1, keepdim=True)
                perturb_text_features = perturb_text_features.detach() # z_ = [g(c_)]
            
            coefficient = torch.sum((perturb_text_features - adv_text_features) * tgt_text_features, dim=-1)  # size = (num_query * batch_size)
            coefficient = coefficient.reshape(num_query, batch_size, 1, 1, 1)
            query_noise = query_noise.reshape(num_query, batch_size, 3, args.input_res, args.input_res)
            pseudo_gradient = coefficient * query_noise / sigma # size = (num_query, batch_size, 3, args.input_res, args.input_res)
            pseudo_gradient = pseudo_gradient.mean(0) # size = (bs, 3, args.input_res, args.input_res)
            
            # step 3. result
            # delta_data = torch.clamp(delta + alpha * torch.sign(pseudo_gradient), min=-epsilon, max=epsilon)
            delta_data = torch.clamp(alpha * torch.sign(pseudo_gradient), min=-epsilon, max=epsilon)
            delta.data = delta_data
                        
            adv_image_in_current_step = torch.clamp(image_clean + delta, 0.0, 255.0)
            # get adv text
            if args.model_name == 'img2prompt_vqa':
                lavis_text_of_adv_image_in_current_step = _i2t(args, txt_processors, model, image=adv_image_in_current_step)
            else:
                with torch.no_grad():
                    lavis_text_of_adv_image_in_current_step = _i2t(args, txt_processors, model, image=adv_image_in_current_step)
            
           
        # log text
        basename = os.path.basename(gt_path[0])

        torchvision.utils.save_image(adv_image_in_current_step / 255.0, os.path.join(args.output, basename))
        torchvision.utils.save_image(image_clean / 255.0, os.path.join(args.output, "clean_" + basename))
        torchvision.utils.save_image(delta_data / 255.0, os.path.join(args.output, "pertu_" + basename))
        
        original_cap = _i2t(args, txt_processors, model, image=image_clean) # c = p(x)
        print("Adv image's caption:", lavis_text_of_adv_image_in_current_step)
        print("original cap:", original_cap)

        with open(os.path.join(args.output + '.txt'), 'a') as f:
            # print(''.join([best_caption]), file=f)
            if better_flag:
                f.write(lavis_text_of_adv_image_in_current_step[0]+'\n')
            else:
                f.write(lavis_text_of_adv_image_in_current_step[0])
            f.close()
            
        break
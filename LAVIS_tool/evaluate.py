import argparse
import os
import random
import clip
import torch
import numpy as np
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

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(args):
    seed_everything(22520691)
    with open(args.result_path, "r") as f:
        lines = [line.strip().split("\t") for line in f.readlines()]
        adv_cap = [line[3] for line in lines]
        tar_cap = [line[2] for line in lines]
        c_clean = [line[1] for line in lines]
        
        
    
    clip_score_model_names = ["ViT-B/32", "ViT-L/14", "ViT-B/16"]
    for model_name in clip_score_model_names:
        evaluate_clip_model, evaluate_preprocess = clip.load(model_name, device="cuda")
        evaluate_clip_model.eval()
        
        adv_cap_embedding = clip_encode_text(adv_cap, evaluate_clip_model)
        tar_cap_embedding = clip_encode_text(tar_cap, evaluate_clip_model)
        c_clean_embedding = clip_encode_text(c_clean, evaluate_clip_model)
        
        score = torch.sum(adv_cap_embedding * tar_cap_embedding, dim=1)
        clean_score = torch.sum(c_clean_embedding * tar_cap_embedding, dim=1)
        print(f"{model_name}, {torch.mean(score).item()}, {torch.mean(clean_score).item()}")
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str)
    args = parser.parse_args()
    
    main(args)
import argparse
import os
import random
import clip
import torch

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
def main():
    
    with open(args.result_path, "r") as f:
        lines = [line.strip().split("\t") for line in f.readlines()]
        adv_cap = [line[3] for line in lines]
        tar_cap = [line[2] for line in lines]
        
        
    
    clip_score_model_names = ["Vit-B/32", "Vit-L/14", "ViT-B/16"]
    for model_name in clip_score_model_names:
        evaluate_clip_model, evaluate_preprocess = clip.load(args.clip_score_model, device="cuda")
        evaluate_clip_model.eval()
        
        adv_cap_embedding = clip_encode_text(adv_cap, evaluate_clip_model)
        tar_cap_embedding = clip_encode_text(tar_cap, evaluate_clip_model)
        
        score = torch.sum(adv_cap_embedding * tar_cap_embedding, dim=1)
        print(f"{model_name}, {torch.mean(score).item()}")
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="zo", type=str)
    parser.add_argument("--result_path", type=str)
    args = parser.parse_args()
    
    main(args)
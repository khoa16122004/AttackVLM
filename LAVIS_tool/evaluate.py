from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import argparse
import os
import random
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class TextEncoder:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).cuda().eval()
    
    def encode(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Lấy embedding của token [CLS]
        embeddings = F.normalize(embeddings, p=2, dim=1)  # Chuẩn hóa vector
        return embeddings

def main(args):
    seed_everything(22520691)
    
    with open(args.result_path, "r") as f:
        lines = [line.strip().split("\t") for line in f.readlines()]
        adv_cap = [line[3] for line in lines]
        tar_cap = [line[2] for line in lines]
    
    with open(args.annotation_path, "r") as r:
        c_clean = [r.readline().strip().split("\t")[1] for _ in range(len(adv_cap))]

    print(c_clean[0])
    print(tar_cap[0])

    encoder = TextEncoder("bert-base-uncased")

    adv_cap_embedding = encoder.encode(adv_cap)
    tar_cap_embedding = encoder.encode(tar_cap)
    c_clean_embedding = encoder.encode(c_clean)

    score = torch.sum(adv_cap_embedding * tar_cap_embedding, dim=1)
    clean_score = torch.sum(c_clean_embedding * tar_cap_embedding, dim=1)
    
    print(f"BERT, {torch.mean(score).item()}, {torch.mean(clean_score).item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str)
    parser.add_argument("--annotation_path", type=str)
    args = parser.parse_args()
    
    main(args)

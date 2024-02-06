import argparse
import math

import pandas as pd
import os
import json
import numpy as np
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from peft import (
    PeftModel,
)
import torch
from tqdm.notebook import tqdm
import pickle
from arguments import ModelArguments, DataArguments, TrainingArguments
from model.model import RecComModel



def load_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_pkl(file):
    with open(file, 'rb') as file:
        data = pickle.load(file)
    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0, help='ID of running GPU')
    parser.add_argument("--lora_ckpt", type=str,
                        default="./ckpt/LSVCR/")
    parser.add_argument("--lora", action="store_true", default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model_args = ModelArguments()
    data_args = DataArguments()

    print(vars(args))

    device = torch.device("cuda", args.gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              model_max_length=512,
                                              trust_remote_code=True)
    data_path = data_args.data_path
    print(data_path)

    photos = load_json(os.path.join(data_path, "photo.json"))
    all_photos = load_pkl(os.path.join(data_path, "all_photos.pkl"))
    photo2id = {str(photo): i for i, photo in enumerate(all_photos)}
    n_photos = len(all_photos)

    if args.lora:
        model = RecComModel.from_pretrained(
            model_args.model_name_or_path,
            n_photos=n_photos,
            args=model_args,
            empty_init=False,
            torch_dtype=torch.bfloat16,
            device_map=None,
        ).to(torch.bfloat16)
        print(model.llm_emb_mlp[0].weight)
        print(model.contrast_adapter.weight)

        model = PeftModel.from_pretrained(
            model,
            args.lora_ckpt,
            torch_dtype=torch.bfloat16
        )

        print(model.llm_emb_mlp.modules_to_save['default'][0].weight)
        print(model.contrast_adapter.modules_to_save['default'].weight)


    else:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    model = model.to(device)

    print("Total photos:", n_photos)

    photo_text = {}
    for photo in photos:
        text = photos[str(photo)]["title"]
        photo_text[photo] = text

    if args.lora:
        photo_emb = np.zeros((n_photos, model.base_model.model.llm_emb_size), dtype=np.float32)
        photo_emb_full = np.zeros((n_photos, 4096), dtype=np.float32)

    else:
        photo_emb = np.zeros((n_photos, 4096), dtype=np.float32)

    with torch.no_grad():
        for i, photo in tqdm(enumerate(all_photos)):
            if photo == '[PAD]': continue
            if (i + 1) % 1000 == 0:
                print("==>", (i + 1))

            text = photo_text[photo]
            id = photo2id[photo]

            inputs = tokenizer(text, max_length=512, truncation=True, return_tensors='pt', padding="longest").to(device)
            if args.lora:

                text_emb, text_emb_full = model.base_model.model.get_text_hidden_states(inputs)
                if i == 1:
                    test_emb = model(**inputs, return_dict=True).hidden_states
                    test_emb = test_emb.transpose(0, 1).contiguous() * inputs['attention_mask'].unsqueeze(-1)
                    test_emb = test_emb.sum(dim=1) / inputs['attention_mask'].sum(dim=-1, keepdim=True)
                    assert torch.all(text_emb_full == test_emb)

                text_emb = text_emb.squeeze().detach().to(torch.float32).cpu().numpy()
                text_emb_full = text_emb_full.squeeze().detach().to(torch.float32).cpu().numpy()
                text_emb = np.nan_to_num(text_emb, nan=0.0, posinf=0.0, neginf=0.0)
                text_emb_full = np.nan_to_num(text_emb_full, nan=0.0, posinf=0.0, neginf=0.0)

                photo_emb[id] = text_emb
                photo_emb_full[id] = text_emb_full
            else:
                text_emb = model.transformer(**inputs)[0]

                text_emb = text_emb.transpose(0, 1).contiguous() * inputs['attention_mask'].unsqueeze(-1)
                text_emb = text_emb.sum(dim=1) / inputs['attention_mask'].sum(dim=-1, keepdim=True)
                text_emb = text_emb.squeeze().detach().to(torch.float32).cpu().numpy()
                text_emb = np.nan_to_num(text_emb, nan=0.0, posinf=0.0, neginf=0.0)

                photo_emb[id] = text_emb

    print(all_photos[:10])

    if args.lora:
        file_name = "photo_embs.npy"
    else:
        file_name = "photo_embs_chatglm3.npy"

    np.save(os.path.join(data_path, file_name), photo_emb)


    if args.lora:
        file_name = "photo_full_embs.npy"
        np.save(os.path.join(data_path, file_name), photo_emb_full)

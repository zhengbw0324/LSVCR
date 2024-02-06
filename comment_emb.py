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
    TaskType,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
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
    parser.add_argument("--batch_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1000000)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(vars(args))

    model_args = ModelArguments()
    data_args = DataArguments()


    device = torch.device("cuda", args.gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path,
                                              model_max_length=512,
                                              trust_remote_code=True)
    data_path = data_args.data_path
    print(data_path)

    all_photos = load_pkl(os.path.join(data_path, "all_photos.pkl"))
    n_photos = len(all_photos)

    comments = load_json(os.path.join(data_path, "comment.json"))

    if args.lora:
        model = RecComModel.from_pretrained(
            model_args.model_name_or_path,
            n_photos=n_photos,
            args=model_args,
            empty_init=False,
            torch_dtype=torch.bfloat16,
            # device_map=None,
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


    all_comments = sorted(list(comments.keys()))
    n_comments = len(all_comments)

    print("Total comments:", n_comments)
    print("Total batch:", math.ceil(n_comments/args.batch_size))
    print("Current batch:", args.batch_index + 1)

    start = args.batch_index * args.batch_size
    end = min((args.batch_index + 1) * args.batch_size, n_comments)

    print("Batch range:", start, end)

    batch_comments = all_comments[start:end]

    comment_text = {}
    for comment in batch_comments:
        text = comments[str(comment)]["content"]
        comment_text[comment] = text

    comment_emb = {}
    comment_emb_full = {}
    with torch.no_grad():
        for i, com in tqdm(enumerate(batch_comments)):
            if (i + 1) % 1000 == 0:
                print("==>", (i + 1))

            text = comment_text[com]

            inputs = tokenizer(text, max_length=512, truncation=True, return_tensors='pt', padding="longest").to(device)
            if args.lora:

                text_emb, text_emb_full = model.base_model.model.get_text_hidden_states(inputs)
                if i == 0:
                    test_emb = model(**inputs, return_dict=True).hidden_states
                    test_emb = test_emb.transpose(0, 1).contiguous() * inputs['attention_mask'].unsqueeze(-1)
                    test_emb = test_emb.sum(dim=1) / inputs['attention_mask'].sum(dim=-1, keepdim=True)
                    assert torch.all(text_emb_full == test_emb)

                text_emb = text_emb.squeeze().detach().to(torch.float32).cpu().numpy()
                text_emb_full = text_emb_full.squeeze().detach().to(torch.float32).cpu().numpy()
                text_emb = np.nan_to_num(text_emb, nan=0.0, posinf=0.0, neginf=0.0)
                text_emb_full = np.nan_to_num(text_emb_full, nan=0.0, posinf=0.0, neginf=0.0)

                comment_emb[com] = text_emb
                comment_emb_full[com] = text_emb_full
            else:
                text_emb = model.transformer(**inputs)[0]

                text_emb = text_emb.transpose(0, 1).contiguous() * inputs['attention_mask'].unsqueeze(-1)
                text_emb = text_emb.sum(dim=1) / inputs['attention_mask'].sum(dim=-1, keepdim=True)
                text_emb = text_emb.squeeze().detach().to(torch.float32).cpu().numpy()
                text_emb = np.nan_to_num(text_emb, nan=0.0, posinf=0.0, neginf=0.0)

                comment_emb[com] = text_emb

    if args.lora:
        file_name = "comment_embs_{}_{}.pkl".format(start, end)
    else:
        file_name = "comment_embs_chatglm3_{}_{}.pkl".format(start, end)

    with open(os.path.join(data_path, file_name), 'wb') as file:
        pickle.dump(comment_emb, file)

    all_comments = ['[PAD]'] + sorted(list(comments.keys()))
    print(all_comments[:10])

    if args.lora:
        file_name = "comment_full_embs_{}_{}.pkl".format(start, end)
        with open(os.path.join(data_path, file_name), 'wb') as file:
            pickle.dump(comment_emb_full, file)


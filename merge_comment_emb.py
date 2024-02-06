import argparse
import math

import pandas as pd
import os
import json
import numpy as np
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
    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--file_prefix", type=str, default="comment_embs_")
    parser.add_argument("--output_file", type=str, default="comment_embs.npy")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(vars(args))

    model_args = ModelArguments()
    data_args = DataArguments()

    data_path = data_args.data_path
    print(data_path)

    all_comments = load_pkl(os.path.join(data_path, "all_comments.pkl"))
    comment2id = {str(comment): i for i, comment in enumerate(all_comments)}
    n_comments = len(all_comments)
    print("n_comments", n_comments)
    print(all_comments[:10])

    file_list = os.listdir(data_path)
    file_list = [os.path.join(data_path, file) for file in file_list if
                 file.startswith(args.file_prefix) and file.endswith(".pkl")]

    for f in file_list:
        print(f)
    print("file num:", len(file_list))

    all_emb = np.zeros((n_comments, args.emb_size), dtype=np.float32)

    _all_comments = set()
    _all_comments.add('[PAD]')
    for file in file_list:
        print(file)
        embs = load_pkl(file)
        comment_list = list(embs.keys())
        for com in comment_list:
            _all_comments.add(com)
            id = comment2id[com]
            all_emb[id] = np.nan_to_num(embs.pop(com).astype(np.float32),
                                                 nan=0.0, posinf=0.0, neginf=0.0)
        del embs

    print(len(all_comments))
    print(len(list(_all_comments)))

    np.save(os.path.join(data_path, args.output_file), all_emb)


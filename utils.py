import json
import logging
import os
import random
import datetime

import numpy as np
import torch

from torch.utils.data import ConcatDataset, DataLoader

from collator import FinetuneCollator

from data import RecDataset, CommRankDataset
from finetune_data import FtRecDataset, FtCommRankDataset

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime("%b-%d-%Y_%H-%M-%S")

    return cur


def load_pretrain(model_args):
    checkpoint = model_args.pretrain_checkpoint
    max_position = model_args.max_position
    pretrain_state_dict = torch.load(checkpoint, map_location="cpu")
    new_pretrain_state_dict = {}
    for key, value in pretrain_state_dict.items():
        key = ".".join(key.split(".")[2:])
        if "position_embedding" in key:
            if value.shape[0] != max_position:
                new_value = torch.zeros((max_position, value.shape[1]), dtype=value.dtype)
                new_value[:value.shape[0]] = value
                new_value[value.shape[0]:] = torch.mean(value, dim=0)
                value = new_value
        new_pretrain_state_dict[key] = value.to(torch.float32)

    return new_pretrain_state_dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)


def load_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_row_data(file):
    data = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))

    return data


def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def load_datasets(data_args, tokenizer):
    train_rec_data = RecDataset(data_args, tokenizer, mode="train", sample_num=data_args.train_num)
    photos = train_rec_data.photos
    comments = train_rec_data.comments
    # val_rec_data = RecDataset(data_args, tokenizer, mode="val", sample_num=data_args.val_num, photos=photos, comments=comments)
    train_com_data = CommRankDataset(data_args, tokenizer, mode="train", sample_num=data_args.train_num, photos=photos, comments=comments)
    # val_com_data = CommRankDataset(data_args, tokenizer, mode="val", sample_num=data_args.val_num, photos=photos, comments=comments)

    n_photos = len(train_rec_data.all_photos)

    train_data = ConcatDataset([train_rec_data, train_com_data])

    # valid_data = ConcatDataset([val_rec_data, val_com_data])

    return train_data, None, n_photos


def load_finetune_data(data_args, training_args):

    print("Loading data from {}".format(data_args.data_path))

    if data_args.finetune_task.lower() == "rec":
        train_data = FtRecDataset(data_args, mode="train")
        valid_data = FtRecDataset(data_args, mode="val")
        test_data = FtRecDataset(data_args, mode="test")
    elif data_args.finetune_task.lower() == "commrank":
        train_data = FtCommRankDataset(data_args, mode="train")
        valid_data = FtCommRankDataset(data_args, mode="val")
        test_data = FtCommRankDataset(data_args, mode="test")
    else:
        raise NotImplementedError

    collate_fn = FinetuneCollator(data_args, train_data.photo_embs, train_data.comment_embs)

    train_data_loader = DataLoader(train_data, num_workers=training_args.dataloader_num_workers,
                                   collate_fn=collate_fn, batch_size=training_args.per_device_train_batch_size,
                                   shuffle=True, pin_memory=True)
    val_data_loader = DataLoader(valid_data, num_workers=training_args.dataloader_num_workers,
                                   collate_fn=collate_fn, batch_size=training_args.per_device_eval_batch_size,
                                   shuffle=False, pin_memory=True)
    test_data_loader = DataLoader(test_data, num_workers=training_args.dataloader_num_workers,
                                   collate_fn=collate_fn, batch_size=training_args.per_device_eval_batch_size,
                                   shuffle=False, pin_memory=True)

    return train_data_loader, val_data_loader, test_data_loader

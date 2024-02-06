import copy
import pickle
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import logging
import re
import pdb
import json
from prompt import *
import numpy as np


class FtBaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.data_path = args.data_path
        self.max_phis_len = args.max_phis_len
        self.max_chis_len = args.max_chis_len
        self.max_candidate_num = args.max_candidate_num
        self.photo_emb_file = args.photo_emb_file
        self.comment_emb_file = args.comment_emb_file

    def load_json(self, file):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def load_pkl(self, file):
        with open(file, 'rb') as file:
            data = pickle.load(file)
        return data

    def load_row_data(self, file):
        data = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                data.append(json.loads(line))

        return data


class FtRecDataset(FtBaseDataset):

    def __init__(self, args, mode="train"):
        super().__init__(args)

        self.mode = mode
        self.neg_photo_num = args.neg_photo_num

        self.all_photos = self.load_pkl(os.path.join(self.data_path, "all_photos.pkl"))
        self.photo2id = {str(photo): i for i, photo in enumerate(self.all_photos)}
        self.all_comments = self.load_pkl(os.path.join(self.data_path, "all_comments.pkl"))
        self.comment2id = {str(comment): i for i, comment in enumerate(self.all_comments)}

        self._load_data()


    def _load_data(self):
        self.inter_data = self.load_row_data(os.path.join(self.data_path, f"rec.{self.mode}.json"))

        if self.mode == "train":
            self.photo_embs = np.load(os.path.join(self.data_path, self.photo_emb_file))
            self.comment_embs = np.load(os.path.join(self.data_path, self.comment_emb_file))


        photos = self.load_json(os.path.join(self.data_path, "photo.json"))
        self.photo_comment_list = {photo: feat["comment_list"] for photo, feat in photos.items()}



    def __len__(self):

        return len(self.inter_data)

    def _sample_candidates(self, target_photo_id, neg_num):

        neg_photos = np.random.choice(np.arange(len(self.all_photos))[1:], neg_num, replace=False)
        while target_photo_id in neg_photos:
            neg_photos = np.random.choice(np.arange(len(self.all_photos))[1:], neg_num, replace=False)
        cands = np.concatenate([[target_photo_id], neg_photos])
        labels = np.array([1] + [0] * neg_num)

        indices = np.random.permutation(np.arange(len(cands)))
        cands = cands[indices]
        labels = labels[indices]

        return cands, labels

    def __getitem__(self, index):


        d = self.inter_data[index]
        user_id = d["user_id"]
        target_photo = d["target_photo"]
        photo_inter_his = d["photo_inter_his"][-self.max_phis_len:]
        comment_inter_his = d["comment_inter_his"]

        target_photo_id = self.photo2id[str(target_photo)]

        if self.mode == "train":
            if self.neg_photo_num > 0:
                candidates, labels = self._sample_candidates(target_photo_id, self.neg_photo_num)
                labels = int(labels.argmax())
            else:
                candidates = None
                labels = target_photo_id
        else:
            if self.max_candidate_num > 0:
                candidates, labels = self._sample_candidates(target_photo_id, self.max_candidate_num - 1)
            else:
                candidates = None
                labels = np.zeros(len(self.all_photos), dtype=int)
                labels[target_photo_id] = 1


        photo_his_pid = []
        photo_his_cid = []
        for p in photo_inter_his:
            photo_his_pid.append(self.photo2id[str(p)])
            comment_list = self.photo_comment_list[str(p)]
            if len(comment_list) == 0:
                comment_id = 0
            else:
                if self.mode == "train":
                    pop_comment_list = comment_list[:3]
                    comment = np.random.choice(pop_comment_list, 1)[0]
                else:
                    comment = comment_list[0]
                comment_id = self.comment2id[str(comment)]
            photo_his_cid.append(comment_id)

        comment_his_pid = []
        comment_his_cid = []
        for p, c_list in zip(comment_inter_his[0][-self.max_chis_len:], comment_inter_his[1][-self.max_chis_len:]):
            comment_his_pid.append(self.photo2id[str(p)])
            comment_his_cid.append([self.comment2id[str(c)] for c in c_list])


        return dict(candidates=candidates,
                    labels = labels,
                    photo_his_pid = photo_his_pid,
                    photo_his_cid = photo_his_cid,
                    comment_his_pid = comment_his_pid,
                    comment_his_cid = comment_his_cid
                    )


class FtCommRankDataset(FtBaseDataset):

    def __init__(self, args, mode="train"):
        super().__init__(args)

        self.mode = mode
        self.neg_comment_num = args.neg_comment_num

        self.all_photos = self.load_pkl(os.path.join(self.data_path, "all_photos.pkl"))
        self.photo2id = {str(photo): i for i, photo in enumerate(self.all_photos)}
        self.all_comments = self.load_pkl(os.path.join(self.data_path, "all_comments.pkl"))
        self.comment2id = {str(comment): i for i, comment in enumerate(self.all_comments)}

        self._load_data()

    def _load_data(self):
        self.inter_data = self.load_row_data(os.path.join(self.data_path, f"comm_rank.{self.mode}.json"))

        if self.mode == "train":
            self.photo_embs = np.load(os.path.join(self.data_path, self.photo_emb_file))
            self.comment_embs = np.load(os.path.join(self.data_path, self.comment_emb_file))

        photos = self.load_json(os.path.join(self.data_path, "photo.json"))
        self.photo_comment_list = {photo: feat["comment_list"] for photo, feat in photos.items()}


    def __len__(self):

        return len(self.inter_data)

    def _sample_candidates(self, pos_comment, pos_comments, all_candidates):

        if len(all_candidates)==0 or len(all_candidates) == len(pos_comments):
            neg_comments = np.random.choice(self.all_comments[1:], self.neg_comment_num, replace=False)
        else:
            all_labels = np.array([1 if c in pos_comments else 0 for c in all_candidates])
            neg_ids = np.array([]).astype(int)
            while len(neg_ids) < self.neg_comment_num:
                neg_ids = np.concatenate((
                    neg_ids,
                    np.random.permutation(np.where(all_labels==0)[0]),
                ))
            neg_ids = neg_ids[:self.neg_comment_num].astype(int)
            neg_comments = all_candidates[neg_ids]

        cands = np.concatenate([[pos_comment], neg_comments])
        labels = np.array([1] + [0] * self.neg_comment_num)

        indices = np.random.permutation(np.arange(len(cands)))
        cands = cands[indices]
        labels = labels[indices]

        return cands, labels

    def __getitem__(self, index):


        d = self.inter_data[index]
        user_id = d["user_id"]
        target_photo = d["target_photo"]
        pos_comments = d["pos_comments"]
        photo_inter_his = d["photo_inter_his"][-self.max_phis_len:]
        comment_inter_his = d["comment_inter_his"]
        pos_comment = np.random.choice(pos_comments, 1)[0]

        com_photo_id = self.photo2id[str(target_photo)]

        all_candidates = np.array(self.photo_comment_list[str(target_photo)])

        if self.mode == "train":
            candidates, labels = self._sample_candidates(pos_comment, pos_comments, all_candidates)
            labels = int(labels.argmax())
        else:
            all_labels = np.array([1 if c in pos_comments else 0 for c in all_candidates])
            if len(all_candidates) > self.max_candidate_num:
                neg_ids = np.array([]).astype(int)
                neg_num = self.max_candidate_num - len(pos_comments)
                while len(neg_ids) < neg_num:
                    neg_ids = np.concatenate((
                        neg_ids,
                        np.random.permutation(np.where(all_labels == 0)[0]),
                    ))
                neg_ids = neg_ids[:neg_num]
                neg_comments = all_candidates[neg_ids]
                candidates = np.concatenate([pos_comments, neg_comments])
                labels = np.array([1] * len(pos_comments) + [0] * neg_num)
                indices = np.random.permutation(np.arange(self.max_candidate_num))
                candidates = candidates[indices]
                labels = labels[indices]
            else:
                candidates = all_candidates
                labels = all_labels

        candidates = np.array([self.comment2id[str(c)] for c in candidates])

        photo_his_pid = []
        photo_his_cid = []
        for p in photo_inter_his:
            photo_his_pid.append(self.photo2id[str(p)])
            comment_list = self.photo_comment_list[str(p)]
            if len(comment_list) == 0:
                comment_id = 0
            else:
                if self.mode == "train":
                    pop_comment_list = comment_list[:3]
                    comment = np.random.choice(pop_comment_list, 1)[0]
                else:
                    comment = comment_list[0]
                comment_id = self.comment2id[str(comment)]
            photo_his_cid.append(comment_id)

        comment_his_pid = []
        comment_his_cid = []
        for p, c_list in zip(comment_inter_his[0][-self.max_chis_len:], comment_inter_his[1][-self.max_chis_len:]):
            comment_his_pid.append(self.photo2id[str(p)])
            comment_his_cid.append([self.comment2id[str(c)] for c in c_list])

        return dict(com_photo_id=com_photo_id,
                    candidates=candidates,
                    labels=labels,
                    photo_his_pid=photo_his_pid,
                    photo_his_cid=photo_his_cid,
                    comment_his_pid=comment_his_pid,
                    comment_his_cid=comment_his_cid
                    )

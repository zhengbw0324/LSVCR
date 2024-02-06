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


class BaseDataset(Dataset):

    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args
        self.data_path = args.data_path
        self.tokenizer = tokenizer
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length
        self.max_phis_len = args.max_phis_len
        self.max_chis_len = args.max_chis_len
        self.instruction_emb = args.instruction_emb

        self.inter_aug_p = args.inter_aug_p
        self.phis_aug_p = args.phis_aug_p
        self.chis_aug_p = args.chis_aug_p

        self.photos = None
        self.comments = None

        self.all_photos = self.load_pkl(os.path.join(self.data_path, "all_photos.pkl"))
        self.photo2id = {str(photo): i for i, photo in enumerate(self.all_photos)}
        self.all_comments = self.load_pkl(os.path.join(self.data_path, "all_comments.pkl"))
        self.comment2id = {str(comment): i for i, comment in enumerate(self.all_comments)}

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


    def _get_llm_inputs_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        a_ids = self.tokenizer.encode(text=instruction, add_special_tokens=True, truncation=True,
                                      max_length=self.max_source_length)
        b_ids = self.tokenizer.encode(text=response, add_special_tokens=False, truncation=True,
                                      max_length=self.max_target_length)

        context_length = len(a_ids)
        input_ids = a_ids + b_ids + [self.tokenizer.eos_token_id]
        labels = [self.tokenizer.pad_token_id] * context_length + b_ids + [self.tokenizer.eos_token_id]

        labels = [(l if l != self.tokenizer.pad_token_id else -100) for l in labels]

        return input_ids, labels

    def _get_photo_text(self, photo):

        template = "视频标题：{title}；热门评论：{comment}"
        photo_feature = self.photos[str(photo)]
        title = photo_feature["title"]
        comment_list = photo_feature["comment_list"]
        if len(comment_list) == 0:
            comment = ""
        else:
            pop_comment_list = comment_list[:3]
            comment_id = np.random.choice(pop_comment_list, 1)[0]
            comment_feature = self.comments[str(comment_id)]
            comment = comment_feature["content"]
        dic = {"title": title, "comment": comment}

        photo_text = template.format(**dic)

        return photo_text, title, comment

    def _get_aug_photo_text(self, photo):

        template = "视频标题：{title}；热门评论：{comment}"
        photo_feature = self.photos[str(photo)]
        title = photo_feature["title"]
        comment_list = photo_feature["comment_list"]
        if len(comment_list) == 0:
            pair_comment = ["",""]
        else:
            pop_comment_list = comment_list[:3]
            if len(pop_comment_list) < 2:
                pair_comment_id = np.random.choice(pop_comment_list, 2)
            else:
                pair_comment_id = np.random.choice(pop_comment_list, 2, replace=False)
            pair_comment = [self.comments[str(com)]["content"] for com in pair_comment_id]

        dic = {"title": title, "comment": pair_comment[0]}

        photo_text = template.format(**dic)

        return photo_text, title, pair_comment[1]

    def _get_comment_text(self, photo, comment_list):

        template = "视频标题：{title}；用户交互评论：{comment_list}"

        title = self.photos[str(photo)]["title"]
        c_text_list = []
        if len(comment_list) > 3:
            comment_list = np.random.choice(comment_list, 3, replace=False)
        for c in comment_list:
            content = self.comments[str(c)]["content"]
            c_text_list.append(content)
        c_text = "{" + str(c_text_list)[1:-1] + "}"

        d = {"title": title, "comment_list": c_text}

        comment_text = template.format(**d)

        return comment_text, title, c_text_list

    def _get_aug_comment_text(self, photo, comment_list):

        template = "视频标题：{title}；用户交互评论：{comment_list}"

        title = self.photos[str(photo)]["title"]
        comment_list = np.array(comment_list)
        if len(comment_list) > 3:
            comment_list = np.random.choice(comment_list, 3, replace=False)


        pop_comment_list = np.array(self.photos[str(photo)]["comment_list"][:10])

        if len(pop_comment_list)==0 or len(pop_comment_list) == len(comment_list):
            aug_comment_list = np.random.choice(self.all_comments[1:], len(comment_list), replace=False)
        else:
            pop_comment_labels = np.array([1 if c in comment_list else 0 for c in pop_comment_list])
            aug_ids = np.array([]).astype(int)
            while len(aug_ids) < len(comment_list):
                aug_ids = np.concatenate((
                    aug_ids,
                    np.random.permutation(np.where(pop_comment_labels==0)[0]),
                ))
            aug_ids = aug_ids[:len(comment_list)].astype(int)
            aug_comment_list = pop_comment_list[aug_ids]

        # aug_comment_list = np.random.choice(pop_comment_list, len(comment_list), replace=False)
        comment_list = np.concatenate([comment_list, aug_comment_list])
        comment_list = np.random.permutation(comment_list)

        c_text_list_1 = []
        c_text_list_2 = []
        for i, c in enumerate(comment_list):
            content = self.comments[str(c)]["content"]
            if i % 2==0:
                c_text_list_1.append(content)
            else:
                c_text_list_2.append(content)


        c_text = "{" + str(c_text_list_1)[1:-1] + "}"

        d = {"title": title, "comment_list": c_text}

        comment_text = template.format(**d)

        return comment_text, title, c_text_list_2


class RecDataset(BaseDataset):

    def __init__(self, args, tokenizer, mode="train", prompt_id=0, sample_num=-1,
                 photos=None, comments=None):
        super().__init__(args, tokenizer)

        self.mode = mode

        self.prompts = seqrec_prompt
        self.prompt_id = prompt_id
        self.sample_num = sample_num

        self.photos = photos
        self.comments = comments

        self._load_data()

    def _load_data(self):
        self.inter_data = self.load_row_data(os.path.join(self.data_path, f"rec.{self.mode}.json"))

        if self.photos is None:
            self.photos = self.load_json(os.path.join(self.data_path, "photo.json"))
        if self.comments is None:
            self.comments = self.load_json(os.path.join(self.data_path, "comment.json"))

        if self.sample_num > 0 and self.sample_num < len(self.inter_data):
            all_inter_idx = range(len(self.inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            self.inter_data = np.array(self.inter_data)[sample_idx].tolist()

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def __getitem__(self, index):


        d = self.inter_data[index]
        user_id = d["user_id"]
        target_photo = d["target_photo"]
        photo_inter_his = d["photo_inter_his"][-self.max_phis_len:]
        comment_inter_his = d["comment_inter_his"]

        target_text = self.photos[str(target_photo)]["title"]
        photo_inter_his_text = []
        comment_inter_his_text = []
        phis_titles = []
        phis_comments = []
        chis_titles = []
        chis_comments = []

        if random.random() < self.phis_aug_p:
            for p in photo_inter_his:
                if random.random() < self.inter_aug_p:
                    photo_text, title, comment = self._get_aug_photo_text(p)
                else:
                    photo_text, title, comment = self._get_photo_text(p)
                photo_inter_his_text.append(photo_text)
                phis_titles.append(title)
                phis_comments.append(comment)
        else:
            for p in photo_inter_his:
                photo_text, title, comment = self._get_photo_text(p)
                photo_inter_his_text.append(photo_text)
                phis_titles.append(title)
                phis_comments.append(comment)

        if random.random() < self.chis_aug_p:
            for p, c_list in zip(comment_inter_his[0][-self.max_chis_len:],
                                 comment_inter_his[1][-self.max_chis_len:]):
                if random.random() < self.inter_aug_p:
                    comment_text, title, c_text_list = self._get_aug_comment_text(p, c_list)
                else:
                    comment_text, title, c_text_list = self._get_comment_text(p, c_list)

                comment_inter_his_text.append(comment_text)
                chis_titles.append(title)
                chis_comments.append(c_text_list)
        else:
            for p, c_list in zip(comment_inter_his[0][-self.max_chis_len:],
                                 comment_inter_his[1][-self.max_chis_len:]):
                comment_text, title, c_text_list = self._get_comment_text(p, c_list)
                comment_inter_his_text.append(comment_text)
                chis_titles.append(title)
                chis_comments.append(c_text_list)



        target_id = self.photo2id[str(target_photo)]
        photo_inter_his_id = [self.photo2id[str(p)] for p in photo_inter_his]
        comment_inter_his_id = [self.photo2id[str(p)] for p in comment_inter_his[0][-self.max_chis_len:]]

        photo_his_str = "\n".join([str( i + 1) + ". " + s for i, s in enumerate(photo_inter_his_text)])
        comment_his_str = "\n".join([str(i + 1) + ". " + s for i, s in enumerate(comment_inter_his_text)])

        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        else:
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]
        prepared_data = {"response": target_text, "photo_his": photo_his_str, "comment_his": comment_his_str}

        input_ids, labels = self._get_llm_inputs_data(prepared_data, prompt)


        if self.instruction_emb:
            phis_titles = [photo_emb_prompt.format(_) for _ in phis_titles]
            phis_comments = [comment_emb_prompt.format(_) for _ in phis_comments]
            chis_titles = [photo_emb_prompt.format(_) for _ in chis_titles ]
            chis_comments = [comment_emb_prompt.format(_) for c_list in chis_comments for _ in c_list]


        return dict(input_ids=input_ids,
                    labels=labels,
                    target_text=target_text,
                    photo_his_id=photo_inter_his_id,
                    phis_titles=phis_titles,
                    phis_comments=phis_comments,
                    comment_his_id=comment_inter_his_id,
                    chis_titles=chis_titles,
                    chis_comments=chis_comments
                    )


class CommRankDataset(BaseDataset):

    def __init__(self, args, tokenizer, mode="train", prompt_id = 0, sample_num=-1,
                 photos=None, comments=None):
        super().__init__(args, tokenizer)

        self.mode = mode

        self.prompts = commrank_prompt
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.neg_comment_num = args.neg_comment_num

        self.photos = photos
        self.comments = comments

        self._load_data()


    def _load_data(self):
        self.inter_data = self.load_row_data(os.path.join(self.data_path, f"comm_rank.{self.mode}.json"))
        if self.photos is None:
            self.photos = self.load_json(os.path.join(self.data_path, "photo.json"))
        if self.comments is None:
            self.comments = self.load_json(os.path.join(self.data_path, "comment.json"))

        if self.sample_num > 0 and self.sample_num < len(self.inter_data):
            all_inter_idx = range(len(self.inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            self.inter_data = np.array(self.inter_data)[sample_idx].tolist()

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)

    def _sample_candidates(self, pos_comment, pos_comments, all_candidates):

        all_labels = np.array([1 if c in pos_comments else 0 for c in all_candidates])

        neg_ids = np.array([]).astype(int)
        neg_ids = np.concatenate((
            neg_ids,
            np.random.permutation(np.where(all_labels==0)[0]),
        ))
        neg_ids = neg_ids[:self.neg_comment_num]
        neg_comments = all_candidates[neg_ids]

        cands = np.concatenate([[pos_comment], neg_comments])

        indices = np.random.permutation(np.arange(len(cands)))
        cands = cands[indices]

        return cands

    def __getitem__(self, index):


        d = self.inter_data[index]
        user_id = d['user_id']
        target_photo = d['target_photo']
        pos_comments = d['pos_comments']
        photo_inter_his = d['photo_inter_his'][-self.max_phis_len:]
        comment_inter_his = d['comment_inter_his']
        pos_comment = np.random.choice(pos_comments, 1)[0]

        pos_comment_text = self.comments[str(pos_comment)]['content']
        com_photo_text = self.photos[str(target_photo)]['title']
        # all_candidates = np.array(self.photos[str(target_photo)]['comment_list'])
        # candidates = self._sample_candidates(pos_comment, pos_comments, all_candidates)

        photo_inter_his_text = []
        comment_inter_his_text = []
        phis_titles = []
        phis_comments = []
        chis_titles = []
        chis_comments = []

        if random.random() < self.phis_aug_p:
            for p in photo_inter_his:
                if random.random() < self.inter_aug_p:
                    photo_text, title, comment = self._get_aug_photo_text(p)
                else:
                    photo_text, title, comment = self._get_photo_text(p)
                photo_inter_his_text.append(photo_text)
                phis_titles.append(title)
                phis_comments.append(comment)
        else:
            for p in photo_inter_his:
                photo_text, title, comment = self._get_photo_text(p)
                photo_inter_his_text.append(photo_text)
                phis_titles.append(title)
                phis_comments.append(comment)

        if random.random() < self.chis_aug_p:
            for p, c_list in zip(comment_inter_his[0][-self.max_chis_len:],
                                 comment_inter_his[1][-self.max_chis_len:]):
                if random.random() < self.inter_aug_p:
                    comment_text, title, c_text_list = self._get_aug_comment_text(p, c_list)
                else:
                    comment_text, title, c_text_list = self._get_comment_text(p, c_list)

                comment_inter_his_text.append(comment_text)
                chis_titles.append(title)
                chis_comments.append(c_text_list)
        else:
            for p, c_list in zip(comment_inter_his[0][-self.max_chis_len:],
                                 comment_inter_his[1][-self.max_chis_len:]):
                comment_text, title, c_text_list = self._get_comment_text(p, c_list)
                comment_inter_his_text.append(comment_text)
                chis_titles.append(title)
                chis_comments.append(c_text_list)
        # candidates_text = [self.comments[str(c)]['content'] for c in candidates]

        com_photo_id = self.photo2id[str(target_photo)]
        photo_inter_his_id = [self.photo2id[str(p)] for p in photo_inter_his]
        comment_inter_his_id = [self.photo2id[str(p)] for p in comment_inter_his[0][-self.max_chis_len:]]

        photo_his_str = "\n".join([str(i + 1) + ". " + s for i, s in enumerate(photo_inter_his_text)])
        comment_his_str = "\n".join([str(i + 1) + ". " + s for i, s in enumerate(comment_inter_his_text)])
        # candidates_str = "\n".join(candidates_text)

        if self.mode == "train":
            prompt_id = random.randint(0, len(self.prompts) - 1)
        else:
            prompt_id = self.prompt_id

        prompt = self.prompts[prompt_id]
        prepared_data = {"response": pos_comment_text, "photo_his": photo_his_str,
                         "comment_his": comment_his_str, "photo": com_photo_text,
                         # "candidates": candidates_str
                         }

        input_ids, labels = self._get_llm_inputs_data(prepared_data, prompt)

        if self.instruction_emb:
            phis_titles = [photo_emb_prompt.format(_) for _ in phis_titles]
            phis_comments = [comment_emb_prompt.format(_) for _ in phis_comments]
            chis_titles = [photo_emb_prompt.format(_) for _ in chis_titles]
            chis_comments = [comment_emb_prompt.format(_) for c_list in chis_comments for _ in c_list]
            com_photo_text = photo_emb_prompt.format(com_photo_text)

        return dict(input_ids=input_ids,
                    labels=labels,
                    target_text=pos_comment_text,
                    photo_his_id=photo_inter_his_id,
                    phis_titles=phis_titles,
                    phis_comments=phis_comments,
                    comment_his_id=comment_inter_his_id,
                    chis_titles=chis_titles,
                    chis_comments=chis_comments,
                    com_photo_text=com_photo_text,
                    com_photo_id=com_photo_id
                    )

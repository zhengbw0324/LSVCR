import numpy as np
import torch
import copy
import argparse

import transformers
import math
from transformers import DataCollatorForSeq2Seq
import torch.nn.utils.rnn as rnn_utils


class Collator(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = args.model_max_length
        self.llm_collator = DataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
            padding="longest",
        )

    def _make_batch_assignees(self, batch_seqs):
        seq_sizes = torch.tensor([len(seq) for seq in batch_seqs])
        batch_index = torch.repeat_interleave(torch.arange(len(batch_seqs)), seq_sizes)
        return batch_index

    def __call__(self, batch):

        batch = np.array(batch)
        comm_rank_idx = []
        rec_idx = []
        for i, data in enumerate(batch):
            if "com_photo_text" in data:
                comm_rank_idx.append(i)
            else:
                rec_idx.append(i)

        if len(comm_rank_idx) == 0:
            comm_rank_photo_inputs = None
            comm_rank_photo_id = None
        else:
            comm_rank_data = batch[comm_rank_idx].tolist()
            batch = batch[rec_idx].tolist() + comm_rank_data

            comm_rank_photo_inputs = self.tokenizer(
                text=[d["com_photo_text"] for d in comm_rank_data],
                return_tensors="pt",
                padding="longest",
                max_length=self.args.max_target_length,
                truncation=True,
                return_attention_mask=True,
            )
            comm_rank_photo_id = torch.LongTensor([d["com_photo_id"] for d in comm_rank_data])

        batch_target_texts = [d["target_text"] for d in batch]

        target_text_inputs = self.tokenizer(
            text=batch_target_texts,
            return_tensors="pt",
            padding="longest",
            max_length=self.args.max_target_length,
            truncation=True,
            return_attention_mask=True,
        )


        inputs = [{"input_ids": d["input_ids"], "labels": d["labels"]} for d in batch]
        inputs = self.llm_collator(inputs)

        batch_phis_titles = [d["phis_titles"] for d in batch]
        batch_phis_comments = [d["phis_comments"] for d in batch]
        batch_chis_titles = [d["chis_titles"] for d in batch]
        batch_photo_his_id = [torch.LongTensor(d["photo_his_id"]) for d in batch]
        batch_comment_his_id = [torch.LongTensor(d["comment_his_id"]) for d in batch]

        photo_his_batch_idx = self._make_batch_assignees(batch_phis_titles)
        comment_his_batch_idx = self._make_batch_assignees(batch_chis_titles)

        batch_phis_titles = np.concatenate(batch_phis_titles).tolist()
        batch_phis_comments = np.concatenate(batch_phis_comments).tolist()

        batch_chis_titles = np.concatenate(batch_chis_titles).tolist()

        tot_inter = 0
        comment_pooling_index = []
        batch_chis_comments = []
        for d in batch:
            for c_list in d["chis_comments"]:
                comment_pooling_index.extend([tot_inter] * len(c_list))
                batch_chis_comments.extend(c_list)
                tot_inter += 1
        comment_pooling_index = torch.LongTensor(comment_pooling_index)

        phis_title_inputs = self.tokenizer(
            text=batch_phis_titles,
            return_tensors="pt",
            padding="longest",
            max_length=self.args.max_target_length,
            truncation=True,
            return_attention_mask=True,
        )

        phis_comment_inputs = self.tokenizer(
            text=batch_phis_comments,
            return_tensors="pt",
            padding="longest",
            max_length=self.args.max_target_length,
            truncation=True,
            return_attention_mask=True,
        )

        chis_title_inputs = self.tokenizer(
            text=batch_chis_titles,
            return_tensors="pt",
            padding="longest",
            max_length=self.args.max_target_length,
            truncation=True,
            return_attention_mask=True,
        )

        chis_comment_inputs = self.tokenizer(
            text=batch_chis_comments,
            return_tensors="pt",
            padding="longest",
            max_length=self.args.max_target_length,
            truncation=True,
            return_attention_mask=True,
        )


        batch_photo_his_id = rnn_utils.pad_sequence(batch_photo_his_id, batch_first=True)
        batch_comment_his_id = rnn_utils.pad_sequence(batch_comment_his_id, batch_first=True)

        return dict(**inputs,
                    target_text_inputs=target_text_inputs,
                    phis_title_inputs=phis_title_inputs,
                    phis_comment_inputs=phis_comment_inputs,
                    chis_title_inputs=chis_title_inputs,
                    chis_comment_inputs=chis_comment_inputs,
                    comment_pooling_index=comment_pooling_index,
                    photo_his_batch_idx=photo_his_batch_idx,
                    comment_his_batch_idx=comment_his_batch_idx,
                    photo_his_id=batch_photo_his_id,
                    comment_his_id=batch_comment_his_id,
                    comm_rank_photo_inputs=comm_rank_photo_inputs,
                    comm_rank_photo_id=comm_rank_photo_id
                    )


class FinetuneCollator(object):

    def __init__(self, args, photo_embs, comment_embs):
        self.args = args
        self.max_phis_len = args.max_phis_len
        self.max_chis_len = args.max_chis_len
        self.photo_embs = photo_embs
        self.comment_embs = comment_embs

    def _make_batch_assignees(self, batch_seqs):
        seq_sizes = torch.tensor([len(seq) for seq in batch_seqs])
        batch_index = torch.repeat_interleave(torch.arange(len(batch_seqs)), seq_sizes)
        return batch_index


    def __call__(self, batch):

        com_photo_id = None
        com_photo_text_emb = None
        comm_rank = False
        if "com_photo_id" in batch[0]:
            comm_rank = True
            com_photo_id =  torch.LongTensor([d["com_photo_id"] for d in batch])
            com_photo_text_emb = torch.from_numpy(self.photo_embs[com_photo_id]).float()

        batch_phis_pid = [torch.LongTensor(d["photo_his_pid"]) for d in batch]
        batch_phis_cid = [torch.LongTensor(d["photo_his_cid"]) for d in batch]
        batch_chis_pid = [torch.LongTensor(d["comment_his_pid"]) for d in batch]
        comment_his_batch_idx = self._make_batch_assignees(batch_chis_pid)
        tot_inter = 0
        batch_chis_cid = []
        comment_pooling_index = []
        for d in batch:
            for c_list in d["comment_his_cid"]:
                comment_pooling_index.extend([tot_inter] * len(c_list))
                batch_chis_cid.extend(c_list)
                tot_inter += 1
        comment_pooling_index = torch.LongTensor(comment_pooling_index)

        batch_phis_pid = rnn_utils.pad_sequence(batch_phis_pid, batch_first=True)
        batch_phis_cid = rnn_utils.pad_sequence(batch_phis_cid, batch_first=True)
        batch_chis_pid = rnn_utils.pad_sequence(batch_chis_pid, batch_first=True)

        batch_chis_cid = torch.LongTensor(batch_chis_cid)

        photo_his_title_emb = torch.from_numpy(self.photo_embs[batch_phis_pid]).float()
        comment_his_title_emb = torch.from_numpy(self.photo_embs[batch_chis_pid]).float()
        photo_his_comment_emb = torch.from_numpy(self.comment_embs[batch_phis_cid]).float()
        comment_his_comment_emb = torch.from_numpy(self.comment_embs[batch_chis_cid]).float()


        if not isinstance(batch[0]["labels"], np.ndarray):
            batch_labels = [d["labels"] for d in batch]
            batch_labels = torch.LongTensor(batch_labels)
        else:
            batch_labels = [torch.LongTensor(d["labels"]) for d in batch]
            batch_labels = rnn_utils.pad_sequence(batch_labels, batch_first=True)

        batch_candidates = None
        candidate_batch_idx = None
        cand_text_emb = None
        if batch[0]["candidates"] is not None:
            batch_candidates = [d["candidates"].tolist() for d in batch]
            candidate_batch_idx = self._make_batch_assignees(batch_candidates)
            batch_candidates = torch.LongTensor(np.concatenate(batch_candidates))
            if comm_rank:
                cand_text_emb = torch.from_numpy(self.comment_embs[batch_candidates]).float()
            else:
                cand_text_emb = torch.from_numpy(self.photo_embs[batch_candidates]).float()


        return dict(photo_his_pid=batch_phis_pid,
                    photo_his_title_emb=photo_his_title_emb,
                    photo_his_comment_emb=photo_his_comment_emb,

                    comment_his_pid=batch_chis_pid,
                    comment_his_title_emb=comment_his_title_emb,
                    comment_his_comment_emb=comment_his_comment_emb,
                    comment_his_batch_idx=comment_his_batch_idx,
                    comment_pooling_index=comment_pooling_index,

                    com_photo_id=com_photo_id,
                    com_photo_text_emb=com_photo_text_emb,

                    candidates=batch_candidates,
                    cand_text_emb=cand_text_emb,
                    labels=batch_labels,
                    candidate_batch_idx=candidate_batch_idx,
                    )




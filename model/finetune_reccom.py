import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch_geometric.utils import to_dense_batch

from .modeling_reccom import RecCom


class FinetuneRecComModel(nn.Module):

    def __init__(self, args, dataset):
        super().__init__()
        n_photos = len(dataset.all_photos)

        self.llm_hidden_size = args.llm_hidden_size
        self.llm_emb_size = args.llm_emb_size

        self.llm_emb_mlp = nn.Sequential(
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size // 4),
            nn.GELU(),
            nn.Linear(self.llm_hidden_size // 4, args.llm_emb_size)
        )
        self.disable_id = args.disable_id

        self.reccom = RecCom(args, n_photos, stage="finetune", disable_id=self.disable_id)

        self.temperature = args.temperature
        self.id_text_loss_weight = args.id_text_loss_weight

        if hasattr(dataset, "photo_embs"):
            self.photo_text_embs = self.weight2emb(dataset.photo_embs)

    def weight2emb(self, weight):
        llm_embedding = nn.Embedding(weight.shape[0], weight.shape[1])
        llm_embedding.weight.requires_grad = False
        llm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return llm_embedding

    def state_dict(self, **kwargs):

        state_dict =  super().state_dict(**kwargs)

        for key, value in state_dict.items():
            if (key.startswith("photo_text_embs") or key.startswith("comment_text_embs")):
                del state_dict[key]

        return state_dict


    def forward(self, photo_his_pid, comment_his_pid,
                photo_his_title_emb, comment_his_title_emb,
                photo_his_comment_emb, comment_his_comment_emb,
                comment_his_batch_idx, comment_pooling_index,
                candidates=None, candidate_batch_idx=None, cand_text_emb=None,
                com_photo_id=None, com_photo_text_emb=None,
                ):

        comment_his_comment_emb = torch_scatter.scatter_mean(comment_his_comment_emb, comment_pooling_index, dim=0)

        assert comment_his_comment_emb.shape[0] == comment_his_batch_idx.shape[0]
        comment_his_comment_emb, _ = to_dense_batch(comment_his_comment_emb, comment_his_batch_idx)

        if candidates is None and cand_text_emb is None:
            cand_text_emb = self.photo_text_embs.weight

        if photo_his_title_emb.shape[-1] > self.llm_emb_size:

            phis_rec_representions, chis_comm_rank_representions, cand_text_representions, id_text_loss = self.reccom(
                photo_his_id=photo_his_pid,
                photo_his_title_emb=self.llm_emb_mlp(photo_his_title_emb),
                photo_his_comment_emb=self.llm_emb_mlp(photo_his_comment_emb),
                comment_his_id=comment_his_pid,
                comment_his_title_emb=self.llm_emb_mlp(comment_his_title_emb),
                comment_his_comment_emb=self.llm_emb_mlp(comment_his_comment_emb),
                target_text_emb=self.llm_emb_mlp(cand_text_emb),
                com_photo_id=com_photo_id,
                com_photo_text_emb=self.llm_emb_mlp(com_photo_text_emb) if com_photo_text_emb is not None else None,
            )
        else:

            phis_rec_representions, chis_comm_rank_representions, cand_text_representions, id_text_loss = self.reccom(
                photo_his_id=photo_his_pid,
                photo_his_title_emb=photo_his_title_emb,
                photo_his_comment_emb=photo_his_comment_emb,
                comment_his_id=comment_his_pid,
                comment_his_title_emb=comment_his_title_emb,
                comment_his_comment_emb=comment_his_comment_emb,
                target_text_emb=cand_text_emb,
                com_photo_id=com_photo_id,
                com_photo_text_emb=com_photo_text_emb,
            )

        if phis_rec_representions is not None:
            phis_rec_representions = F.normalize(phis_rec_representions, dim=-1)
            if candidates is None:
                if self.disable_id:
                    all_cand_photo_id_emb = 0
                else:
                    if hasattr(self.reccom, "get_all_photo_emb"):
                        all_cand_photo_id_emb = self.reccom.get_all_photo_emb()
                    else:
                        all_cand_photo_id_emb = self.reccom.module.get_all_photo_emb()

                all_cand_photo_representions = F.normalize(all_cand_photo_id_emb + cand_text_representions, dim=-1)
                logits = torch.matmul(phis_rec_representions,
                                      all_cand_photo_representions.transpose(0, 1)) / self.temperature
                logits[:, 0] = -1000000.0  # pad index




            else:

                if self.disable_id:
                    cand_photo_id_emb = 0
                else:
                    if hasattr(self.reccom, "get_photo_emb"):
                        cand_photo_id_emb = self.reccom.get_photo_emb(candidates)
                    else:  # ddp
                        cand_photo_id_emb = self.reccom.module.get_photo_emb(candidates)

                cand_photo_representions = F.normalize(cand_photo_id_emb + cand_text_representions, dim=-1)
                cand_photo_representions, mask_cand = to_dense_batch(cand_photo_representions, candidate_batch_idx)
                logits = torch.matmul(phis_rec_representions.unsqueeze(1),
                                      cand_photo_representions.permute(0, 2, 1)).squeeze(1) / self.temperature
                mask_cand = torch.where(mask_cand, 0.0, -1000000.0)
                logits = logits + mask_cand

        elif chis_comm_rank_representions is not None:
            chis_comm_rank_representions = F.normalize(chis_comm_rank_representions, dim=-1)

            cand_comment_representions = F.normalize(cand_text_representions, dim=-1)
            cand_comment_representions, mask_cand = to_dense_batch(cand_comment_representions, candidate_batch_idx)
            logits = torch.matmul(chis_comm_rank_representions.unsqueeze(1),
                                  cand_comment_representions.permute(0, 2, 1)).squeeze(1) / self.temperature
            mask_cand = torch.where(mask_cand, 0.0, -1000000.0)
            logits = logits + mask_cand
        else:
            raise NotImplementedError


        return logits, self.id_text_loss_weight * id_text_loss



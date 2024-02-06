import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps=1e-8,
        residual=True,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        self.residual = residual

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.out_linear = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, query_states, key_value_states, attention_mask):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_value_states)
        mixed_value_layer = self.value(key_value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads query_seq_len key_seq_len] scores
        # [batch_size 1 1 key_seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        attention_probs = attention_probs.type_as(value_layer)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.out_linear(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        if self.residual:
            hidden_states = self.LayerNorm(hidden_states + query_states)
        else:
            hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class FeedForward(nn.Module):

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.linear_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):

        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.linear_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, query_states, key_value_states, attention_mask):
        attention_output = self.multi_head_attention(query_states, key_value_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output

class Transformer(nn.Module):

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.1,
        attn_dropout_prob=0.1,
        hidden_act="gelu",
        layer_norm_eps=1e-8,
    ):
        super(Transformer, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, key_value_states, attention_mask, output_all_layers=True):

        all_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, key_value_states, attention_mask)
            if output_all_layers:
                all_layers.append(hidden_states)
        if not output_all_layers:
            all_layers.append(hidden_states)
        return all_layers



class AdditiveAttention(nn.Module):
    def __init__(self,
                hidden_size=64,
                inner_size=256,
                hidden_dropout_prob=0.1,
                layer_norm_eps=1e-8,
    ):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.Dropout(hidden_dropout_prob),
            nn.LayerNorm(inner_size, eps=layer_norm_eps),
            nn.GELU(),
            nn.Linear(inner_size, 1),
        )

    def forward(self, hidden_states, attention_mask):

        weights = self.attn(hidden_states).squeeze(-1)
        weights = weights + attention_mask.squeeze(1).squeeze(1)
        weights = torch.softmax(weights, dim=-1).type_as(hidden_states)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)



class RecCom(nn.Module):

    def __init__(self, args, n_photos, stage="pretrain", disable_id=False):
        super(RecCom, self).__init__()

        # load parameters info
        self.n_layers = args.n_layers
        self.n_mix_layers = args.n_mix_layers
        self.n_heads = args.n_heads
        self.hidden_size = args.hidden_size
        self.inner_size = args.inner_size  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.attn_dropout_prob = args.attn_dropout_prob
        self.hidden_act = args.hidden_act

        self.max_position = args.max_position

        self.n_photos = n_photos
        self.llm_emb_size = args.llm_emb_size
        self.stage = stage
        self.disable_id = disable_id

        self.temperature = args.temperature
        self.contrast_neg_num = args.contrast_neg_num
        self.id_text_loss_weight =args.id_text_loss_weight

        # define layers and loss
        if self.stage=="pretrain" or self.disable_id:
            self.photo_embedding = None
        else:
            self.photo_embedding = nn.Embedding(
                self.n_photos, self.hidden_size, padding_idx=0
            )
            self.gate = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.pposition_embedding = nn.Embedding(self.max_position, self.hidden_size)
        self.cposition_embedding = nn.Embedding(self.max_position, self.hidden_size)

        self.photo_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-8)
        self.comment_layernorm = nn.LayerNorm(self.hidden_size, eps=1e-8)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.llm_adapter = nn.Linear(self.llm_emb_size, self.hidden_size, bias=False)
        self.pc_merge = nn.Linear(2 * self.hidden_size, self.hidden_size)


        self.phis_encoder = Transformer(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act
        )
        self.chis_encoder = copy.deepcopy(self.phis_encoder)

        self.phis_mix_layer = Transformer(
            n_layers=self.n_mix_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act
        )
        self.chis_mix_layer = copy.deepcopy(self.phis_mix_layer)

        self.p_linear = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.c_linear = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.comm_rank_att = MultiHeadAttention(
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
        )

        self.rec_att = AdditiveAttention(
            hidden_size=self.hidden_size,
            inner_size=2 * self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

        # parameters initialization
        self.apply(self._init_weights)



    def _init_weights(self, module):
        """Initialize the weights"""

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -1000000.0)
        return extended_attention_mask

    def get_pooling_mask(self, seq):
        mask = (seq != 0)
        # mask = mask.long()
        return mask

    def mean_pooling(self, hidden_states, mask):

        hidden_states = hidden_states * mask.unsqueeze(-1)
        mean_output = hidden_states.sum(dim=1) / mask.sum(dim=-1, keepdim=True)

        return mean_output


    def id_text_contrast_loss(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)

        N = x.size(0)


        pos_logits = torch.sum( x * y, dim=-1, keepdim=True)

        tmp = torch.rand((N,N), device=x.device)
        tmp.fill_diagonal_(-1000000.00)

        _, neg_idx = torch.topk(tmp, 2*self.contrast_neg_num, dim=1)  # N x 2*neg_num

        neg_x = x[neg_idx[:,:self.contrast_neg_num]]
        neg_y = y[neg_idx[:,self.contrast_neg_num:]]

        neg_y_logits = torch.sum( x.unsqueeze(1) * neg_y, dim=-1)

        neg_x_logits = torch.sum( y.unsqueeze(1) * neg_x, dim=-1)

        logits_x2y = torch.cat([pos_logits, neg_y_logits], dim=1) / self.temperature
        logits_y2x = torch.cat([pos_logits, neg_x_logits], dim=1) / self.temperature

        labels = torch.zeros(N, dtype=torch.long, device=x.device)

        return (F.cross_entropy(logits_x2y, labels) + F.cross_entropy(logits_y2x, labels)) / 2


    def forward(self,
                photo_his_id,
                photo_his_title_emb,
                photo_his_comment_emb,
                comment_his_id,
                comment_his_title_emb,
                comment_his_comment_emb,
                target_text_emb,
                com_photo_id=None,
                com_photo_text_emb=None,
                ):
        if self.stage == "pretrain":
            phis_len = photo_his_id.size(1)
            chis_len = comment_his_id.size(1)

            photo_his_pids = torch.randperm(self.max_position, dtype=torch.long, device=photo_his_id.device)[:phis_len]
            comment_his_pids = torch.randperm(self.max_position, dtype=torch.long, device=comment_his_id.device)[:chis_len]
            photo_his_pids, _ = torch.sort(photo_his_pids)
            comment_his_pids, _ = torch.sort(comment_his_pids)

        else:
            photo_his_pids = torch.arange(
                photo_his_id.size(1), dtype=torch.long, device=photo_his_id.device
            )
            comment_his_pids = torch.arange(
                comment_his_id.size(1), dtype=torch.long, device=comment_his_id.device
            )

        target_representions = self.llm_adapter(target_text_emb)

        photo_his_pids = photo_his_pids.unsqueeze(0).expand_as(photo_his_id)
        comment_his_pids = comment_his_pids.unsqueeze(0).expand_as(comment_his_id)

        phis_position_emb = self.pposition_embedding(photo_his_pids)
        chis_position_emb = self.cposition_embedding(comment_his_pids)

        if self.stage=="pretrain" or self.disable_id:
            phis_id_emb = None
            chis_id_emb = None
        else:
            phis_id_emb = self.photo_embedding(photo_his_id)
            chis_id_emb = self.photo_embedding(comment_his_id)

        photo_his_title_emb = self.llm_adapter(photo_his_title_emb)
        photo_his_comment_emb = self.llm_adapter(photo_his_comment_emb)
        comment_his_title_emb = self.llm_adapter(comment_his_title_emb)
        comment_his_comment_emb = self.llm_adapter(comment_his_comment_emb)
        photo_his_text_emb = self.pc_merge(torch.cat([photo_his_title_emb, photo_his_comment_emb], dim=-1))
        comment_his_text_emb = self.pc_merge(torch.cat([comment_his_title_emb, comment_his_comment_emb], dim=-1))


        id_text_loss = 0
        if self.training and self.stage != "pretrain" and not self.disable_id and self.id_text_loss_weight > 0:

            phis_select_mask = self.get_pooling_mask(photo_his_id).unsqueeze(-1)
            all_phis_id_emb = torch.masked_select(phis_id_emb.contiguous(),
                                                  phis_select_mask).reshape(-1, phis_id_emb.size(-1))
            all_phis_text_emb = torch.masked_select(photo_his_text_emb.contiguous(),
                                                    phis_select_mask).reshape(-1, photo_his_text_emb.size(-1))

            chis_select_mask = self.get_pooling_mask(comment_his_id).unsqueeze(-1)
            all_chis_id_emb = torch.masked_select(chis_id_emb.contiguous(),
                                                  chis_select_mask).reshape(-1, chis_id_emb.size(-1))
            all_chis_text_emb = torch.masked_select(comment_his_text_emb.contiguous(),
                                                    chis_select_mask).reshape(-1, comment_his_text_emb.size(-1))

            id_text_loss = self.id_text_contrast_loss(all_phis_id_emb, all_phis_text_emb) + \
                            self.id_text_contrast_loss(all_chis_id_emb, all_chis_text_emb)


        if self.stage=="pretrain" or self.disable_id:
            phis_input_emb = phis_position_emb + photo_his_text_emb
            chis_input_emb = chis_position_emb + comment_his_text_emb
        else:
            phis_input_emb = self.gate * phis_id_emb + photo_his_text_emb + phis_position_emb
            chis_input_emb = self.gate * chis_id_emb + comment_his_text_emb + chis_position_emb


        phis_input_emb = self.photo_layernorm(phis_input_emb)
        phis_input_emb = self.dropout(phis_input_emb)

        chis_input_emb = self.comment_layernorm(chis_input_emb)
        chis_input_emb = self.dropout(chis_input_emb)

        photo_his_encode_mask = self.get_attention_mask(photo_his_id)
        comment_his_encode_mask = self.get_attention_mask(comment_his_id)

        phis_hidden_states = self.phis_encoder(
            phis_input_emb, phis_input_emb, photo_his_encode_mask
        )[-1]

        chis_hidden_states = self.chis_encoder(
            chis_input_emb, chis_input_emb, comment_his_encode_mask
        )[-1]

        photo_his_mix_mask = self.get_attention_mask(photo_his_id, bidirectional=True)
        comment_his_mix_mask = self.get_attention_mask(comment_his_id, bidirectional=True)

        phis_mix_hidden_states = self.phis_mix_layer(
            phis_hidden_states, chis_hidden_states, comment_his_mix_mask
        )[-1]

        chis_mix_hidden_states = self.chis_mix_layer(
            chis_hidden_states, phis_hidden_states, photo_his_mix_mask
        )[-1]


        phis_hidden_states = torch.cat((phis_hidden_states, phis_mix_hidden_states), -1)
        chis_hidden_states = torch.cat((chis_hidden_states, chis_mix_hidden_states), -1)

        phis_hidden_states = self.p_linear(phis_hidden_states)
        chis_hidden_states = self.c_linear(chis_hidden_states)

        phis_rec_representions = None
        chis_rec_representions = None
        phis_comm_rank_representions = None
        chis_comm_rank_representions = None

        B = photo_his_id.shape[0]

        if com_photo_id != None:

            if self.stage == "pretrain" or self.disable_id:
                com_photo_emb = self.llm_adapter(com_photo_text_emb)
            else:
                com_photo_id_emb = self.photo_embedding(com_photo_id)
                com_photo_text_emb = self.llm_adapter(com_photo_text_emb)
                com_photo_emb = self.gate * com_photo_id_emb + com_photo_text_emb


            com_num = com_photo_id.shape[0]
            rec_num = B - com_num

            if rec_num==0:

                chis_comm_rank_representions = self.comm_rank_att(com_photo_emb.unsqueeze(1),
                                                                     chis_hidden_states,
                                                                     comment_his_mix_mask
                                                                     ).squeeze(1)

                if self.stage == "pretrain":
                    phis_comm_rank_representions = self.comm_rank_att(com_photo_emb.unsqueeze(1),
                                                                 phis_hidden_states,
                                                                 photo_his_mix_mask
                                                                 ).squeeze(1)

                    tmp_zero = self.ensure_param_grad(input_type=chis_comm_rank_representions.dtype,
                                                      input_device=chis_comm_rank_representions.device)

                    chis_comm_rank_representions = chis_comm_rank_representions + tmp_zero

            else:


                phis_rec_representions = self.rec_att(phis_hidden_states[:rec_num], photo_his_mix_mask[:rec_num])
                chis_rec_representions = self.rec_att(chis_hidden_states[:rec_num], comment_his_mix_mask[:rec_num])


                phis_comm_rank_representions = self.comm_rank_att(com_photo_emb.unsqueeze(1),
                                                                 phis_hidden_states[-com_num:],
                                                                 photo_his_mix_mask[-com_num:]
                                                                 ).squeeze(1)
                chis_comm_rank_representions = self.comm_rank_att(com_photo_emb.unsqueeze(1),
                                                                 chis_hidden_states[-com_num:],
                                                                 comment_his_mix_mask[-com_num:]
                                                                 ).squeeze(1)
        else:

            phis_rec_representions = self.rec_att(phis_hidden_states, photo_his_mix_mask)
            if self.stage == "pretrain":
                chis_rec_representions = self.rec_att(chis_hidden_states, comment_his_mix_mask)


                tmp_zero = self.ensure_param_grad(input_type = phis_rec_representions.dtype,
                                                  input_device = phis_rec_representions.device)

                phis_rec_representions = phis_rec_representions + tmp_zero
                # chis_rec_representions = chis_rec_representions + tmp_zero

        if self.stage == "pretrain":
            return (phis_rec_representions, chis_rec_representions,
                    phis_comm_rank_representions, chis_comm_rank_representions, target_representions)
        else:
            return phis_rec_representions, chis_comm_rank_representions, target_representions, id_text_loss


    def ensure_param_grad(self, input_type, input_device):

        qshape = (1,1,self.hidden_size)
        kvshape = (1,2,self.hidden_size)
        maskshape = (1,1,1,2)
        zeroshape = (1,self.hidden_size)

        q = torch.randn(*qshape, dtype=input_type, device=input_device)
        kv = torch.randn(*kvshape, dtype=input_type, device=input_device)
        mask = torch.zeros(*maskshape, dtype=input_type, device=input_device)
        zero = torch.zeros(*zeroshape, dtype=input_type, device=input_device)

        tmp = self.rec_att(kv, mask) + self.comm_rank_att(q, kv, mask).squeeze(1)


        return tmp * zero

    def get_all_photo_emb(self, ):

        if self.disable_id:
            return 0

        return self.gate * self.photo_embedding.weight


    def get_photo_emb(self, photo_id):

        if self.disable_id:
            return 0

        return self.gate * self.photo_embedding(photo_id)





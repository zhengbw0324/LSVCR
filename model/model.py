
import math
import copy
import warnings
import re
import sys

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from .modeling_chatglm import *
from .modeling_reccom import *
from torch_geometric.utils import to_dense_batch
import torch_scatter
from torch.nn import functional as F
import torch.distributed as dist

from transformers import LlamaForCausalLM


class RecComModel(ChatGLMPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['reccom.c_linear.weight', 'reccom.comm_rank_att.value.weight', 'reccom.chis_mix_layer.layer.0.multi_head_attention.value.weight', 'reccom.chis_mix_layer.layer.0.feed_forward.linear_1.weight', 'reccom.p_linear.bias', 'llm_emb_mlp.0.bias', 'reccom.phis_mix_layer.layer.0.multi_head_attention.query.bias', 'reccom.phis_encoder.layer.0.multi_head_attention.LayerNorm.bias', 'reccom.phis_encoder.layer.0.feed_forward.LayerNorm.weight', 'reccom.chis_encoder.layer.1.multi_head_attention.LayerNorm.weight', 'reccom.phis_encoder.layer.1.feed_forward.linear_1.weight', 'reccom.phis_mix_layer.layer.0.multi_head_attention.out_linear.bias', 'reccom.rec_att.attn.2.bias', 'reccom.phis_mix_layer.layer.0.multi_head_attention.key.weight', 'reccom.pc_merge.weight', 'reccom.chis_encoder.layer.0.multi_head_attention.LayerNorm.bias', 'reccom.comment_layernorm.weight', 'reccom.chis_encoder.layer.1.multi_head_attention.value.bias', 'reccom.phis_encoder.layer.0.feed_forward.linear_2.weight', 'reccom.phis_encoder.layer.1.feed_forward.LayerNorm.bias', 'reccom.llm_adapter.weight', 'reccom.chis_encoder.layer.1.multi_head_attention.query.bias', 'reccom.chis_encoder.layer.0.multi_head_attention.query.weight', 'reccom.chis_mix_layer.layer.0.feed_forward.LayerNorm.weight', 'reccom.phis_encoder.layer.0.feed_forward.linear_2.bias', 'reccom.phis_encoder.layer.0.multi_head_attention.value.weight', 'reccom.chis_mix_layer.layer.0.multi_head_attention.LayerNorm.bias', 'reccom.phis_encoder.layer.0.multi_head_attention.query.weight', 'reccom.p_linear.weight', 'reccom.chis_encoder.layer.0.feed_forward.linear_2.weight', 'reccom.chis_encoder.layer.1.multi_head_attention.value.weight', 'reccom.rec_att.attn.0.weight', 'reccom.phis_mix_layer.layer.0.feed_forward.linear_1.weight', 'reccom.comm_rank_att.out_linear.weight', 'reccom.chis_encoder.layer.1.multi_head_attention.out_linear.weight', 'reccom.phis_encoder.layer.1.feed_forward.linear_2.bias', 'reccom.chis_encoder.layer.0.multi_head_attention.out_linear.weight', 'reccom.chis_encoder.layer.1.multi_head_attention.out_linear.bias', 'reccom.chis_mix_layer.layer.0.multi_head_attention.out_linear.weight', 'reccom.phis_encoder.layer.0.feed_forward.linear_1.weight', 'reccom.cposition_embedding.weight', 'reccom.chis_encoder.layer.1.multi_head_attention.query.weight', 'reccom.phis_encoder.layer.1.multi_head_attention.LayerNorm.bias', 'reccom.phis_mix_layer.layer.0.feed_forward.LayerNorm.weight', 'reccom.chis_encoder.layer.1.feed_forward.linear_1.bias', 'reccom.phis_encoder.layer.0.feed_forward.linear_1.bias', 'reccom.phis_mix_layer.layer.0.multi_head_attention.query.weight', 'reccom.phis_encoder.layer.1.multi_head_attention.value.weight', 'reccom.comment_layernorm.bias', 'reccom.phis_encoder.layer.0.multi_head_attention.key.bias', 'reccom.phis_encoder.layer.1.multi_head_attention.LayerNorm.weight', 'reccom.phis_encoder.layer.1.multi_head_attention.query.bias', 'reccom.comm_rank_att.key.bias', 'reccom.chis_encoder.layer.1.feed_forward.linear_1.weight', 'reccom.phis_encoder.layer.1.feed_forward.linear_1.bias', 'reccom.phis_encoder.layer.0.multi_head_attention.query.bias', 'reccom.phis_encoder.layer.1.multi_head_attention.query.weight', 'contrast_adapter.weight', 'reccom.comm_rank_att.value.bias', 'reccom.phis_mix_layer.layer.0.feed_forward.linear_2.weight', 'reccom.phis_encoder.layer.0.multi_head_attention.out_linear.weight', 'reccom.phis_encoder.layer.1.feed_forward.linear_2.weight', 'reccom.chis_mix_layer.layer.0.multi_head_attention.query.weight', 'reccom.pposition_embedding.weight', 'reccom.phis_mix_layer.layer.0.multi_head_attention.LayerNorm.weight', 'reccom.chis_encoder.layer.1.feed_forward.LayerNorm.bias', 'reccom.chis_mix_layer.layer.0.multi_head_attention.out_linear.bias', 'reccom.chis_mix_layer.layer.0.feed_forward.linear_2.bias', 'reccom.chis_mix_layer.layer.0.multi_head_attention.key.weight', 'reccom.chis_mix_layer.layer.0.multi_head_attention.value.bias', 'reccom.phis_encoder.layer.0.multi_head_attention.LayerNorm.weight', 'reccom.chis_encoder.layer.0.multi_head_attention.key.bias', 'reccom.comm_rank_att.query.bias', 'reccom.phis_mix_layer.layer.0.multi_head_attention.key.bias', 'reccom.c_linear.bias', 'reccom.phis_encoder.layer.1.feed_forward.LayerNorm.weight', 'reccom.chis_mix_layer.layer.0.feed_forward.linear_1.bias', 'reccom.phis_mix_layer.layer.0.feed_forward.LayerNorm.bias', 'reccom.chis_encoder.layer.0.feed_forward.linear_2.bias', 'reccom.chis_encoder.layer.1.feed_forward.linear_2.weight', 'reccom.pc_merge.bias', 'reccom.chis_mix_layer.layer.0.multi_head_attention.query.bias', 'llm_emb_mlp.2.weight', 'reccom.phis_mix_layer.layer.0.feed_forward.linear_1.bias', 'reccom.rec_att.attn.4.weight', 'reccom.phis_encoder.layer.1.multi_head_attention.out_linear.weight', 'reccom.chis_encoder.layer.0.multi_head_attention.LayerNorm.weight', 'reccom.comm_rank_att.out_linear.bias', 'reccom.phis_encoder.layer.0.multi_head_attention.value.bias', 'reccom.phis_mix_layer.layer.0.multi_head_attention.value.weight', 'reccom.chis_encoder.layer.0.feed_forward.linear_1.weight', 'reccom.rec_att.attn.4.bias', 'reccom.comm_rank_att.key.weight', 'reccom.phis_encoder.layer.1.multi_head_attention.out_linear.bias', 'reccom.chis_encoder.layer.0.feed_forward.LayerNorm.bias', 'reccom.chis_encoder.layer.0.feed_forward.LayerNorm.weight', 'reccom.phis_mix_layer.layer.0.multi_head_attention.value.bias', 'reccom.chis_mix_layer.layer.0.feed_forward.linear_2.weight', 'reccom.chis_encoder.layer.0.multi_head_attention.value.bias', 'reccom.chis_encoder.layer.1.feed_forward.linear_2.bias', 'reccom.phis_encoder.layer.0.feed_forward.LayerNorm.bias', 'reccom.chis_encoder.layer.0.multi_head_attention.key.weight', 'reccom.phis_encoder.layer.0.multi_head_attention.out_linear.bias', 'reccom.comm_rank_att.LayerNorm.weight', 'reccom.photo_layernorm.bias', 'contrast_adapter.bias', 'reccom.chis_mix_layer.layer.0.multi_head_attention.key.bias', 'reccom.chis_encoder.layer.1.multi_head_attention.key.weight', 'reccom.photo_embedding.weight', 'reccom.phis_mix_layer.layer.0.multi_head_attention.LayerNorm.bias', 'reccom.phis_encoder.layer.1.multi_head_attention.key.weight', 'reccom.phis_encoder.layer.1.multi_head_attention.key.bias', 'reccom.rec_att.attn.2.weight', 'reccom.chis_encoder.layer.1.multi_head_attention.key.bias', 'reccom.phis_encoder.layer.1.multi_head_attention.value.bias', 'llm_emb_mlp.2.bias', 'reccom.phis_encoder.layer.0.multi_head_attention.key.weight', 'reccom.chis_encoder.layer.0.feed_forward.linear_1.bias', 'reccom.chis_mix_layer.layer.0.feed_forward.LayerNorm.bias', 'reccom.chis_encoder.layer.0.multi_head_attention.value.weight', 'reccom.chis_encoder.layer.1.multi_head_attention.LayerNorm.bias', 'reccom.comm_rank_att.LayerNorm.bias', 'reccom.phis_mix_layer.layer.0.feed_forward.linear_2.bias', 'reccom.chis_encoder.layer.0.multi_head_attention.out_linear.bias', 'llm_emb_mlp.0.weight', 'reccom.rec_att.attn.0.bias', 'reccom.comm_rank_att.query.weight', 'reccom.photo_layernorm.weight', 'reccom.chis_encoder.layer.0.multi_head_attention.query.bias', 'reccom.phis_mix_layer.layer.0.multi_head_attention.out_linear.weight', 'reccom.chis_mix_layer.layer.0.multi_head_attention.LayerNorm.weight', 'reccom.chis_encoder.layer.1.feed_forward.LayerNorm.weight']
    _no_split_modules = ["GLMBlock","TransformerLayer"]
    def __init__(self, config: ChatGLMConfig,
                 n_photos, args, empty_init=False, device=None):
        super().__init__(config)

        self.max_sequence_length = config.max_length
        self.n_photos = n_photos
        self.llm_emb_size = args.llm_emb_size

        self.llm_emb_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.GELU(),
            nn.Linear(config.hidden_size // 4 , args.llm_emb_size)
        )

        self.contrast_adapter = nn.Linear(config.hidden_size, args.hidden_size)

        self.reccom = RecCom(args, n_photos)

        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)

        self.config = config
        self.args = args
        self.contrast_loss_weight = args.contrast_loss_weight
        self.rec_loss_weight = args.rec_loss_weight
        self.temperature = args.temperature
        self.quantized = False

        if self.config.quantization_bit:
            self.quantize(self.config.quantization_bit, empty_init=True)

        self.post_init()


    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def mean_pooling(self, hidden_states, mask):

        hidden_states = hidden_states * mask.unsqueeze(-1)
        mean_output = hidden_states.sum(dim=1) / mask.sum(dim=-1, keepdim=True)

        return mean_output

    def last_pooling(self, hidden_states, mask):


        last_index = mask.argmax(dim=-1) - 1
        last_index = last_index.view(-1, 1, 1).expand(-1, -1, hidden_states.shape[-1])
        last_hidden_state = torch.gather(hidden_states, dim=1, index=last_index).squeeze(1)

        return last_hidden_state

    def get_text_emb(self, inputs):

        hidden_states = self.transformer(**inputs)[0]
        text_emb = self.mean_pooling(hidden_states.transpose(0, 1).contiguous(), inputs['attention_mask'])
        text_emb = self.llm_emb_mlp(text_emb)

        return text_emb

    def get_text_hidden_states(self, inputs):

        hidden_states = self.transformer(**inputs)[0]
        hidden_states = self.mean_pooling(hidden_states.transpose(0, 1).contiguous(), inputs['attention_mask'])

        text_emb = self.llm_emb_mlp(hidden_states)

        return text_emb, hidden_states


    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            return_last_logit: Optional[bool] = False,
            target_text_inputs=None,
            phis_title_inputs=None,
            phis_comment_inputs=None,
            chis_title_inputs=None,
            chis_comment_inputs=None,
            comment_pooling_index=None,
            photo_his_batch_idx=None,
            comment_his_batch_idx=None,
            photo_his_id=None,
            comment_his_id=None,
            comm_rank_photo_inputs=None,
            comm_rank_photo_id=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.transformer.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if phis_title_inputs is None:
            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=hidden_states,
                attentions=transformer_outputs.attentions,
            )


        hidden_states = hidden_states.transpose(0, 1).contiguous()
        hidden_states_mask = copy.deepcopy(attention_mask)
        hidden_states_mask[labels == -100] = 0
        llm_his_representions = self.last_pooling(hidden_states, hidden_states_mask)
        llm_his_representions = self.contrast_adapter(llm_his_representions)

        target_text_emb = self.get_text_emb(target_text_inputs)

        photo_his_title_emb = self.get_text_emb(phis_title_inputs)
        photo_his_title_emb, _ = to_dense_batch(photo_his_title_emb, photo_his_batch_idx)

        photo_his_comment_emb = self.get_text_emb(phis_comment_inputs)
        photo_his_comment_emb, _ = to_dense_batch(photo_his_comment_emb, photo_his_batch_idx)

        comment_his_title_emb = self.get_text_emb(chis_title_inputs)
        comment_his_title_emb, _ = to_dense_batch(comment_his_title_emb, comment_his_batch_idx)

        comment_his_comment_emb = self.get_text_emb(chis_comment_inputs)
        comment_his_comment_emb = torch_scatter.scatter_mean(comment_his_comment_emb, comment_pooling_index, dim=0)

        assert comment_his_comment_emb.shape[0] == comment_his_batch_idx.shape[0]
        comment_his_comment_emb, _ = to_dense_batch(comment_his_comment_emb, comment_his_batch_idx)

        com_photo_text_emb = None
        if comm_rank_photo_inputs is not None:
            com_photo_text_emb = self.get_text_emb(comm_rank_photo_inputs)

        (phis_rec_representions, chis_rec_representions,
         phis_comm_rank_representions, chis_comm_rank_representions,
         target_representions) = self.reccom(
            photo_his_id=photo_his_id,
            photo_his_title_emb=photo_his_title_emb,
            photo_his_comment_emb=photo_his_comment_emb,
            comment_his_id=comment_his_id,
            comment_his_title_emb=comment_his_title_emb,
            comment_his_comment_emb=comment_his_comment_emb,
            target_text_emb=target_text_emb,
            com_photo_id=comm_rank_photo_id,
            com_photo_text_emb=com_photo_text_emb,
        )

        llm_his_representions = F.normalize(llm_his_representions, dim=-1)
        target_representions = F.normalize(target_representions, dim=-1)

        if phis_rec_representions is not None and chis_comm_rank_representions is not None:
            rec_num = phis_rec_representions.shape[0]
            com_num = chis_comm_rank_representions.shape[0]
            assert (rec_num + com_num) == photo_his_id.shape[0]

            phis_rec_representions = F.normalize(phis_rec_representions, dim=-1)
            chis_rec_representions = F.normalize(chis_rec_representions, dim=-1)
            phis_comm_rank_representions = F.normalize(phis_comm_rank_representions, dim=-1)
            chis_comm_rank_representions = F.normalize(chis_comm_rank_representions, dim=-1)

            rec_llm_his_representions = llm_his_representions[:rec_num]
            com_rank_llm_his_representions = llm_his_representions[-com_num:]

            # B1, 1, d   * B1, d, 2   =>  B1, 2
            rec_bhv_contrast = torch.matmul(rec_llm_his_representions.unsqueeze(1),
                                            torch.stack([phis_rec_representions,
                                                         chis_rec_representions], dim=-1)).squeeze(1)
            # B2, 1, d   * B2, d, 2   =>  B2, 2
            com_rank_bhv_contrast = torch.matmul(com_rank_llm_his_representions.unsqueeze(1),
                                                 torch.stack([chis_comm_rank_representions,
                                                              phis_comm_rank_representions], dim=-1)).squeeze(1)
            # B,2
            bhv_contrast_scores = torch.cat([rec_bhv_contrast, com_rank_bhv_contrast], dim=0)


            reccom_representions = torch.cat([phis_rec_representions, chis_comm_rank_representions], dim=0)

        elif phis_rec_representions is not None:

            phis_rec_representions = F.normalize(phis_rec_representions, dim=-1)
            chis_rec_representions = F.normalize(chis_rec_representions, dim=-1)
            # B,2
            bhv_contrast_scores = torch.matmul(llm_his_representions.unsqueeze(1),
                                               torch.stack([phis_rec_representions,
                                                            chis_rec_representions], dim=-1)).squeeze(1)
            reccom_representions = phis_rec_representions

        elif chis_comm_rank_representions is not None:

            phis_comm_rank_representions = F.normalize(phis_comm_rank_representions, dim=-1)
            chis_comm_rank_representions = F.normalize(chis_comm_rank_representions, dim=-1)
            # B, 1, d   * B, d, 2   =>  B, 2
            bhv_contrast_scores = torch.matmul(llm_his_representions.unsqueeze(1),
                                               torch.stack([chis_comm_rank_representions,
                                                            phis_comm_rank_representions], dim=-1)).squeeze(1)
            reccom_representions = chis_comm_rank_representions
        else:
            raise RuntimeError


        bhv_contrast_scores = bhv_contrast_scores.to(torch.float32) / self.temperature
        bhv_contrast_labels = torch.tensor([0] * bhv_contrast_scores.shape[0],
                                            device=bhv_contrast_scores.device, dtype=torch.long)

        ddp_contrast_loss_fct = DistributedContrastiveLoss(temperature = self.temperature)

        contrast_loss = (F.cross_entropy(bhv_contrast_scores, bhv_contrast_labels) +
                         ddp_contrast_loss_fct(reccom_representions, llm_his_representions) / 2 +
                         ddp_contrast_loss_fct(llm_his_representions, reccom_representions) / 2 ).to(hidden_states.dtype)

        rec_loss = ddp_contrast_loss_fct(reccom_representions, target_representions).to(hidden_states.dtype)

        loss = loss + self.rec_loss_weight * rec_loss + self.contrast_loss_weight * contrast_loss

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # update position ids
        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        model_kwargs["is_first_forward"] = False
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if position_ids is None:
            position_ids = self.get_position_ids(input_ids, device=input_ids.device)
        if not is_first_forward:
            if past_key_values is not None:
                position_ids = position_ids[..., -1:]
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "return_last_logit": True,
            "use_cache": use_cache
        }

    @staticmethod
    def _reorder_cache(
            past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        return tuple(
            (
                layer_past[0].index_select(1, beam_idx.to(layer_past[0].device)),
                layer_past[1].index_select(1, beam_idx.to(layer_past[1].device)),
            )
            for layer_past in past
        )

    def quantize(self, bits: int, empty_init=False, device=None, **kwargs):
        if bits == 0:
            return

        from .quantization import quantize

        if self.quantized:
            logger.info("Already quantized.")
            return self

        self.quantized = True

        self.config.quantization_bit = bits

        self.transformer.encoder = quantize(self.transformer.encoder, bits, empty_init=empty_init, device=device,
                                            **kwargs)
        return self



class AllGatherFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]

        dist.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone().contiguous()
        dist.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = dist.get_rank() * ctx.batch_size
        idx_to = (dist.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to].contiguous()



class DistributedContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        assert dist.is_initialized(), "Distributed training has not been properly initialized."
        super().__init__()
        self.word_size = dist.get_world_size()
        self.local_rank = dist.get_rank()
        self.temperature = temperature

    def forward(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)


        B = y.shape[0]
        # word_size * B, d
        dist_y = AllGatherFunction.apply(y)

        # B, word_size * B
        logits = torch.matmul(x, dist_y.transpose(0, 1).contiguous()).to(torch.float32) / self.temperature
        labels = torch.arange(B, device=x.device, dtype=torch.long) + self.local_rank * B

        return F.cross_entropy(logits, labels)



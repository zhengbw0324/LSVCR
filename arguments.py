from dataclasses import dataclass, field
from typing import Optional, List
import transformers


@dataclass
class ModelArguments:

    model_name_or_path: Optional[str] = field(
        default="/THUDM/chatglm3-6b/",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    lora: bool = field(
        default=False,
        metadata={"help": "Whether to use lora training."},
    )
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_target_modules: Optional[str] = field(default="query_key_value,dense,dense_h_to_4h,dense_4h_to_h")
    lora_modules_to_save: Optional[str] = field(default="reccom,llm_emb_mlp,contrast_adapter")


    n_layers: Optional[int] = field(default=2)
    n_mix_layers: Optional[int] = field(default=1)
    n_heads: Optional[int] = field(default=2)
    hidden_size: Optional[int] = field(default=64)
    inner_size: Optional[int] = field(default=256)
    hidden_dropout_prob: Optional[float] = field(default=0.3)
    attn_dropout_prob: Optional[float] = field(default=0.3)
    hidden_act: Optional[str] = field(default="gelu")

    contrast_loss_weight: Optional[float] = field(default=0.5)
    rec_loss_weight: Optional[float] = field(default=1.0)
    temperature: Optional[float] = field(default=0.07)

    max_position: Optional[int] = field(default=50)

    contrast_neg_num: Optional[int] = field(default=99)
    id_text_loss_weight: Optional[float] = field(default=0.5)
    disable_id: Optional[bool] = field(default=False)

    llm_vocab_size: Optional[int] = field(default=65024)
    llm_hidden_size: Optional[int] = field(default=4096)
    llm_emb_size: Optional[int] = field(default=256)

    pretrain_checkpoint: Optional[str] = field(
        default="./ckpt/LSVCR/adapter_model.bin",
        metadata={"help": "Parameters of pretrained model."}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_path: Optional[str] = field(
        default="./data/",
        metadata={"help": "The data directory."}
    )

    model_max_length: Optional[int] = field(
        default=1024*5,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )

    max_source_length: Optional[int] = field(
        default=1024*4 - 256,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={"help": "The maximum total sequence length for target text after tokenization."},
    )

    max_phis_len: Optional[int] = field(
        default=20,
        metadata={"help": "The max number of photo in history sequence."},
    )
    max_chis_len: Optional[int] = field(
        default=20,
        metadata={"help": "The max number of comment records in history sequence."},
    )

    train_num: Optional[int] = field(
        default=300400 + 128000 + 128000,
        metadata={"help": "The max number of training data."},
    )

    val_num: Optional[int] = field(
        default=20000,
        metadata={"help": "The max number of validation data."},
    )

    neg_photo_num: Optional[int] = field(default=-1)
    neg_comment_num: Optional[int] = field(default=9)
    max_candidate_num: Optional[int] = field(
        default=100,
        metadata={"help": "The max number of candidate photo/comment when test."},
    )

    instruction_emb: bool = field(
        default=False,
        metadata={"help": "Whether to use instruction for photo or comment embedding."},
    )

    finetune_task: Optional[str] = field(
        default="Rec",
        metadata={"help": "Rec or CommRank."}
    )

    photo_emb_file: Optional[str] = field(
        default="photo_embs.npy",
    )
    comment_emb_file: Optional[str] = field(
        default="comment_embs.npy",
    )

    phis_aug_p: Optional[float] = field(default=0.5)
    chis_aug_p: Optional[float] = field(default=0.5)
    inter_aug_p: Optional[float] = field(default=0.3)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."},
    )
    load_best_model_at_end: Optional[bool] = field(default=True)
    save_total_limit: Optional[int] = field(default=10)
    logging_steps: float = field(default=2)
    report_to: Optional[List[str]] = field(default="none")
    metrics: str = field(
        default="recall@1,recall@5,recall@10,ndcg@5,ndcg@10,mrr@5,mrr@10",
        metadata={"help": "Metrics used in validation and testing, separated by commas."}
    )
    valid_metric: str = field(
        default="ndcg@5",
        metadata={"help": "Metric used to select the best model during the validation process."}
    )



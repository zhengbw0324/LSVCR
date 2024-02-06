import argparse
import os

import sys

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, AutoTokenizer

from fttrainer import FtTrainer
from utils import *

from collator import FinetuneCollator
from arguments import ModelArguments, DataArguments, TrainingArguments
from model.finetune_reccom import FinetuneRecComModel


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if local_rank == 0:
        print(vars(model_args))
        print(vars(data_args))
        print(vars(training_args))

    set_seed(training_args.seed)
    ensure_dir(training_args.output_dir)
    torch.backends.cuda.enable_flash_sdp(True)

    train_data, valid_data, test_data = load_finetune_data(data_args, training_args)

    model = FinetuneRecComModel(
        args = model_args,
        dataset = train_data.dataset
    )
    pretrain_state_dict = load_pretrain(model_args)

    if local_rank == 0:
        print("Before load", model.reccom.llm_adapter.weight)
        print("Before load", model.llm_emb_mlp[0].weight)

    missing_keys, unexpected_keys = model.load_state_dict(pretrain_state_dict, strict=False)

    if local_rank == 0:
        print("After load", model.reccom.llm_adapter.weight)
        print("After load", model.llm_emb_mlp[0].weight)
        print("missing_keys: ", missing_keys)
        # print("unexpected_keys: ", unexpected_keys)

    trainer = FtTrainer(training_args, model, train_data, valid_data, test_data)

    best_score, test_results = trainer.train()

    if trainer.accelerator.is_main_process:
        print("Best Validation Score:", best_score)
        print("Test Results:", test_results)



if __name__ == "__main__":

    main()

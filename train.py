import argparse
import os

import sys

import torch
import transformers
from peft.utils import ModulesToSaveWrapper
from torch.utils.data import DataLoader

from transformers import HfArgumentParser, AutoTokenizer


from utils import *
from collator import Collator
from arguments import ModelArguments, DataArguments, TrainingArguments
from model.model import RecComModel
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from functools import partial
import torch.utils.checkpoint


class CastOutputToFloat(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs).float()
def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    ensure_dir(training_args.output_dir)

    # device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if local_rank == 0:
        print(vars(model_args))
        print(vars(data_args))
        print(vars(training_args))

    if ddp:
        # device_map = {"": local_rank}
        device = torch.device("cuda", local_rank)
        training_args.ddp_find_unused_parameters = False
    else:
        device = torch.device("cuda")


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length = data_args.model_max_length,
        trust_remote_code = True,
    )

    train_data, valid_data, n_photos = load_datasets(data_args, tokenizer)
    if local_rank==0:
        print("data number:", len(train_data))

    collator = Collator(data_args, tokenizer)

    torch_dtype = torch.bfloat16
    model = RecComModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        n_photos=n_photos,
        args=model_args,
        empty_init=False,
        device_map=None,
    )
    model = model.to(torch_dtype)
    model = model.to(device)

    if model_args.lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            target_modules=model_args.lora_target_modules.split(","),
            modules_to_save=model_args.lora_modules_to_save.split(","),
            lora_alpha=model_args.lora_alpha,
            bias="none",
            lora_dropout=model_args.lora_dropout,
        )

        model = get_peft_model(model, peft_config)

        if training_args.resume_from_checkpoint:
            checkpoint_name = os.path.join(
                training_args.resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            training_args.resume_from_checkpoint = False  # So the trainer won't try loading its state
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                if local_rank == 0:
                    print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                model = set_peft_model_state_dict(model, adapters_weights)
            else:
                if local_rank == 0:
                    print(f"Checkpoint {checkpoint_name} not found")


        if local_rank == 0:
            model.print_trainable_parameters()




    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model.config.use_cache = False
    torch.backends.cuda.enable_flash_sdp(True)
    model.lm_head = CastOutputToFloat(model.transformer.output_layer)

    if training_args.gradient_checkpointing:
        torch.utils.checkpoint.checkpoint = partial(torch.utils.checkpoint.checkpoint,
                                                    use_reentrant=False)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()


    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=valid_data,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()

    trainer.save_state()
    trainer.save_model()




if __name__ == "__main__":

    main()

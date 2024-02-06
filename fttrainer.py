import logging

import numpy as np
import torch
from time import time

from accelerate import Accelerator, PartialState
from accelerate.logging import get_logger
from torch import optim
from tqdm.auto import tqdm
import math
import torch.nn as nn

import os
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import *
from metrics import metrics_to_function

from torch import autograd

class FtTrainer(object):

    def __init__(self, args, model, train_data, valid_data, test_data=None):
        self.args = args
        self.model = model
        self.lr = args.learning_rate
        self.optimizer_name = args.optim
        self.lr_scheduler_type = args.lr_scheduler_type
        self.weight_decay = args.weight_decay
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.epochs = int(args.num_train_epochs)
        self.eval_steps = min(int(args.eval_steps), self.epochs)
        self.all_metrics = args.metrics.split(",")
        self.valid_metric = args.valid_metric
        self.max_topk = 0
        self.all_metric_name = []
        for m in self.all_metrics:
            m_name, top_k = m.split("@")
            self.max_topk = max(self.max_topk, int(top_k))
            if m_name.lower() not in self.all_metric_name:
                self.all_metric_name.append(m_name.lower())

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.max_steps = self.get_train_steps()
        self.warmup_steps = self.args.get_warmup_steps(self.max_steps)
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._get_scheduler()


        self.accelerator = Accelerator(
            gradient_accumulation_steps = self.gradient_accumulation_steps,
            mixed_precision="no"
        )


        self.model, self.optimizer, self.lr_scheduler, self.train_data, self.valid_data, self.test_data = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_data, self.valid_data, self.test_data)

        # assert dist.is_initialized(), "Distributed training has not been properly initialized."

        self.state = PartialState()
        self.world_size = self.state.num_processes
        self.device = self.state.device

        self.ckpt_dir = args.output_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_score = 0
        self.best_ckpt = "best_model.pth"
        self.loss_func = nn.CrossEntropyLoss()


    def _build_optimizer(self):

        params = self.model.parameters()
        optimizer_name = self.optimizer_name
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if optimizer_name.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'adamw_torch':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            print(
                "Received unrecognized optimizer, set default AdamW optimizer"
            )
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                                num_warmup_steps=self.warmup_steps,
                                                                num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                                num_warmup_steps=self.warmup_steps)


        return lr_scheduler

    def get_train_steps(self):

        len_dataloader = len(self.train_data)
        num_update_steps_per_epoch = len_dataloader // self.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(self.epochs * num_update_steps_per_epoch)

        return max_steps

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _train_epoch(self, epoch_idx):

        self.model.train()

        total_num = 0
        total_loss = 0
        iter_data = tqdm(
            self.train_data,
            total=len(self.train_data),
            ncols=100,
            desc=set_color(f"Train {epoch_idx}", "pink"),
            disable=not self.accelerator.is_main_process,
        )

        for batch_idx, data in enumerate(iter_data):

            with self.accelerator.accumulate(self.model):

                labels = data.pop("labels")
                total_num += 1
                self.optimizer.zero_grad()
                logits, id_text_loss = self.model(**data)
                loss = self.loss_func(logits, labels) + id_text_loss
                self._check_nan(loss)
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.lr_scheduler.step()

                total_loss += loss.item()
                iter_data.set_postfix(loss=loss.item(), lr=self.lr_scheduler.get_last_lr())

        self.accelerator.wait_for_everyone()

        return total_loss/total_num

    @torch.no_grad()
    def test(self, eval_data=None, load_best_model = False, model_file=None):

        if eval_data is None:
            eval_data = self.test_data

        if load_best_model:
            checkpoint_file = model_file or os.path.join(self.ckpt_dir, self.best_ckpt)
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            if dist.is_initialized():
                missing_keys, unexpected_keys = self.model.module.load_state_dict(checkpoint["state_dict"],
                                                                                  strict=False)
            else:
                missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["state_dict"], strict=False)

            if self.accelerator.is_main_process:
                message_output = "Loading model parameters from {}".format(
                    checkpoint_file
                )
                print(message_output)
                print("missing_keys: ", missing_keys)
                print("unexpected_keys: ", unexpected_keys)

        self.model.eval()

        iter_data = tqdm(
            eval_data,
            total=len(eval_data),
            ncols=100,
            desc=set_color(f"Evaluate  ", "pink"),
            disable=not self.accelerator.is_main_process,
        )

        total = 0
        metrics = {m:0 for m in self.all_metrics}
        for batch_idx, data in enumerate(iter_data):

            labels = data.pop("labels")
            total += len(labels)

            scores, _ = self.model(**data)

            _metrics = self.evaluate(scores, labels)
            for m, v in _metrics.items():
                metrics[m] += v

        for m in metrics:
            metrics[m] = metrics[m] / total

        return metrics

    def evaluate(self, scores, labels):

        metrics = {m:0 for m in self.all_metrics}

        _, topk_idx = torch.topk(
            scores, self.max_topk, dim=-1
        )  # B x k
        topk_idx = topk_idx.detach().cpu()
        labels = labels.detach().cpu()

        top_k_labels = torch.gather(labels, dim=1, index=topk_idx).numpy()
        pos_nums = labels.sum(dim=1).numpy()

        topk_metrics = {}
        for m_name in self.all_metric_name:
            value = metrics_to_function[m_name](top_k_labels, pos_nums)
            topk_metrics[m_name] = value.sum(axis=0)

        for m in self.all_metrics:
            m_name, top_k = m.split("@")
            m_name = m_name.lower()
            top_k = int(top_k)
            value = topk_metrics[m_name]
            metrics[m] = value[top_k - 1]

        return metrics

    def _save_checkpoint(self, epoch, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, self.best_ckpt)
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_score": self.best_score,
            "state_dict": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        print(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss):
        train_loss_output = (
                                    set_color("epoch %d training", "green")
                                    + " ["
                                    + set_color("time", "blue")
                                    + ": %.2fs, "
                            ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        return train_loss_output + "]"

    def train(self,):

        cur_eval_step = 0
        stop = False
        for epoch_idx in range(self.epochs):

            self.accelerator.wait_for_everyone()
            # train
            training_start_time = time()
            train_loss = self._train_epoch(epoch_idx)
            training_end_time = time()
            if dist.is_initialized():
                train_loss = torch.tensor(train_loss).to(self.device)
                train_loss = self.accelerator.gather(train_loss).mean().item()

            if self.accelerator.is_main_process:
                train_loss_output = self._generate_train_loss_output(
                    epoch_idx, training_start_time, training_end_time, train_loss
                )
                print(train_loss_output)

            if (epoch_idx + 1) % self.eval_steps == 0:
                metrics = self.test(eval_data=self.valid_data)
                if dist.is_initialized():
                    metrics_list = [None for _ in range(self.world_size)]
                    dist.all_gather_object(obj=metrics, object_list=metrics_list)
                    total_metrics = {m:0 for m in self.all_metrics}
                    for m in self.all_metrics:
                        for metric_dict in metrics_list:
                            total_metrics[m] += metric_dict[m]
                        total_metrics[m] = total_metrics[m] / self.world_size
                else:
                    total_metrics = metrics


                if total_metrics[self.valid_metric] > self.best_score:
                    self.best_score = total_metrics[self.valid_metric]
                    cur_eval_step = 0
                    if self.accelerator.is_main_process:
                        self._save_checkpoint(epoch_idx)
                else:
                    cur_eval_step += 1

                if cur_eval_step >= 5:
                    stop = True

                if self.accelerator.is_main_process:
                    print(str(total_metrics))

            self.accelerator.wait_for_everyone()

            if stop:
                break

        test_results=None
        if self.test_data is not None:
            metrics = self.test(eval_data=self.test_data, load_best_model=True)
            if dist.is_initialized():
                metrics_list = [None for _ in range(self.world_size)]
                dist.all_gather_object(obj=metrics, object_list=metrics_list)
                test_results = {m: 0 for m in self.all_metrics}
                for m in self.all_metrics:
                    for metric_dict in metrics_list:
                        test_results[m] += metric_dict[m]
                    test_results[m] = test_results[m] / self.world_size
            else:
                test_results = metrics


        return self.best_score, test_results








#!/usr/bin/env python
# coding=utf-8

import argparse
import math
import os
import random
import csv
import statistics
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

import datasets
from datasets import load_dataset
import evaluate

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    PretrainedConfig
)
from experimental.diffgrad import diffgrad
from experimental.exp import Experimental


###############################################################################
# Stub implementations for custom optimizers. Replace with your own if needed.
###############################################################################

###############################################################################

def get_optimizer(optimizer_name, model_params, lr):
    """
    Return a PyTorch optimizer based on its name.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adagrad':
        return optim.Adagrad(model_params, lr=lr)
    elif optimizer_name == 'adadelta':
        return optim.Adadelta(model_params, lr=lr)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_params, lr=lr)
    elif optimizer_name == 'amsgrad':
        return optim.Adam(model_params, lr=lr, amsgrad=True)
    elif optimizer_name == 'adam':
        return optim.Adam(model_params, lr=lr)
    elif optimizer_name == 'experimental':
        return Experimental(model_params, lr=lr)
    elif optimizer_name == 'diffgrad':
        return diffgrad(model_params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

# GLUE Task -> (sentence1 key, sentence2 key)
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a GLUE task with multiple seeds.")
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The name of the GLUE task to train on.",
        choices=list(task_to_keys.keys())
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adagrad", "adadelta", "rmsprop", "amsgrad", "adam", "experimental", "diffgrad"],
        help="Optimizer to use."
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (per device).")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length.")
    parser.add_argument("--pad_to_max_length", action="store_true", help="Pad all samples to max_length.")
    parser.add_argument("--ignore_mismatched_sizes", action="store_true",
                        help="Whether or not to load a pretrained model whose head dimensions are different.")

    # Optionally, you can add arguments if you want a custom seeds list, etc.
    # We'll just fix 5 seeds in the script.
    return parser.parse_args()

def run_single_training(task_name, model_name_or_path, optimizer_name, lr, epochs, batch_size,
                        max_length, pad_to_max_length, seed, ignore_mismatched_sizes=False):
    """
    Runs a single training+evaluation with the specified seed and returns a dictionary of the final metrics.
    """
    # Set seed
    set_seed(seed)
    
    accelerator = Accelerator()
    # Load dataset
    raw_datasets = load_dataset("nyu-mll/glue", task_name)

    # Distinguish regression tasks
    is_regression = (task_name == "stsb")

    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=ignore_mismatched_sizes
    )


    sentence1_key, sentence2_key = task_to_keys[task_name]

    # Possibly adjust model label mapping if necessary
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and task_name is not None
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            print(f"[Seed {seed}] Model has label correspondence: {label_name_to_id}. Using it.")
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            print(f"[Seed {seed}] Warning: model labels don't match dataset labels. Ignoring model labels.")
    elif task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)
        if "label" in examples:
            if label_to_id is not None:
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if task_name == "mnli" else "validation"]

    if pad_to_max_length:
        data_collator = default_data_collator
    else:
        pad_to_multiple_of = 8 if accelerator.mixed_precision != "no" else None
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = get_optimizer(optimizer_name, optimizer_grouped_parameters, lr=lr)

    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    total_steps = epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=SchedulerType.LINEAR,  # Hard-coded linear here, or you can also parameterize it
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    metric = evaluate.load("glue", task_name)

    print(f"[Seed {seed}] Starting training on {task_name} with {optimizer_name} ...")
    progress_bar = tqdm(range(epochs), disable=not accelerator.is_local_main_process)
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Evaluate
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            if is_regression:
                predictions = outputs.logits.squeeze()
            else:
                predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            metric.add_batch(predictions=predictions, references=references)
        eval_result = metric.compute()
        progress_bar.set_description(f"Epoch {epoch} - eval: {eval_result}")
        progress_bar.update(1)
        metric = evaluate.load("glue", task_name)  # re-init for next epoch so we don't accumulate.

    # Final evaluation result (after last epoch)
    # We'll do one last pass to get the final result for this run:
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        if is_regression:
            predictions = outputs.logits.squeeze()
        else:
            predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        metric.add_batch(predictions=predictions, references=references)
    final_result = metric.compute()
    print(f"[Seed {seed}] Final result: {final_result}")
    return final_result

def main():
    args = parse_args()

    # Prepare seeds; you can customize as desired
    seeds = [41,42,43,44,45]
    results_for_seeds = []
    
    for sd in seeds:
        single_result = run_single_training(
            task_name=args.task_name,
            model_name_or_path=args.model_name_or_path,
            optimizer_name=args.optimizer,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pad_to_max_length=args.pad_to_max_length,
            seed=sd,
            ignore_mismatched_sizes=args.ignore_mismatched_sizes
        )
        results_for_seeds.append(single_result)

    # Different GLUE tasks may return a dictionary with different metric keys
    # We'll take the median across seeds for each metric key that appears.
    # e.g. for MRPC, we might have {'accuracy': ..., 'f1': ...}
    # for CoLA, we might have {'matthews_correlation': ...}
    all_keys = set()
    for r in results_for_seeds:
        all_keys.update(r.keys())

    median_results = {}
    for k in all_keys:
        vals = [r[k] for r in results_for_seeds if k in r]
        median_results[k] = statistics.median(vals) if len(vals) > 0 else None

    # Now let's store them to ./results/glue.csv
    os.makedirs("./results", exist_ok=True)
    output_csv = "./results/glue.csv"


    # We'll have columns: model_name_or_path, task_name, optimizer, epochs, batch_size, lr, plus whatever metrics appear
    ALL_GLUE_METRICS = ["accuracy", "f1", "matthews_correlation", "pearson", "spearmanr"]
    fieldnames = ["model_name_or_path", "task_name", "optimizer", "epochs", "batch_size", "lr"] + ALL_GLUE_METRICS

    # If file doesn't exist, create it with header
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        row_dict = {
            "model_name_or_path": args.model_name_or_path,
            "task_name": args.task_name,
            "optimizer": args.optimizer,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr
        }
        # For every expected metric, fill in the value if present; else leave blank.
        for metric in ALL_GLUE_METRICS:
            row_dict[metric] = median_results.get(metric, "")
        writer.writerow(row_dict)

    print(f"Median results across seeds stored in {output_csv}.")
    print("Done.")



if __name__ == "__main__":
    main()


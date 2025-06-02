#!/usr/bin/env python
# coding=utf-8

import math
import os
import random
from pathlib import Path
import statistics # Keep for potential future use, not strictly needed for this version

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

# Hardcoded configuration for isolating the issue
CONFIG = {
    "task_name": "cola",
    "model_name_or_path": "albert-base-v2", # Matches table format which Hugging Face hub usually resolves
    "optimizer_name": "rmsprop",
    "epochs": 3,
    "lr": 3e-5,
    "batch_size": 32, # User table showed 32, original script default was 8
    "max_length": 128,
    "pad_to_max_length": False,
    "seed": 42, # Using a single seed for deterministic diagnosis
    "ignore_mismatched_sizes": False
}

# GLUE Task -> (sentence1 key, sentence2 key) - Simplified
task_to_keys = {
    "cola": ("sentence", None),
}

def get_optimizer(optimizer_name, model_params, lr):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'rmsprop':
        # Note: The learning rate 3e-5 is very low for typical RMSprop (default is 1e-2).
        # This could be a key factor.
        print(f"Instantiating RMSprop with lr={lr}")
        return optim.RMSprop(model_params, lr=lr, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)
    elif optimizer_name == 'adam': # Adding Adam for comparison if needed later, but not used by default
        return optim.Adam(model_params, lr=lr)
    else:
        raise ValueError(f"Unknown or unsupported optimizer for this diagnostic script: {optimizer_name}")

def run_diagnostic_training(config):
    """
    Runs a single training+evaluation with the specified config and performs diagnostics.
    """
    set_seed(config["seed"])
    accelerator = Accelerator()

    print(f"--- Initializing Diagnostic Run for: {config['model_name_or_path']} on {config['task_name']} with {config['optimizer_name']} ---")

    raw_datasets = load_dataset("nyu-mll/glue", config["task_name"])
    is_regression = (config["task_name"] == "stsb") # CoLA is not regression

    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1 # Not relevant for CoLA

    model_config = AutoConfig.from_pretrained(
        config["model_name_or_path"],
        num_labels=num_labels,
        finetuning_task=config["task_name"],
        token=False
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], use_fast=True,token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model_config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name_or_path"],
        config=model_config,
        ignore_mismatched_sizes=config["ignore_mismatched_sizes"],
        token=False
    )

    sentence1_key, sentence2_key = task_to_keys[config["task_name"]]
    label_to_id = None # Simplified label handling for CoLA, assuming standard 0/1
    if not is_regression:
         model.config.label2id = {l: i for i, l in enumerate(label_list)}
         model.config.id2label = {id: label for label, id in model.config.label2id.items()}


    padding = "max_length" if config["pad_to_max_length"] else False

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=config["max_length"], truncation=True)
        if "label" in examples:
            result["labels"] = examples["label"] # Assuming labels are already 0/1 for CoLA
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"] # CoLA uses "validation"

    data_collator = default_data_collator if config["pad_to_max_length"] else DataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config["batch_size"])
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=config["batch_size"])

    # Prepare optimizer (no weight decay for any params as per original script's effective behavior for RMSprop)
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad], "weight_decay": 0.0}
    ]
    optimizer = get_optimizer(config["optimizer_name"], optimizer_grouped_parameters, lr=config["lr"])

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    total_steps = config["epochs"] * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=SchedulerType.LINEAR,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    metric = evaluate.load("glue", config["task_name"])

    # Diagnostic collectors
    train_losses = []
    gradient_norms = []
    parameter_update_norms = []

    print(f"\n--- Starting Training (Seed {config['seed']}) ---")
    for epoch in range(config["epochs"]):
        model.train()
        epoch_total_loss = 0
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        progress_bar = tqdm(range(len(train_dataloader)), disable=not accelerator.is_local_main_process, desc="Training")
        for step, batch in enumerate(train_dataloader):
            # Store parameters before optimizer step
            params_before_step = [p.clone().detach() for p in model.parameters() if p.requires_grad]

            outputs = model(**batch)
            loss = outputs.loss
            epoch_total_loss += loss.item()
            train_losses.append(loss.item())

            accelerator.backward(loss)

            # --- Gradient Norm Diagnostic ---
            current_grad_norms = []
            for p in model.parameters():
                if p.grad is not None:
                    current_grad_norms.append(torch.norm(p.grad.detach(), p=2).item())
            if current_grad_norms:
                avg_grad_norm = sum(current_grad_norms) / len(current_grad_norms)
                gradient_norms.append(avg_grad_norm)
                if step % 50 == 0 : # Log every 50 steps
                    print(f"  Step {step}: Loss = {loss.item():.4f}, Avg Grad Norm = {avg_grad_norm:.4e}")
            else: # Should not happen if loss is computed and model has trainable params
                 if step % 50 == 0 :
                    print(f"  Step {step}: Loss = {loss.item():.4f}, No gradients found.")


            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # --- Parameter Update Norm Diagnostic ---
            current_update_norms = []
            for i, p_after in enumerate(model.parameters()):
                if p_after.requires_grad:
                    update_norm = torch.norm(p_after.detach() - params_before_step[i], p=2).item()
                    current_update_norms.append(update_norm)
            if current_update_norms:
                avg_update_norm = sum(current_update_norms) / len(current_update_norms)
                parameter_update_norms.append(avg_update_norm)
                if step % 50 == 0 and avg_grad_norm is not None: # Log every 50 steps
                     print(f"  Step {step}: Avg Param Update Norm = {avg_update_norm:.4e}")


            progress_bar.update(1)
            progress_bar.set_postfix({"Loss": loss.item()})
        
        avg_epoch_loss = epoch_total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_epoch_loss:.4f}")


        # --- Evaluation after each epoch ---
        model.eval()
        all_predictions = []
        all_references = []
        print("Evaluating...")
        for eval_step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Evaluation"):
            with torch.no_grad():
                outputs = model(**batch)
            
            predictions = outputs.logits.argmax(dim=-1)
            gathered_preds, gathered_labels = accelerator.gather_for_metrics((predictions, batch["labels"]))
            all_predictions.extend(gathered_preds.cpu().tolist())
            all_references.extend(gathered_labels.cpu().tolist())
            metric.add_batch(predictions=gathered_preds, references=gathered_labels)
        
        eval_result = metric.compute()
        print(f"Epoch {epoch+1} Evaluation Result: {eval_result}")
        # Re-init metric for next epoch or final eval (important!)
        metric = evaluate.load("glue", config["task_name"])


    # --- Final Evaluation and Diagnostics ---
    print("\n--- Final Evaluation and Diagnostic Analysis ---")
    model.eval()
    final_all_predictions = []
    final_all_references = []
    for step, batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc="Final Evaluation"):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        metric.add_batch(predictions=predictions, references=references)
        final_all_predictions.extend(predictions.cpu().tolist())
        final_all_references.extend(references.cpu().tolist())

    final_result = metric.compute()
    print(f"Final Evaluation Result (Seed {config['seed']}): {final_result}")

    # --- Automated Diagnostic Statement ---
    mcc = final_result.get("matthews_correlation", None)
    diagnostic_message = f"\n--- Diagnostic Summary for {config['optimizer_name']} with LR={config['lr']} ---"

    # 1. Check training loss behavior
    if not train_losses:
        diagnostic_message += "\n- Training did not record any losses."
    elif any(math.isnan(l) or math.isinf(l) for l in train_losses):
        diagnostic_message += "\n- Training Loss became NaN or Inf. Training diverged."
    elif train_losses:
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        avg_loss = sum(train_losses) / len(train_losses)
        diagnostic_message += f"\n- Training Loss: Initial={initial_loss:.4f}, Final={final_loss:.4f}, Avg={avg_loss:.4f}."
        if final_loss > initial_loss * 0.9: # Heuristic for "not decreasing much"
             diagnostic_message += " Loss did not decrease substantially."


    # 2. Check gradient norms
    if not gradient_norms:
        diagnostic_message += "\n- No gradient norms were recorded."
    else:
        avg_grad_norm = sum(gradient_norms) / len(gradient_norms)
        max_grad_norm = max(gradient_norms)
        min_grad_norm = min(gradient_norms)
        diagnostic_message += f"\n- Gradient Norms (L2): Avg={avg_grad_norm:.2e}, Max={max_grad_norm:.2e}, Min={min_grad_norm:.2e}."
        if avg_grad_norm < 1e-6 and max_grad_norm < 1e-5 :
            diagnostic_message += " Gradients are very small (potential vanishing gradients)."
    
    # 3. Check parameter update norms
    if not parameter_update_norms:
        diagnostic_message += "\n- No parameter update norms were recorded."
    else:
        avg_update_norm = sum(parameter_update_norms) / len(parameter_update_norms)
        max_update_norm = max(parameter_update_norms)
        min_update_norm = min(parameter_update_norms)
        diagnostic_message += f"\n- Parameter Update Norms (L2): Avg={avg_update_norm:.2e}, Max={max_update_norm:.2e}, Min={min_update_norm:.2e}."
        if avg_update_norm < 1e-7 and max_update_norm < 1e-6:
            diagnostic_message += " Parameter updates are extremely small, suggesting ineffective learning steps."

    # 4. Analyze predictions
    if final_all_predictions:
        pred_counts = {k: final_all_predictions.count(k) for k in sorted(list(set(final_all_predictions)))}
        diagnostic_message += f"\n- Prediction Distribution on Eval Set: {pred_counts} (Total: {len(final_all_predictions)})."
        if len(pred_counts) == 1:
            predicted_class = list(pred_counts.keys())[0]
            diagnostic_message += f" Model predicted only class '{predicted_class}' for all evaluation samples."
            if mcc == 0.0:
                diagnostic_message += " This is the direct cause of the Matthews Correlation Coefficient being 0."
        elif mcc == 0.0:
             diagnostic_message += " Matthews Correlation Coefficient is 0 despite varied predictions. This might indicate issues like predictions being uncorrelated with true labels, or a specific type of failure mode for MCC not captured by single-class prediction."
    else:
        diagnostic_message += "\n- No predictions were recorded from the final evaluation."


    # 5. Concluding statement based on MCC and other diagnostics
    diagnostic_message += "\n\n--- Final Conclusion ---"
    if mcc == 0.0:
        diagnostic_message += f"\nThe Matthews Correlation of 0.0 with {config['optimizer_name']} (lr={config['lr']}) is likely due to:"
        if len(set(final_all_predictions)) == 1:
            diagnostic_message += "\n  1. The model collapsing to predict a single class for all inputs."
            if parameter_update_norms and sum(parameter_update_norms)/len(parameter_update_norms) < 1e-6 :
                 diagnostic_message += "\n  2. This collapse appears to be a result of extremely small parameter updates during training."
                 diagnostic_message += f" The learning rate of {config['lr']} is substantially lower than the default for RMSprop (typically 1e-2) and may be too small for effective parameter space exploration, causing the optimizer to get stuck or make negligible progress."
            elif gradient_norms and sum(gradient_norms)/len(gradient_norms) < 1e-5:
                 diagnostic_message += "\n  2. This collapse might be linked to very small gradients (vanishing gradients), preventing meaningful updates."
            else:
                 diagnostic_message += "\n  2. The learning process failed to generalize, leading to a trivial solution. This could be due to the specific learning rate and optimizer combination being unsuitable for this model/task, failing to effectively navigate the loss landscape."

        else: # MCC is 0 but predictions are not all one class (less common, but possible)
            diagnostic_message += "\n  1. A complex failure mode where predictions, though varied, show no correlation with true labels according to the MCC formula. This could happen if true positives and true negatives are roughly equal to false positives and false negatives in a way that nullifies the numerator of MCC."
        diagnostic_message += f"\nIt is strongly recommended to test {config['optimizer_name']} with a higher learning rate (e.g., in the range of 1e-4 to 1e-2) or to ensure that the chosen learning rate is appropriate based on hyperparameter tuning for this specific optimizer."

    elif mcc is not None:
        diagnostic_message += f"\nThe model achieved a Matthews Correlation of {mcc:.4f}. The diagnostics above provide context on the training dynamics."
    else:
        diagnostic_message += "\nMatthews Correlation was not computed or reported. Check evaluation metrics."

    print(diagnostic_message)
    print("\n--- End of Diagnostic Run ---")


def main():
    run_diagnostic_training(CONFIG)
    print("Diagnostic script finished.")

if __name__ == "__main__":
    main()

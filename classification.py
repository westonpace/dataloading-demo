#!/usr/bin/env python
# coding=utf-8
# Fine-tuning a model on a text classification task
# Based on: https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb

import argparse
import numpy as np
import datasets
import random
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# GLUE Tasks
GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

# Task to keys mapping
TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model on a GLUE task")
    parser.add_argument(
        "--task",
        type=str,
        default="cola",
        choices=GLUE_TASKS,
        help="The GLUE task to train on",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="distilbert-base-uncased",
        help="Model checkpoint to use for fine-tuning",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for model checkpoints (default: {model_name}-finetuned-{task})",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the Hugging Face Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="Model ID for pushing to Hub (default: output_dir name)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples to use (for debugging)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Maximum number of evaluation samples to use (for debugging)",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    task = args.task
    model_checkpoint = args.model_checkpoint
    batch_size = args.batch_size

    print(f"Training on task: {task}")
    print(f"Using model: {model_checkpoint}")
    print(f"Batch size: {batch_size}")

    # Load dataset and metric
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)

    print(f"Dataset loaded: {dataset}")
    print(f"Metric: {metric}")

    # Limit dataset size if specified
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))

    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    if args.max_eval_samples is not None and validation_key in dataset:
        dataset[validation_key] = dataset[validation_key].select(range(args.max_eval_samples))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    # Get column names for the task
    sentence1_key, sentence2_key = TASK_TO_KEYS[task]

    # Show example
    if sentence2_key is None:
        print(f"Sentence: {dataset['train'][0][sentence1_key]}")
    else:
        print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
        print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

    # Preprocessing function
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    # Encode dataset
    print("Encoding dataset...")
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # Determine number of labels
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

    # Load model
    print(f"Loading model with {num_labels} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    # Determine metric name for best model selection
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

    # Set output directory
    model_name = model_checkpoint.split("/")[-1]
    output_dir = args.output_dir if args.output_dir else f"{model_name}-finetuned-{task}"

    # Training arguments
    training_args = TrainingArguments(
        output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        seed=args.seed,
    )

    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    # Create trainer
    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Evaluate
    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Push to hub if requested
    if args.push_to_hub:
        print("Pushing model to Hub...")
        trainer.push_to_hub()

    print("Training complete!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3 

import sys
import argparse
import datasets
import evaluate
import logging

import torch
import transformers
import numpy as np

from typing import Dict
from datasets import Dataset
from seqeval.metrics import precision_score, recall_score, f1_score
from seqeval.scheme import IOB2
from transformers import (AutoTokenizer, AutoModelForTokenClassification, AutoConfig,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)

from masking import DatasetMasking
from dataset import DatasetLoader, Data


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset_name_or_path")
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--text_column_name")
    parser.add_argument("--ner_tags_column_name")
    parser.add_argument("--output_dir")
    parser.add_argument("--path_to_results")

    parser.add_argument("--windows_size", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="no_mask")
    parser.add_argument("--p_mask", type=float, default=0.3)

    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class Configuration:
    def __init__(self, model_name_or_path, num_labels, label_to_id, id_to_label):
        self.config = AutoConfig.from_pretrained(model_name_or_path,
                                                 num_labels=num_labels,
                                                 label2id=label_to_id,
                                                 id2label=id_to_label)

    def __call__(self):
        return self.config


class Tokenization:
    def __init__(self, model_name_or_path, max_length: int, label2id: Dict, text_column_name: str,
                 ner_tag_column_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, add_prefix_space=True)
        self.max_length = max_length
        self.label2id = label2id
        self.text_column_name = text_column_name
        self.ner_tag_column_name = ner_tag_column_name

    def __call__(self, features: Dataset):
        tokenized_features = features.map(self.tokenize_and_align_labels,
                                          batched=True,
                                          remove_columns=[self.text_column_name,
                                                          self.ner_tag_column_name]
                                          )
        return tokenized_features

    def tokenize_and_align_labels(self, examples):
        tokenized = dict(input_ids=[],
                         attention_mask=[],
                         ner_tags=[])
        tokenized_inputs = self.tokenizer(examples[self.text_column_name],
                                          max_length=self.max_length,
                                          truncation=True,
                                          is_split_into_words=True
                                          )
        labels = []
        for i, label in enumerate(examples[self.ner_tag_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    if type(label[word_idx]) is str:
                        label_ids.append(self.align_labels_to_ids(label[word_idx], label2id=self.label2id))
                    else:
                        label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            tokenized_inputs["labels"] = labels
        return tokenized_inputs

    @staticmethod
    def align_labels_to_ids(label, label2id):
        return label2id[label]


class ModelForTokenClassification:
    def __init__(self, model_name_or_path, config):
        self.model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, config=config)


class AutoTokenClassifier:
    def __init__(self, trainer: transformers.Trainer, tokenized_dataset: datasets.Dataset, id_to_label_map: Dict):
        self.trainer = trainer
        self.tokenized_dataset = tokenized_dataset
        self.id_to_label_map = id_to_label_map
        self.predictions, self.gold = self._predict()

    def predict(self):
        # predictions, gold = self._predict()
        result_metrics = self.compute_metrics(self.predictions, self.gold)
        print(result_metrics)
        return result_metrics

    def _predict(self):
        predictions, gold, _ = self.trainer.predict(self.tokenized_dataset)
        predictions = np.argmax(predictions, axis=2)
        return predictions, gold

    def compute_metrics(self, predictions: torch.Tensor, gold: torch.Tensor):
        evaluations = dict(precision=None,
                           recall=None,
                           f1=None
                           )
        label_list = list(self.id_to_label_map.values())
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, gold)
        ]

        true_labels = [
            [label_list[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, gold)
        ]

        # Metrics
        evaluations["precision"] = precision_score(true_labels, true_predictions, mode="strict", scheme=IOB2)
        evaluations["recall"] = recall_score(true_labels, true_predictions, mode="strict", scheme=IOB2)
        evaluations["f1"] = f1_score(true_labels, true_predictions, mode="strict", scheme=IOB2)
        return evaluations


def main():
    args = arguments()

    # Prepare HF dataset
    dataset_loader = DatasetLoader(args.hf_dataset_name_or_path)
    # dataset_loader = DatasetLoader("conll2003")
    dataset, label2id, id2label = dataset_loader()
    label_list = list(label2id.keys())

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    num_labels = len(id2label)

    metrics = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metrics.compute(predictions=true_predictions, references=true_labels)

        return {"precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"]
                }

    config = Configuration(args.model_name_or_path,
                           num_labels=num_labels,
                           label_to_id=label2id,
                           id_to_label=id2label
                           )
    config = config()

    # Prepare data
    tokenization = Tokenization(args.model_name_or_path,
                                max_length=args.max_length,
                                label2id=label2id,
                                text_column_name=args.text_column_name,
                                ner_tag_column_name=args.ner_tags_column_name
                                )

    tokenizer = tokenization.tokenizer
    masking_token = tokenizer.mask_token

    # Training
    if args.strategy == "no_mask":
        args.windows_size = 0
        args.p_mask = 0.0

    if args.strategy == "entity":
        args.windows_size = 0

    train_data_masking = DatasetMasking(train_dataset, id2label)
    train_dataset = train_data_masking.mask(strategy=args.strategy, 
                                            windows_size=args.windows_size,
                                            mask_token=masking_token, 
                                            p_mask=args.p_mask)
    tokenized_train_dataset = tokenization(train_dataset)

    if args.do_eval:
        # Validation
        tokenized_val_dataset = tokenization(val_dataset)

    if args.do_test:
        # Test Prediction
        tokenized_test_dataset = tokenization(test_dataset)

    # Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer,
                                                       return_tensors="pt"
                                                       )

    # Setup logging
    print("Setting up logger...")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename='tmp.log')
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers
    )

    # TODO: Revise this logging section and make the whole ner code run.
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    #log_level = args.get_process_log_level()
    #logger.setLevel(log_level)
    #datasets.utils.logging.set_verbosity(log_level)
    #transformers.utils.logging.set_verbosity(log_level)

    logger.info(f"Training/evaluation parameters {args}")

    # Training
    model_for_token_classification = ModelForTokenClassification(args.model_name_or_path, config=config)
    model = model_for_token_classification.model
    training_args = TrainingArguments(output_dir=args.output_dir,
                                      learning_rate=args.learning_rate,
                                      per_device_train_batch_size=args.batch_size,
                                      per_device_eval_batch_size=args.batch_size,
                                      num_train_epochs=args.epochs,
                                      weight_decay=0.001,
                                      seed=args.seed,
                                      evaluation_strategy="epoch",
                                      save_strategy="no",
                                      log_level="info",
                                      logging_strategy="epoch",
                                      push_to_hub=False  # Make sure nothing is pushed to hub
                                      )
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=tokenized_train_dataset,
                      eval_dataset=tokenized_val_dataset,
                      data_collator=data_collator,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics,
                      )

    train_results = trainer.train()
    train_metrics = train_results.metrics
    # print(train_metrics)
    trainer.save_model()

    if args.do_eval:
        eval_metrics = trainer.evaluate()
        print(eval_metrics)

    if args.do_test:
        output_predicitons = []
        classifier = AutoTokenClassifier(trainer, tokenized_test_dataset, id2label)
        test_results = classifier.predict()
        predictions = classifier.predictions
        gold = classifier.gold

        # Convert ids to labels
        normalised_predictions = [[label_list[p] for (p, l) in zip(pred, label) if l != -100]
                                  for pred, label in zip(predictions, gold)
                                  ]

        # Write predictions to tsv
      #  try:    
      #      with open(f"./results/{args.hf_dataset_name_or_path}_{args.model_name_or_path.split('/')[1]}_{args.seed}_{args.windows_size}_{args.strategy}_{args.p_mask}_{args.epochs}_{args.learning_rate}_{args.batch_size}.tsv",
       #             "w", encoding="utf-8") as out:
       #         test_data = Data(test_dataset, id2label)
       #         i = 0
       #         for sequence, tags in test_data.sequence_generator():
       #             assert len(sequence) == len(
       #                 normalised_predictions[i]), f"{i}\t{len(sequence)}\t{len(normalised_predictions[i])}"
       #             for s, t, p in zip(sequence, tags, normalised_predictions[i]):
       #                 out.write(f"{s}\t{t}\t{p}\n")
       #             out.write("\n")
       #             i += 1
       # except Exception as e:
       #     print(e)
            
        # Save predictions metrics        
        with open(args.path_to_results, "a", encoding="utf-8") as out_f:
            out_f.write(f"{args.hf_dataset_name_or_path}\t{args.model_name_or_path}\t{args.seed}\t{args.windows_size}\t{args.strategy}\t{args.p_mask}\t{args.epochs}\t{args.learning_rate}\t{args.batch_size}\t{test_results['precision']}\t{test_results['recall']}\t{test_results['f1']}\n")
        print(f"P: {test_results['precision']}\tR: {test_results['recall']}\tF1: {test_results['f1']}")


if __name__ == "__main__":
    main()

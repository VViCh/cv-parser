import os
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from seqeval.metrics import accuracy_score, f1_score

models_to_compare = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "albert-base-v2",
    # "microsoft/deberta-v3-base",
    "google/electra-base-discriminator",
    # "SpanBERT/spanbert-base-cased",
    "distilroberta-base"
]

OUTPUT_DIR = "./benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

label_list = ["O", "B-Skill", "I-Skill", "B-Experience", "I-Experience", "B-Degree", "I-Degree", "B-Designation", "I-Designation"]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

def load_dataset():
    dataset_dict = {
        # "tokens": [["I", "know", "Python", "and", "ML"], ["B.Sc", "in", "CS"]],
        # "ner_tags": [[0, 0, 1, 0, 1], [5, 6, 6]]
    }
    ds = Dataset.from_dict(dataset_dict)
    return DatasetDict({"train": ds, "test": ds})

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=128)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

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
    
    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions)
    }

def run_benchmark():
    dataset = load_dataset()
    results = {}
    print("Starting full benchmark with Early Stopping (Max 20 Epochs)...")

    for model_name in models_to_compare:
        print(f"\nEvaluating: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True if "roberta" in model_name else False)
        
        tokenized_datasets = dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer),
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=len(label_list), 
            id2label=id2label, 
            label2id=label2id
        )

        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/{model_name}-ner",
            evaluation_strategy="epoch",  
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=20,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",   
            greater_is_better=True,
            report_to="none"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
        )
        
        trainer.train()
        metrics = trainer.evaluate()
        
        results[model_name] = {
            "eval_f1": metrics.get("eval_f1", 0.0),
            "eval_accuracy": metrics.get("eval_accuracy", 0.0),
            "eval_loss": metrics.get("eval_loss", 0.0)
        }
        print(f"Results for {model_name}: {results[model_name]}")

    sorted_results = sorted(results.items(), key=lambda item: item[1]["eval_f1"], reverse=True)
    for name, r in sorted_results:
        print(f"{name:30}: F1: {r['eval_f1']:.4f} | Acc: {r['eval_accuracy']:.4f} | Loss: {r['eval_loss']:.4f}")

    winner_name, winner_metrics = sorted_results[0]
    print(f"\nWinner: {winner_name} (F1: {winner_metrics['eval_f1']:.4f})")

if __name__ == "__main__":
    run_benchmark()
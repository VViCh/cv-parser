import torch
import gc
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from model_ner_v2 import ResumeNERModel
from preprocess_training_v2 import ResumeDataset, LABEL_LIST
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import json
import time
import logging
import random
from seqeval.metrics import precision_score, recall_score, f1_score

log_file = "training_log.vfinal-final.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MODELS_TO_TRAIN = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "bert-base-cased",
    "roberta-base",
    "albert-base-v2",
    "allenai/scibert_scivocab_uncased",
    "microsoft/deberta-v3-base",
    "google/electra-base-discriminator",
    "distilroberta-base"
]

EPOCHS = 40
LR = 5e-5
MAX_LEN = 128
BATCH_SIZE = 8
PATIENCE = 4
DATA_PATH = "cv/dataset.json"

# Improvement thresholds
MIN_F1_DELTA = 0.001
MIN_LOSS_DELTA = 0.01
TIE_F1_EPS = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")

results = {}

logger.info("="*60)
logger.info("Calculating dataset splits...")
logger.info("="*60)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    total_samples = sum(1 for line in f if line.strip())

indices = list(range(total_samples))
random.seed(42)
random.shuffle(indices)

train_size = int(0.8 * total_samples)
train_indices = indices[:train_size]
val_indices = indices[train_size:]

logger.info(f"Total resumes: {total_samples}")
logger.info(f"Train resumes: {len(train_indices)}")
logger.info(f"Validation resumes: {len(val_indices)}")


def compute_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            loss = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += loss.item()

            pred_tags = model(input_ids, attention_mask=attention_mask)

            for pred, true_label, mask in zip(pred_tags, labels, attention_mask):
                pred_labels = []
                true_labels = []
                for p, l, m in zip(pred, true_label, mask):
                    if m == 1 and l != -100:
                        pred_labels.append(LABEL_LIST[p] if p < len(LABEL_LIST) else "O")
                        true_labels.append(LABEL_LIST[l] if l < len(LABEL_LIST) else "O")

                all_preds.append(pred_labels)
                all_labels.append(true_labels)

    try:
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
    except Exception as e:
        logger.warning(f"Metric calculation warning: {e}")
        precision, recall, f1 = 0.0, 0.0, 0.0

    avg_val_loss = total_val_loss / max(len(loader), 1)
    return precision, recall, f1, avg_val_loss


def has_improved(current_f1, best_f1, current_loss, best_loss):
    if current_f1 > best_f1 + MIN_F1_DELTA:
        return True, "f1"
    if current_f1 >= best_f1 - TIE_F1_EPS and current_loss < best_loss - MIN_LOSS_DELTA:
        return True, "val_loss"
    return False, ""


for model_idx, model_name in enumerate(MODELS_TO_TRAIN, 1):
    logger.info("\n" + "="*60)
    logger.info(f"Training Model {model_idx}/{len(MODELS_TO_TRAIN)}: {model_name}")
    logger.info("="*60)

    start_time = time.time()

    try:
        logger.info("Loading tokenizer...")
        needs_prefix = True if "roberta" in model_name.lower() else False
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=needs_prefix)

        full_dataset = ResumeDataset(DATA_PATH, tokenizer, max_len=MAX_LEN)
        train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(full_dataset, val_indices), batch_size=BATCH_SIZE, shuffle=False)

        logger.info("Initializing model architecture...")
        model = ResumeNERModel(model_name=model_name, num_labels=len(LABEL_LIST)).to(device)
        model = model.float()

        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.02)

        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        best_val_loss = float('inf')
        best_f1 = -1.0
        best_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        patience_counter = 0
        final_train_loss = 0.0

        logger.info(f"Starting training for {EPOCHS} epochs...")

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            for batch_idx, batch in enumerate(train_loader, 1):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                loss = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss.backward()

                optimizer.step()
                scheduler.step()

                running_loss += loss.item()
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}", end="\r")

            final_train_loss = running_loss / len(train_loader)

            precision, recall, f1, val_loss = compute_metrics(model, val_loader, device)

            logger.info(f"\nEpoch {epoch+1}/{EPOCHS} | Train Loss: {final_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            logger.info(f"Metrics - F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

            improved, reason = has_improved(f1, best_f1, val_loss, best_val_loss)
            if improved:
                best_f1 = max(best_f1, f1)
                best_val_loss = min(best_val_loss, val_loss)
                best_metrics = {"f1": f1, "precision": precision, "recall": recall}
                patience_counter = 0

                safe_name = model_name.replace('/', '_')
                model_path = f"resume_ner_model_{safe_name}.pth"

                torch.save(model.state_dict(), model_path)
                logger.info(f"Improved by {reason}. Saved FULL model weights to {model_path}")
            else:
                patience_counter += 1
                logger.info(f"No improvement ({patience_counter}/{PATIENCE})")

            if patience_counter >= PATIENCE:
                logger.info("Early stopping triggered!")
                break

        elapsed_time = time.time() - start_time

        results[model_name] = {
            "best_val_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "best_f1": best_metrics["f1"],
            "best_precision": best_metrics["precision"],
            "best_recall": best_metrics["recall"],
            "epochs_trained": epoch + 1,
            "time_seconds": elapsed_time,
            "model_path": f"resume_ner_model_final_{model_name.replace('/', '_')}.pth"
        }

        logger.info(f"Model completed in {elapsed_time:.1f}s")

        del model, optimizer, scheduler, tokenizer, train_loader, val_loader, full_dataset
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Failed to train {model_name}: {e}")
        results[model_name] = {"error": str(e)}

logger.info("\n" + "="*60)
logger.info("FINAL TRAINING SUMMARY")
logger.info("="*60)

for model_name, metrics in results.items():
    logger.info(f"\n{model_name}:")
    if "error" in metrics:
        logger.info(f"Error: {metrics['error']}")
    else:
        logger.info(f"Best Val Loss: {metrics['best_val_loss']:.4f}")
        logger.info(f"F1 Score:      {metrics['best_f1']:.4f}")
        logger.info(f"Precision:     {metrics['best_precision']:.4f}")
        logger.info(f"Recall:        {metrics['best_recall']:.4f}")
        logger.info(f"Epochs:        {metrics['epochs_trained']}")
        logger.info(f"Time taken:    {metrics['time_seconds']:.1f}s")

with open("training_results.vfinal-final.json", 'w') as f:
    json.dump(results, f, indent=4)

logger.info("\nDetailed results saved to training_results.vfinal-final.json")
logger.info(f"Full log saved to {log_file}")

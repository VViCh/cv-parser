import json

import torch
from torch.utils.data import Dataset


LABEL_LIST = [
    "O",
    "B-Skill",
    "I-Skill",
    "B-Experience",
    "I-Experience",
    "B-Degree",
    "I-Degree",
    "B-Designation",
    "I-Designation",
]

TARGET_LABELS = {
    "Skills": "Skill",
    "Designation": "Designation",
    "Degree": "Degree",
    "Experience": "Experience",
    "Years of Experience": "Experience",
}


def convert_to_bio(text, annotations):
    words = text.split()
    tags = ["O"] * len(words)

    for item in annotations or []:
        if not item.get("label"):
            continue
            
        label = TARGET_LABELS.get(item["label"][0])
        if label is None:
            continue
        start_char = item["points"][0]["start"]
        end_char = item["points"][0]["end"]

        current_char_idx = 0
        started = False

        for word_index, word in enumerate(words):
            word_start = text.find(word, current_char_idx)
            if word_start == -1:
                continue

            word_end = word_start + len(word)
            current_char_idx = word_end

            if word_start >= start_char and word_end <= end_char:
                prefix = "B-" if not started else "I-"
                tags[word_index] = f"{prefix}{label}"
                started = True

    return words, tags


class ResumeDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(file_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        text = sample["content"]
        annotations = sample.get("annotation", [])

        words, tags = convert_to_bio(text, annotations)

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                tag = tags[word_id] if word_id < len(tags) else "O"
                label_ids.append(LABEL_LIST.index(tag) if tag in LABEL_LIST else 0)
            else:
                tag = tags[word_id] if word_id < len(tags) else "O"
                if tag.startswith("B-"):
                    tag = tag.replace("B-", "I-", 1)
                label_ids.append(LABEL_LIST.index(tag) if tag in LABEL_LIST else 0)
            previous_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }

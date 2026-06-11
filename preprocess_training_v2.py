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

        spans = []
        for item in annotations or []:
            if not item.get("label"):
                continue

            label = TARGET_LABELS.get(item["label"][0])
            if label is None:
                continue
            points = item.get("points") or []
            if not points:
                continue
            start_char = points[0].get("start")
            end_char = points[0].get("end")
            if start_char is None or end_char is None or end_char <= start_char:
                continue
            spans.append({"start": start_char, "end": end_char, "label": label})

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offsets = encoding["offset_mapping"].squeeze(0).tolist()
        label_ids = []

        for start_char, end_char in offsets:
            if start_char == 0 and end_char == 0:
                label_ids.append(-100)
                continue

            tag = "O"
            for span in spans:
                if start_char < span["end"] and end_char > span["start"]:
                    if start_char <= span["start"] < end_char:
                        tag = f"B-{span['label']}"
                    else:
                        tag = f"I-{span['label']}"
                    break

            label_ids.append(LABEL_LIST.index(tag) if tag in LABEL_LIST else 0)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }

import torch

from preprocess_training import LABEL_LIST


def _merge_wordpieces(tokens):
    merged = ""
    for token in tokens:
        if token.startswith("##"):
            merged += token[2:]
        elif token.startswith('\u2581') or token.startswith('Ġ'):
            if merged:
                merged += " " + token[1:]
            else:
                merged = token[1:]
        elif merged:
            merged += token
        else:
            merged = token
    return merged.strip()


def _extract_entities(tokens, tag_ids, id2label):
    extracted = {"skills": [], "experience": [], "education": [], "designation": []}
    current_tokens = []
    current_label = None

    def flush_current():
        if not current_label or not current_tokens:
            return
        phrase = _merge_wordpieces(current_tokens)
        if not phrase:
            return
        if "Skill" in current_label:
            extracted["skills"].append(phrase)
        elif "Experience" in current_label:
            extracted["experience"].append(phrase)
        elif "Degree" in current_label:
            extracted["education"].append(phrase)
        elif "Designation" in current_label:
            extracted["designation"].append(phrase)

    for token, tag_id in zip(tokens, tag_ids):
        if token in ("[CLS]", "[SEP]", "[PAD]"):
            continue

        label = id2label.get(tag_id, "O")
        if label.startswith("B-"):
            flush_current()
            current_label = label
            current_tokens = [token]
        elif label.startswith("I-"):
            # Handle I- tags: if no current_label, treat I- as starting a new entity
            if current_label is None:
                current_label = f"B-{label[2:]}"
                current_tokens = [token]
            elif current_label == f"B-{label[2:]}":
                current_tokens.append(token)
            else:
                flush_current()
                current_label = f"B-{label[2:]}"
                current_tokens = [token]
        else:
            flush_current()
            current_label = None
            current_tokens = []

    flush_current()
    return extracted


def predict_cv_info(text, model, tokenizer, device, max_len=128, verbose=False):
    if not text:
        return {"skills": "", "experience": "", "education": "", "designation": ""}

    words = text.split()
    if not words:
        return {"skills": "", "experience": "", "education": "", "designation": ""}

    all_extracted = {"skills": set(), "experience": set(), "education": set(), "designation": set()}
    id2label = {i: label for i, label in enumerate(LABEL_LIST)}
    
    # Process in chunks of 75 words (approx 100 subwords, safe for max_len=128)
    # with an overlap of 15 words to prevent cutting entities in half
    chunk_size = 75
    overlap = 15
    
    start_idx = 0
    while start_idx < len(words):
        chunk_words = words[start_idx : start_idx + chunk_size]
        if not chunk_words:
            break
            
        encoding = tokenizer(
            chunk_words,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            predictions = model(input_ids, attention_mask=attention_mask)

        tag_ids = predictions[0] if predictions else []
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
        
        extracted = _extract_entities(tokens, tag_ids, id2label)
        for key in all_extracted:
            for item in extracted[key]:
                if item:
                    all_extracted[key].add(item)
                    
        if start_idx + chunk_size >= len(words):
            break
        start_idx += (chunk_size - overlap)

    return {
        "skills": ", ".join(sorted(all_extracted["skills"])),
        "experience": ", ".join(sorted(all_extracted["experience"])),
        "education": ", ".join(sorted(all_extracted["education"])),
        "designation": ", ".join(sorted(all_extracted["designation"])),
    }
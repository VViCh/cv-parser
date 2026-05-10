from transformers import AutoTokenizer

print("Preparing Tokenizer (DistilBERT)...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

words = ["I", "have", "experience", "with", "TensorFlow", "and", "PyTorch", "."]

print(f"\nOriginal Sentence: {' '.join(words)}")

tokenization_result = tokenizer(
    words,
    is_split_into_words=True,
    padding='max_length',
    truncation=True,
    max_length=15,
    return_tensors="pt"
)

split_tokens = tokenizer.convert_ids_to_tokens(tokenization_result["input_ids"][0])

print("\n1. WordPiece Tokens:")
print(split_tokens)

print("\n2. Input IDs:")
print(tokenization_result["input_ids"][0].tolist())

print("\n3. Attention Mask:")
print(tokenization_result["attention_mask"][0].tolist())

print("\nTokenization verified.")
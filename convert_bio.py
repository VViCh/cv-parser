def convert_to_bio(text, annotations):
    words = text.split()
    bio_tags = ["O"] * len(words)
    
    for item in annotations:
        label = item['label'][0] 
        start_char = item['points'][0]['start']
        end_char = item['points'][0]['end']
        
        current_char_idx = 0
        in_entity = False
        
        for i, word in enumerate(words):
            word_start = text.find(word, current_char_idx)
            word_end = word_start + len(word)
            current_char_idx = word_end
            
            if word_start >= start_char and word_end <= end_char:
                if not in_entity:
                    bio_tags[i] = f"B-{label}"
                    in_entity = True
                else:
                    bio_tags[i] = f"I-{label}"
    
    return words, bio_tags

if __name__ == "__main__":
    sample_text = "I have experience with Python and Java."
    sample_annotations = [
        {"label": ["Skill"], "points": [{"start": 23, "end": 28, "text": "Python"}]},
        {"label": ["Skill"], "points": [{"start": 34, "end": 37, "text": "Java"}]}
    ]

    words, labels = convert_to_bio(sample_text, sample_annotations)

    print("BIO Tags result:")
    for w, t in zip(words, labels):
        print(f"{w:15} \t {t}")
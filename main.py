for token, tag_id in zip(tokens, predicted_tags):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        label = id2label.get(tag_id, "O")
        
        clean_token = token.replace("##", "") if token.startswith("##") else " " + token
        
        if label.startswith("B-"):
            if current_word.strip() and current_label:
                if "Skill" in current_label: extracted_data['skills'].append(current_word.strip())
                elif "Experience" in current_label: extracted_data['experience'].append(current_word.strip())
                elif "Degree" in current_label: extracted_data['education'].append(current_word.strip())
            
            current_word = clean_token
            current_label = label
            
        elif label.startswith("I-") and current_label == label.replace("I-", "B-"):
            current_word += clean_token if token.startswith("##") else clean_token
        else:
            current_label = None

    if current_word.strip() and current_label:
        if "Skill" in current_label: extracted_data['skills'].append(current_word.strip())
        elif "Experience" in current_label: extracted_data['experience'].append(current_word.strip())
        elif "Degree" in current_label: extracted_data['education'].append(current_word.strip())
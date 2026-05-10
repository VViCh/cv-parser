import json

file_dataset = 'Entity Recognition in Resumes.json'

try:
    with open(file_dataset, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        resume_data = json.loads(first_line)
        
        if 'annotation' in resume_data:
            annotation = resume_data['annotation']
            print(f"Loaded {len(annotation)} entities from resume.")
            
            for item in annotation[:10]:
                label = item['label'][0] 
                original_text = item['points'][0]['text']
                start = item['points'][0]['start']
                end = item['points'][0]['end']
                print(f"[{label:^12}] {original_text} (Index: {start}-{end})")
        else:
            print("No annotations found.")
            
except FileNotFoundError:
    print("Error: Dataset file not found.")
except Exception as e:
    print(f"Error: {e}")
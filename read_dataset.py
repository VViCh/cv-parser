import json

file_path = 'Entity Recognition in Resumes.json' 

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        first_resume = json.loads(f.readline())
        
        print("Dataset read successfully.")
        print("--- File Snippet ---")
        print(first_resume['content'][:300] + "...\n")
        
except FileNotFoundError:
    print("Error: JSON dataset file not found.")
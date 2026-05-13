import json
from collections import Counter

counts = Counter()
invalid = 0

with open('cv/dataset.json', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        data = json.loads(line)
        for a in data.get('annotation', []):
            if not a.get('label'):
                invalid += 1
                continue
            counts[a['label'][0]] += 1

print('Entity counts:', dict(counts))
print('Invalid annotations:', invalid)

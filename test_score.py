import fitz
doc = fitz.open('paper.pdf')
for i in range(min(5, len(doc))):
    text = doc[i].get_text()
    print(f'=== PAGE {i+1} ===')
    print(text[:4000])
    print('---END PAGE---')

import fitz

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            blocks = page.get_text("blocks")
            for block in blocks:
                if len(block) >= 5:
                    text += block[4] + " "
        return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

if __name__ == "__main__":
    pdf_path = "Sample CV/data/data/INFORMATION-TECHNOLOGY/10089536.pdf" 
    print(f"Reading PDF: {pdf_path}")
    text_cv = extract_text_from_pdf(pdf_path)
    
    if text_cv:
        print("--- Text Snippet ---")
        print(text_cv[:500] + "...")
        print("--- End of Snippet ---")
    else:
        print("Unable to extract text.")
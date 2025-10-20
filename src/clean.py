import fitz  # PyMuPDF
import re
import os

def extract_text(pdf_path):
    """Extract raw text from a PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def clean_text(text):
    """Clean legal text while preserving clauses and definitions."""
    # 1. Remove page numbers (e.g., "Page 1 of 12")
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)

    # 2. Remove headers/footers that repeat (common in legal PDFs)
    # Example: lines in ALL CAPS at start/end of page
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # skip very short or repetitive lines (likely headers/footers)
        if len(line) < 5:
            continue
        # skip lines that look like TOC entries: Section 1 ... 5
        if re.match(r'^Section\s+\d+.*\d+$', line):
            continue
        # skip lines that are just page numbers
        if re.match(r'^\d+$', line):
            continue
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)

    # 3. Fix broken words split across lines by hyphens
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

    # 4. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def save_clean_text(clean_text, output_path):
    """Save cleaned text to a .txt file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(clean_text)

def process_pdf_folder(input_folder, output_folder):
    """Process all PDFs in a folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            raw_text = extract_text(pdf_path)
            cleaned = clean_text(raw_text)
            output_path = os.path.join(output_folder, filename.replace('.pdf', '.txt'))
            save_clean_text(cleaned, output_path)
    print("Done!")

# Example usage:
input_folder = "data/raw"
output_folder = "data/processed"
process_pdf_folder(input_folder, output_folder)

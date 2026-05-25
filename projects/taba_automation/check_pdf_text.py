import pdfplumber
import os

pdf_path = 'downloaded_plans/plan_504-1064039_horhaot.pdf'
with pdfplumber.open(pdf_path) as pdf:
    text = pdf.pages[0].extract_text()
    if text:
        # Save to file to avoid console encoding issues
        with open('debug_extracted_text.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Text extracted and saved to debug_extracted_text.txt")
        
        # Check for keywords
        keywords = ['תיירות', 'מלון', 'מלונ', 'קייט', 'אירוח']
        for kw in keywords:
            if kw in text:
                print(f"Matched: {kw}")
            else:
                # Check for reversed
                rev_kw = kw[::-1]
                if rev_kw in text:
                    print(f"Matched REVERSED: {rev_kw} (original: {kw})")
    else:
        print("No text extracted.")

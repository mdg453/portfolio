import pdfplumber
import logging
import re

class DocumentParser:
    def __init__(self):
        self.keywords = ['תיירות', 'מלונ', 'מלון', 'קייט', 'אירוח']

    def parse_pdf(self, pdf_path):
        """
        Parses the 'Horhaot' (Instructions) PDF document.
        Returns a dictionary with findings.
        """
    def parse_pdf(self, pdf_path):
        """
        Parses the 'Horhaot' (Instructions) PDF document.
        Handles reversed Hebrew effectively.
        """
        result = {
            'has_tourism_keywords': False,
            'tourism_keyword_matches': [],
            'has_hotel_in_table_5': False,
            'table_5_data': []
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        # Some PDFs have reversed Hebrew, some don't.
                        # We check against normal and reversed text.
                        lines = text.split('\n')
                        reversed_text = '\n'.join([line[::-1] for line in lines])
                        
                        full_content_to_check = text + "\n" + reversed_text
                        
                        for keyword in self.keywords:
                            if keyword in full_content_to_check:
                                result['has_tourism_keywords'] = True
                                if keyword not in result['tourism_keyword_matches']:
                                    result['tourism_keyword_matches'].append(keyword)
                                    
                    # 2. Extract Tables (Looking for Table 5)
                    tables = page.extract_tables()
                    for table in tables:
                        table_str = str(table)
                        
                        # Check for tourism indicators in both directions
                        found_in_table = False
                        for kw in self.keywords:
                            if kw in table_str or kw[::-1] in table_str:
                                found_in_table = True
                                break
                        
                        if found_in_table:
                            result['has_hotel_in_table_5'] = True
                            
                        # If table contains indicators of Table 5 (rights/zones), gather rows
                        # "עיקרי", "שטח", "זכויות"
                        indicators = ["עיקרי", "שטח", "זכויות", "ירקיע", "חטש", "תויוכז"]
                        if any(ind in table_str for ind in indicators):
                            for row in table:
                                clean_row = [str(cell) if cell else "" for cell in row]
                                row_str = " ".join(clean_row)
                                
                                is_tourism_row = False
                                for kw in self.keywords:
                                    if kw in row_str or kw[::-1] in row_str:
                                        is_tourism_row = True
                                        break
                                
                                if is_tourism_row:
                                    result['table_5_data'].append(clean_row)

        except Exception as e:
            logging.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            
        return result

if __name__ == '__main__':
    # Test block
    parser = DocumentParser()
    print("Document parser initialized.")
    # Example usage:
    # res = parser.parse_pdf('sample.pdf')
    # print(res)

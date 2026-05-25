"""
Run the DocumentParser algorithm on all PDF files in the test_data folder.
"""
import os
import sys
import json

# Fix Windows console encoding for Hebrew
sys.stdout.reconfigure(encoding='utf-8')

from document_parser import DocumentParser

def main():
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'test_data')
    test_data_dir = os.path.abspath(test_data_dir)
    
    if not os.path.exists(test_data_dir):
        print(f"ERROR: test_data directory not found at {test_data_dir}")
        return
    
    pdf_files = [f for f in os.listdir(test_data_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in test_data directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files in test_data/\n")
    print("=" * 80)
    
    parser = DocumentParser()
    
    results_summary = []
    
    for pdf_file in sorted(pdf_files):
        pdf_path = os.path.join(test_data_dir, pdf_file)
        print(f"\n{'=' * 80}")
        print(f"FILE: {pdf_file}")
        print(f"SIZE: {os.path.getsize(pdf_path) / 1024:.1f} KB")
        print(f"{'-' * 80}")
        
        result = parser.parse_pdf(pdf_path)
        
        is_tourism = result.get('has_tourism_keywords') or result.get('has_hotel_in_table_5')
        
        print(f"  Tourism Keywords Found: {result['has_tourism_keywords']}")
        if result['tourism_keyword_matches']:
            print(f"  Matched Keywords: {', '.join(result['tourism_keyword_matches'])}")
        print(f"  Hotel in Table 5: {result['has_hotel_in_table_5']}")
        if result['table_5_data']:
            print(f"  Table 5 Tourism Rows ({len(result['table_5_data'])}):")
            for row in result['table_5_data']:
                print(f"    -> {row}")
        print(f"  ** CLASSIFIED AS TOURISM: {'YES' if is_tourism else 'NO'} **")
        
        results_summary.append({
            'file': pdf_file,
            'is_tourism': is_tourism,
            'keywords_found': result['has_tourism_keywords'],
            'matched_keywords': result['tourism_keyword_matches'],
            'hotel_in_table_5': result['has_hotel_in_table_5'],
            'table_5_rows': len(result['table_5_data'])
        })
    
    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    tourism_count = sum(1 for r in results_summary if r['is_tourism'])
    print(f"Total files: {len(results_summary)}")
    print(f"Tourism plans: {tourism_count}")
    print(f"Non-tourism plans: {len(results_summary) - tourism_count}")
    print()
    
    for r in results_summary:
        status = "✓ TOURISM" if r['is_tourism'] else "✗ NOT TOURISM"
        keywords = f" [{', '.join(r['matched_keywords'])}]" if r['matched_keywords'] else ""
        print(f"  {status:15s} | {r['file']}{keywords}")

if __name__ == '__main__':
    main()

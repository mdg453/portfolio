import logging
from document_fetcher import DocumentFetcher
from document_parser import DocumentParser
from arcgis_client import ArcGISClient

logging.basicConfig(level=logging.INFO)

plan_num = "504-1064039"
fetcher = DocumentFetcher()
parser = DocumentParser()
arcgis = ArcGISClient()

print(f"--- Testing known tourism plan: {plan_num} ---")
pdf_path = fetcher.fetch_plan_instructions(plan_num)
if pdf_path:
    print(f"PDF Downloaded: {pdf_path}")
    res = parser.parse_pdf(pdf_path)
    print(f"Parser Results: {res}")
    
    plots = arcgis.get_land_uses_for_plan(plan_num)
    print(f"ArcGIS Plots found: {len(plots)}")
else:
    print("Failed to download PDF.")

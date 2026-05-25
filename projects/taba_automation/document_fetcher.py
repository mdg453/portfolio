import os
import requests
import logging
from mavat_scraper import MavatScraper
from arcgis_client import ArcGISClient

class DocumentFetcher:
    def __init__(self, download_dir='downloaded_plans'):
        self.download_dir = os.path.abspath(download_dir)
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
        self.mavat_scraper = MavatScraper(download_dir)
        self.arcgis_client = ArcGISClient()

    def fetch_plan_instructions(self, plan_number):
        """
        Fetches the Instructions document (Horhaot/Takanon) for a given plan 
        by leveraging the RMI (Israel Land Authority) API.
        Returns the path to the downloaded PDF, or None if not found/failed.
        """
        expected_filename = f"plan_{plan_number.replace('/', '_')}_horhaot.pdf"
        final_file_path = os.path.join(self.download_dir, expected_filename)
        
        # If we already have the file locally and it's not empty, return it
        if os.path.exists(final_file_path) and os.path.getsize(final_file_path) > 1000:
            return final_file_path
            
        logging.info(f"Attempting to download document for plan {plan_number} via RMI API...")
        
        url = "https://apps.land.gov.il/TabaSearch/api/SerachPlans/GetPlans"
        headers = {"Content-Type": "application/json"}
        payload = {"planNumber": plan_number}
        
        try:
            # 1. Search for the plan to get document paths
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            if response.status_code != 200:
                logging.error(f"Failed to query RMI API for {plan_number}. Status: {response.status_code}")
                return None
                
            data = response.json()
            plans = data.get("plansSmall", [])
            
            if not plans:
                logging.warning(f"No plans found in RMI API for {plan_number}")
                return None
                
            plan_data = plans[0]
            takanon = plan_data.get("documentsSet", {}).get("takanon")
            
            if takanon and takanon.get("path"):
                path = takanon["path"]
                if not path.startswith("/"):
                    path = "/" + path
                path = path.replace("\\", "/")
                
                download_url = "https://apps.land.gov.il" + path
                logging.info(f"Found document URL: {download_url}. Downloading...")
                
                # 2. Download the PDF document
                pdf_response = requests.get(download_url, timeout=60, stream=True)
                if pdf_response.status_code == 200:
                    with open(final_file_path, "wb") as f:
                        for chunk in pdf_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logging.info(f"Successfully downloaded: {final_file_path}")
                    return final_file_path
                else:
                    logging.error(f"Failed to download PDF {download_url}. Status: {pdf_response.status_code}")
                    return None
            else:
                logging.warning(f"Plan {plan_number} found, but no 'takanon' (Instructions) path available.")
                return None
                
        except Exception as e:
            logging.error(f"Failed to fetch document for {plan_number} using RMI API: {str(e)}")
            
        # --- FALLBACK: Try Playwright Mavat Scraper if RMI failed or plan not found ---
        logging.info(f"RMI API failed for {plan_number}. Falling back to Mavat Scraper...")
        mavat_url = self.arcgis_client.get_plan_url(plan_number)
        if not mavat_url or "mavat" not in mavat_url:
            logging.error(f"Could not find a valid Mavat URL for {plan_number}. Cannot use fallback scraper.")
            return None
            
        return self.mavat_scraper.download_instructions_pdf(mavat_url, plan_number)

    def close(self):
        # Included for compatibility with main.py if previously using Selenium
        pass

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fetcher = DocumentFetcher()
    print("Testing RMI Rest API Fetcher for plan 504-1064039...")
    pdf = fetcher.fetch_plan_instructions("504-1064039")
    print("Result:", pdf)

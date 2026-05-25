import os
import time
from playwright.sync_api import sync_playwright, TimeoutError
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MavatScraper')

class MavatScraper:
    def __init__(self, download_dir):
        self.download_dir = download_dir
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def download_instructions_pdf(self, mavat_url, plan_number):
        """
        Navigates to the Mavat URL and downloads the "הוראות התכנית" (Instructions) PDF.
        Returns the path to the downloaded file, or None if failed.
        """
        final_file_path = os.path.join(self.download_dir, f"plan_{plan_number}_horhaot.pdf")
        
        # If we already have the file and it's valid, return it
        if os.path.exists(final_file_path) and os.path.getsize(final_file_path) > 1000:
            return final_file_path

        logger.info(f"Starting Playwright scraper for plan {plan_number} at {mavat_url}")
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                # Create context with standard user agent to avoid basic blocks
                context = browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                    accept_downloads=True
                )
                page = context.new_page()
                page.set_default_timeout(30000)
                
                # Navigate and wait for main content to load
                page.goto(mavat_url, wait_until='domcontentloaded', timeout=60000)
                
                # Click the documents tab
                try:
                    page.get_by_text('מסמכי התכנית').click(timeout=15000)
                    time.sleep(3) # Wait for table to load
                except TimeoutError:
                    logger.warning(f"Could not find 'מסמכי התכנית' tab for {plan_number}.")
                    browser.close()
                    return None

                # Mavat documents are categorized. Category C (catC) is 'הוראות' (Instructions)
                doc_title = page.locator('li.catC .uk-accordion-title').first
                
                if doc_title.count() == 0:
                    logger.warning(f"Could not find 'הוראות' category accordion for {plan_number}.")
                    browser.close()
                    return None
                    
                # Click to expand the accordion if it is hidden
                content = page.locator('li.catC .uk-accordion-content').first
                if content.get_attribute('hidden') is not None:
                    doc_title.click(force=True, timeout=5000)
                    time.sleep(2)
                
                # Now find the download button inside the expanded content for the first document
                download_btn = content.locator('.sv4-icon-docs-download').first

                try:
                    with page.expect_download(timeout=30000) as download_info:
                        download_btn.click(force=True, timeout=5000)
                    
                    download = download_info.value
                    download.save_as(final_file_path)
                    logger.info(f"Successfully downloaded {plan_number} via Playwright Mavat Scraper.")
                    browser.close()
                    return final_file_path
                except Exception as e:
                    logger.error(f"Failed to trigger download for {plan_number}: {e}")
                    browser.close()
                    return None
                    
        except Exception as e:
            logger.error(f"Playwright error for {plan_number}: {e}")
            return None

if __name__ == "__main__":
    # Test script
    scraper = MavatScraper(download_dir="test_downloads")
    res = scraper.download_instructions_pdf('https://mavat.iplan.gov.il/SV4/1/1005134322/310', '101-0967455')
    print("Download result:", res)

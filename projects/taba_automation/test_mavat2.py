from playwright.sync_api import sync_playwright
import sys
sys.stdout.reconfigure(encoding='utf-8')

url = 'https://mavat.iplan.gov.il/SV4/1/1005134322/310'

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url, timeout=60000)
    page.get_by_text('מסמכי התכנית').click(timeout=15000)
    page.wait_for_timeout(5000)
    
    el = page.locator('text="הוראות"').first
    if el.count() > 0:
        html = el.evaluate('el => el.parentElement.parentElement.parentElement.outerHTML')
        print(html[:1500])
    browser.close()

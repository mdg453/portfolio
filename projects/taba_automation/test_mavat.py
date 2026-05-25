from playwright.sync_api import sync_playwright
import sys
import time
import json

sys.stdout.reconfigure(encoding='utf-8')

url = 'https://mavat.iplan.gov.il/SV4/1/1005134322/310'

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url, timeout=60000)
    print('Title:', page.title())
    
    # Wait for the network to be somewhat idle or for the main app to load
    try:
        page.wait_for_selector('app-root', timeout=30000)
    except Exception as e:
        print("Wait app-root error:", e)
        
    time.sleep(10) # Give it time to render
    
    # Try to find 'מסמכי התכנית' tab and click it
    try:
        page.get_by_text('מסמכי התכנית').click(timeout=10000)
        print("Clicked 'מסמכי התכנית' tab!")
        time.sleep(5)
    except Exception as e:
        print("Could not click documents tab:", e)
        
    # See if we can find 'הוראות התכנית'
    els = page.locator('text="הוראות"').all()
    print('Found הוראות elements:', len(els))
    
    # Print hrefs of any links containing pdf
    pdf_links = page.evaluate('''
        () => Array.from(document.querySelectorAll('a, button')).map(el => {
            if (el.tagName.toLowerCase() === 'a') return el.href;
            return el.innerText;
        }).filter(t => t && (t.includes('pdf') || t.includes('הוראות')))
    ''')
    print('PDF/Download Links/Texts:', json.dumps(pdf_links, ensure_ascii=False))

    browser.close()

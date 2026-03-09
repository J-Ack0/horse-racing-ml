from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import re

START_URL = "https://www.racingtv.com/racecards/tomorrow"

def make_absolute_url(link, base):
    if link.startswith("http"):
        return link
    return base.rstrip("/") + "/" + link.lstrip("/")

def scrape_irish_racecards_selenium(url):
    try:
        # Set up headless Chromium
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=chrome_options)
        
        driver.get(url)
        # Optional: wait for JS to load
        driver.implicitly_wait(5)
        
        # Get page source
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        
        # Look for meeting boxes by class name
        boxes = soup.find_all(class_="ui-accordion__row")
        all_links = []

        for box in boxes:
            links = box.find_all("a")
            hrefs = [link.get("href") for link in links if link.get("href")]
            all_links.extend(hrefs)

        # Fallback: all <a> links
        if not all_links:
            links = soup.find_all("a", href=True)
            all_links = [link.get("href") for link in links if link.get("href")]

        # Filter links to Irish racecards
        irish_links = [make_absolute_url(link, START_URL) for link in all_links
                       if "/ireland/" in link or "/down-royal" in link]

        driver.quit()

        return {
            "found": bool(irish_links),
            "links": irish_links,
            "html_snippet": html[:1000]  # first 1000 chars
        }

    except Exception as e:
        print(f"[ERROR] Failed to extract links: {e}")
        return {"found": False, "links": [], "html_snippet": ""}

# Usage
result = scrape_irish_racecards_selenium(START_URL)
print("Irish racecards found:", result["found"])
print("Links:")
for l in result["links"]:
    print(l)

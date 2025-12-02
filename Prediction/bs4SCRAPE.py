# racingpost_links_simple.py
"""
Selenium + BeautifulSoup helper:
- opens RacingPost racecards/tomorrow
- dismisses cookie banner
- expands accordion rows and scrolls
- collects ALL <a class="RC-meetingItem__link" href=...> found inside section.ui-accordion__row
- prints deduplicated absolute URLs

Usage:
    pip install selenium webdriver-manager bs4
    python racingpost_links_simple.py       # visible browser
    python racingpost_links_simple.py --headless
"""

import time
import argparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

START_URL = "https://www.racingpost.com/racecards/tomorrow"
BASE_URL = "https://www.racingpost.com"

def make_absolute_url(url, base=BASE_URL):
    if not url:
        return None
    return url if url.startswith("http") else base.rstrip("/") + url

def get_driver(headless=False):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,900")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    return driver

def wait_for_ready(driver, timeout=15):
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
    except TimeoutException:
        print("[WARN] document.readyState did not become 'complete' within timeout — continuing.")

def try_click(driver, element):
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
        element.click()
        return True
    except (ElementClickInterceptedException, Exception):
        try:
            driver.execute_script("arguments[0].click();", element)
            return True
        except Exception:
            return False

def dismiss_cookie_banner(driver, timeout=6):
    """
    Tries a few common targets for cookie/consent dismissal.
    Returns (True, reason) if something was clicked.
    """
    attempts = []
    attempts.append(("id", "truste-consent-required"))
    attempts.append(("class", "trustarc-declineall-btn"))
    decline_texts = ["decline all", "reject all", "reject", "decline", "no thanks", "deny all"]
    for txt in decline_texts:
        attempts.append(("xpath_text", f"//button[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{txt}')]"))
        attempts.append(("xpath_text_a", f"//a[contains(translate(normalize-space(.), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{txt}')]"))
    attempts.append(("css", "button[aria-label*='decline'], button[data-testid*='decline']"))
    attempts.append(("css", "button[aria-label*='reject']"))

    for method, sel in attempts:
        try:
            if method == "id":
                el = driver.find_element(By.ID, sel)
                if try_click(driver, el):
                    return True, f"clicked id='{sel}'"
            elif method == "class":
                els = driver.find_elements(By.CLASS_NAME, sel)
                for el in els:
                    if try_click(driver, el):
                        return True, f"clicked class='{sel}'"
            elif method.startswith("xpath"):
                els = driver.find_elements(By.XPATH, sel)
                for el in els:
                    if try_click(driver, el):
                        return True, f"clicked xpath='{sel}'"
            elif method == "css":
                els = driver.find_elements(By.CSS_SELECTOR, sel)
                for el in els:
                    if try_click(driver, el):
                        return True, f"clicked css='{sel}'"
        except NoSuchElementException:
            continue
        except Exception:
            continue

    # last-ditch: find any button with decline-like text
    try:
        candidates = driver.find_elements(By.XPATH, "//button|//a")
        for el in candidates:
            txt = (el.text or "").strip().lower()
            if any(k in txt for k in ["decline", "reject", "deny", "no thanks"]):
                if try_click(driver, el):
                    return True, f"clicked generic with text='{txt[:30]}'"
    except Exception:
        pass

    return False, "no cookie dismiss button clicked"

def expand_all_accordion_rows(driver):
    js = """
    (function(){
        const rows = document.querySelectorAll('section.ui-accordion__row');
        let cnt = 0;
        rows.forEach(function(row){
            cnt++;
            try{
                row.classList.add('js-accordion__row_active');
                const panel = row.querySelector('.ui-accordion__panel, .ui-accordion__content');
                if(panel){
                    panel.style.display = 'block';
                    panel.style.visibility = 'visible';
                    panel.style.height = 'auto';
                }
                row.querySelectorAll('[data-lazy], .lazy-load').forEach(function(el){
                    el.style.display='block';
                    el.style.visibility='visible';
                });
            }catch(e){}
        });
        return cnt;
    })();
    """
    try:
        return int(driver.execute_script(js) or 0)
    except Exception:
        return 0

def incremental_scroll(driver, pause=0.15):
    try:
        scroll_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.documentElement.scrollHeight);")
        step = 700
        for top in range(0, int(scroll_height)+1, step):
            driver.execute_script(f"window.scrollTo(0, {top});")
            time.sleep(pause)
        time.sleep(0.4)
    except Exception:
        time.sleep(0.4)

def extract_meetingitem_links_from_html(html):
    """
    Parse HTML and extract all <a class="RC-meetingItem__link" href=...> that live inside
    section.ui-accordion__row elements. No regex filtering — returns whatever hrefs found.
    """
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.find_all("section", class_=lambda c: c and "ui-accordion__row" in c)
    found = []
    for card in cards:
        a_tags = card.find_all("a", class_="RC-meetingItem__link", href=True)
        for a in a_tags:
            href = a.get("href", "").strip()
            if href:
                found.append(make_absolute_url(href))

    # fallback: if nothing found in cards, capture any RC-meetingItem__link anywhere
    if not found:
        anchors = soup.find_all("a", class_="RC-meetingItem__link", href=True)
        for a in anchors:
            href = a.get("href", "").strip()
            if href:
                found.append(make_absolute_url(href))

    # dedupe preserving order
    seen = set(); dedup = []
    for u in found:
        if u not in seen:
            seen.add(u); dedup.append(u)

    debug = {
        "sections": len(cards),
        "inside_card_links": len(found),
        "deduped_links": len(dedup)
    }
    return dedup, debug

def main(headless=False, keep_open_seconds=3):
    print(f"Starting (headless={headless}) -> {START_URL}")
    driver = get_driver(headless=headless)
    try:
        driver.get(START_URL)
        wait_for_ready(driver, timeout=12)

        clicked, reason = dismiss_cookie_banner(driver, timeout=6)
        print(f"[COOKIE] dismissed={clicked}, reason='{reason}'")
        time.sleep(0.5)

        toggled = expand_all_accordion_rows(driver)
        print(f"[INFO] attempted to expand {toggled} accordion rows")
        incremental_scroll(driver)
        time.sleep(0.6)

        html = driver.page_source
        links, debug = extract_meetingitem_links_from_html(html)

        print("===== DEBUG =====")
        print(f"Sections found: {debug['sections']}")
        print(f"Links found inside sections (raw): {debug['inside_card_links']}")
        print(f"Unique links returned: {debug['deduped_links']}")
        print("=================\n")

        if links:
            print("Found RC-meetingItem__link hrefs:")
            for ln in links:
                print(ln)
        else:
            print("No RC-meetingItem__link anchors found. Try headful mode to inspect the page.")

        if not headless:
            print(f"\nKeeping browser open for {keep_open_seconds}s so you can inspect it...")
            time.sleep(keep_open_seconds)

    finally:
        try:
            driver.quit()
        except Exception:
            pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="Run headless")
    ap.add_argument("--keep-open", type=int, default=3, help="Seconds to keep browser open")
    args = ap.parse_args()
    main(headless=args.headless, keep_open_seconds=args.keep_open)

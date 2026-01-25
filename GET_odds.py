import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import os
from datetime import datetime, timedelta

# Selenium Imports
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def clean_text(text):
    """Removes newlines, tabs, and collapses multiple spaces."""
    if not text:
        return ""
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_links_with_retries(url):
    """Fetches racecard links with retry logic for 4xx errors using BeautifulSoup."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    base_url = "https://www.racingpost.com"
    
    # Retry logic for 4xx errors
    for attempt in range(3):
        try:
            print(f"--- Attempt {attempt + 1}: Fetching links from {url} ---")
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # Gentle wait after successful response
                time.sleep(1)
                
                soup = BeautifulSoup(response.text, 'html.parser')
                race_links = soup.find_all('a', attrs={"data-race-is-over": True})
                
                results = []
                for a in race_links:
                    href = a.get('href')
                    if href and "/racecards/" in href:
                        results.append(href if href.startswith('http') else base_url + href)
                
                unique_links = list(dict.fromkeys(results))
                print(f"Found {len(unique_links)} unique racecards.\n")
                return unique_links
            
            elif 400 <= response.status_code < 500:
                print(f"  [!] Received {response.status_code}. Waiting 5 seconds before retry...")
                time.sleep(5)
            else:
                print(f"  [!] Received {response.status_code}. Breaking retry loop.")
                break
                
        except Exception as e:
            print(f"  [!] Connection error: {e}")
            time.sleep(5)
            
    return []

def calculate_trend(odds_list):
    """
    Calculate trend based on odds movement.
    Returns: 'shortening', 'drifting', 'stable', or 'no_data'
    """
    # Filter out empty values and convert to floats
    valid_odds = []
    for odds_str in odds_list:
        if odds_str and odds_str.strip():
            try:
                # Handle fractional odds (e.g., "5/1" -> 6.0, "10/1" -> 11.0)
                if '/' in odds_str:
                    parts = odds_str.split('/')
                    numerator = float(parts[0])
                    denominator = float(parts[1])
                    decimal_odds = (numerator / denominator) + 1
                    valid_odds.append(decimal_odds)
                else:
                    valid_odds.append(float(odds_str))
            except:
                continue
    
    if len(valid_odds) < 2:
        return "no_data"
    
    # Compare first and last odds
    first_odds = valid_odds[0]
    last_odds = valid_odds[-1]
    
    # Calculate percentage change
    change_percent = ((last_odds - first_odds) / first_odds) * 100
    
    if change_percent < -5:  # Odds shortened by more than 5%
        return "shortening"
    elif change_percent > 5:  # Odds drifted by more than 5%
        return "drifting"
    else:
        return "stable"

def scrape_race_data_selenium(driver, url):
    """Uses Selenium to scrape odds data from individual racecard."""
    try:
        print(f"  Scraping: {url}")
        driver.get(url)
        
        # Wait until the runner rows are present in the DOM
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CLASS_NAME, "RC-runnerRow")))
        
        # Additional wait for dynamic data to populate (2.5 seconds)
        time.sleep(2.5)
        
        # Parse the loaded page source with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        main_section = soup.select_one('[data-site-sub-section-1="Races"]')
        
        if not main_section:
            print("  [!] Main section not found")
            return []

        race_rows = []
        runner_containers = main_section.select('.RC-runnerRow')
        
        for container in runner_containers:
            # Get horse name
            horse_name_elem = container.select_one('[data-test-selector="RC-cardPage-runnerName"]')
            horse_name = clean_text(horse_name_elem.text) if horse_name_elem else ""
            # Inside your loop:
            horse_name = horse_name.removesuffix(" right")
            if not horse_name:
                continue

            # Get live odds
            odds_live_elem = container.select_one('[data-test-selector="RC-cardPage-runnerPrice"]')
            odds_live = clean_text(odds_live_elem.text) if odds_live_elem else ""
            
            # Get historical odds (1-4)
            odds_history = []
            for i in range(1, 5):
                odds_elem = container.select_one(f'[data-test-selector="RC-historyPrices-item-{i}"]')
                odds_value = clean_text(odds_elem.text) if odds_elem else ""
                odds_history.append(odds_value)
            
            # Calculate trend
            all_odds = [odds_live] + odds_history
            trend = calculate_trend(all_odds)
            
            row = {
                "URL" : url,
                "horse_name": horse_name,
                "odds_live": odds_live,
                "odds_1": odds_history[0],
                "odds_2": odds_history[1],
                "odds_3": odds_history[2],
                "odds_4": odds_history[3],
                "trend": trend
            }
            print(row["horse_name"],row["odds_live"])
            race_rows.append(row)
        
        print(f"  [+] Scraped {len(race_rows)} horses with odds data")
        return race_rows

    except Exception as e:
        print(f"  [!] Selenium error on {url}: {e}")
        return []

if __name__ == "__main__":
    print("=" * 60)
    print("RACING POST ODDS SCRAPER")
    print("=" * 60)
    
    # --- SETUP SELENIUM ---
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment to run without a window
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=chrome_options)

    tomorrow_url = "https://www.racingpost.com/racecards/tomorrow/"
    
    # 1. Get URLs using BeautifulSoup (with retries)
    print("\nStep 1: Collecting race URLs...")
    all_race_urls = get_links_with_retries(tomorrow_url)
    
    if not all_race_urls:
        print("\n[!] No race URLs found. Exiting.")
        driver.quit()
        exit()
    
    # 2. Scrape odds data using Selenium
    print(f"\nStep 2: Scraping odds data from {len(all_race_urls)} races...")
    master_data = []
    
    try:
        for idx, url in enumerate(all_race_urls, 1):
            print(f"\n[{idx}/{len(all_race_urls)}]")
            data = scrape_race_data_selenium(driver, url)
            master_data.extend(data)
            
            # Gentle delay between requests
            if idx < len(all_race_urls):
                time.sleep(1.5)
                
    finally:
        driver.quit()
        print("\nSelenium driver closed.")

    # 3. Save to CSV
    folder_name = "Inference_Odds"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"\nCreated folder: {folder_name}")

    if master_data:
        df = pd.DataFrame(master_data)
        tomorrow_date = (datetime.now() + timedelta(days=0)).strftime("%Y-%m-%d")        
        output_file = f"{folder_name}/Odds_{tomorrow_date}.csv"
        
        df.to_csv(output_file, index=False)
        print(f"\n{'=' * 60}")
        print(f"SCRAPING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total horses scraped: {len(master_data)}")
        print(f"Data saved to: {output_file}")
        print(f"\nColumns: {', '.join(df.columns.tolist())}")
        print(f"\nSample data:")
        print(df.head(3).to_string(index=False))
    else:
        print("\n[!] No data collected. Check the URLs and selectors.")
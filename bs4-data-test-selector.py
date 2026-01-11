import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from datetime import datetime, timedelta
import os

def clean_text(text):
    """Removes newlines, tabs, and collapses multiple spaces into one."""
    if not text:
        return ""
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_links_directly_from_tag(url):
    """Fetches all racecard links for tomorrow using the data-race-is-over attribute."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    base_url = "https://www.racingpost.com"
    
    try:
        print(f"--- Step 1: Fetching race links from {url} ---")
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Sniper approach: target only <a> tags with the specific data attribute
        race_links = soup.find_all('a', attrs={"data-race-is-over": True})
        
        results = []
        for a in race_links:
            href = a.get('href')
            if href:
                full_url = href if href.startswith('http') else base_url + href
                # Filter to ensure we only get racecards, not results or other pages
                if "/racecards/" in full_url:
                    results.append(full_url)

        unique_links = list(dict.fromkeys(results))
        print(f"Found {len(unique_links)} unique racecards to scrape.\n")
        return unique_links

    except Exception as e:
        print(f"Error fetching links: {e}")
        return []

def scrape_race_data(url):
    """Scrapes individual racecard data and returns a list of dictionaries."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')

        main_section = soup.select_one('[data-site-sub-section-1="Races"]')
        if not main_section:
            print(f"  [!] Skipped: Races section not found for {url}")
            return []

        # Header Extraction
        raw_date = clean_text(getattr(main_section.select_one('[data-test-selector="RC-courseHeader__date"]'), 'text', ""))
        clean_date = re.sub(r'\s*RTV\s*', '', raw_date)

        raw_going = clean_text(getattr(main_section.select_one('[data-test-selector="RC-headerBox__going"]'), 'text', ""))
        clean_going = re.sub(r'^Going:\s*', '', raw_going)

        race_header = {
            "track_name": clean_text(getattr(main_section.select_one('[data-test-selector="RC-courseHeader__name"]'), 'text', "")),
            "race_date": clean_date,
            "distance": clean_text(getattr(main_section.select_one('[data-test-selector="RC-header__raceDistanceRound"]'), 'text', "")),
            "going": clean_going,
            "time": clean_text(getattr(main_section.select_one('[data-test-selector="RC-courseHeader__time"]'), 'text', ""))
        }

        # Runner Extraction
        race_rows = []
        runner_containers = main_section.select('.RC-runnerRow')
        
        for container in runner_containers:
            h_name = clean_text(getattr(container.select_one('[data-test-selector="RC-cardPage-runnerName"]'), 'text', ""))
            if not h_name: continue

            row = {
                **race_header,
                "horse_name": h_name,
                "rating": clean_text(getattr(container.select_one('[data-test-selector="RC-cardPage-runnerRpr"]'), 'text', "")),
                "jockey": clean_text(getattr(container.select_one('[data-test-selector="RC-cardPage-runnerJockey-name"]'), 'text', "")),
                "trainer": clean_text(getattr(container.select_one('[data-test-selector="RC-cardPage-runnerTrainer-name"]'), 'text', "")),
                "weight": clean_text(getattr(container.select_one('[data-test-selector="RC-cardPage-runnerWgt-carried"]'), 'text', "")),
                "age": clean_text(getattr(container.select_one('[data-test-selector="RC-cardPage-runnerAge"]'), 'text', "")),
                "claims": clean_text(getattr(container.select_one('[data-test-selector="RC-cardPage-runnerJockey-allowance"]'), 'text', ""))
            }

            # Formatting
            if len(row["weight"]) >= 3:
                row["weight"] = f"{row['weight'][:-2]}-{row['weight'][-2:]}"
            row["claims"] = row["claims"] if row["claims"] else "0"
            
            race_rows.append(row)
        
        print(f"  [+] Scraped {len(race_rows)} horses from {race_header['track_name']}")
        return race_rows

    except Exception as e:
        print(f"  [!] Error scraping {url}: {e}")
        return []

# --- MAIN EXECUTION PIPELINE ---
if __name__ == "__main__":
    tomorrow_url = "https://www.racingpost.com/racecards/tomorrow"
    
    # 1. Get the list of URLs
    all_race_urls = get_links_directly_from_tag(tomorrow_url)
    
    # 2. Loop through URLs and collect data
    master_data = []
    
    for url in all_race_urls:
        race_data = scrape_race_data(url)
        master_data.extend(race_data)
        # Polite delay to avoid being blocked
        time.sleep(1)

    # 1. Create the folder if it doesn't exist
    folder_name = "Inference_Inputs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if master_data:
        df = pd.DataFrame(master_data)
        df = df.drop_duplicates(subset=["track_name", "horse_name", "jockey", "weight"])

        # 2. Corrected Datetime Output
        # %Y-%m-%d creates: 2026-01-09
        tomorrow_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")        
        # 3. Corrected F-string and pathing
        output_file = f"{folder_name}/Input_{tomorrow_date}.csv"
        
        df.to_csv(output_file, index=False)
        
        print(f"\n--- SCRAPING COMPLETE ---")
        print(f"Total horses collected: {len(df)}")
        print(f"Data saved to: {output_file}")
    else:
        print("\nNo data was collected. Check the race links or site status.")
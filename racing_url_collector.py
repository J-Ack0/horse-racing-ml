#!/usr/bin/env python3
"""
Racing TV URL Collector
Collects race URLs from Racing TV for both Irish and UK racecourses
Saves Irish races and UK races into separate CSV files
"""

import time
import csv
import re
import argparse
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin

# Irish racecourses list
IRISH_RACECOURSES = [
    'punchestown', 'curragh', 'leopardstown', 'fairyhouse', 'naas',
    'gowran-park', 'cork', 'galway', 'killarney', 'tipperary',
    'limerick', 'navan', 'clonmel', 'wexford', 'tramore',
    'kilbeggan', 'sligo', 'bellewstown', 'roscommon', 'listowel',
    'laytown', 'down-royal', 'downpatrick', 'ballinrobe', 'thurles',
    'dundalk'
]

# UK racecourses list
UK_RACECOURSES = [
    'aintree', 'ascot', 'ayr', 'bangor-on-dee', 'bath', 'beverley',
    'brighton', 'carlisle', 'cartmel', 'catterick', 'chelmsford',
    'cheltenham', 'chepstow', 'chester', 'doncaster', 'epsom',
    'exeter', 'fakenham', 'ffos-las', 'fontwell', 'goodwood',
    'hamilton', 'haydock', 'hereford', 'hexham', 'huntingdon',
    'kelso', 'kempton', 'leicester', 'lingfield', 'ludlow',
    'market-rasen', 'musselburgh', 'newbury', 'newcastle', 'newmarket',
    'newton-abbot', 'nottingham', 'perth', 'plumpton', 'pontefract',
    'redcar', 'ripon', 'salisbury', 'sandown', 'sedgefield',
    'southwell', 'stratford', 'taunton', 'thirsk', 'towcester',
    'uttoxeter', 'warwick', 'wetherby', 'wincanton', 'windsor',
    'wolverhampton', 'worcester', 'yarmouth', 'york'
]

class RacingTVURLCollector:
    def __init__(self, headless=True, debug=False):
        """Initialize the collector with Selenium WebDriver"""
        self.debug = debug
        self.setup_driver(headless)
        self.base_url = "https://www.racingtv.com"
        self.irish_races = []
        self.uk_races = []
        
    def setup_driver(self, headless):
        """Configure and initialize Chrome WebDriver"""
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 15)
        
    def debug_print(self, message):
        """Print debug messages if debug mode is on"""
        if self.debug:
            print(f"[DEBUG] {message}")
    
    def generate_date_range(self, start_date, end_date):
        """Generate list of dates between start and end"""
        dates = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
            
        return dates
    
    def identify_track_country(self, track_name):
        """Identify if a track is Irish, UK, or neither"""
        track_lower = track_name.lower()
        
        # Check for Irish tracks
        for irish_track in IRISH_RACECOURSES:
            if irish_track in track_lower:
                return 'irish', irish_track
        
        # Check for UK tracks
        for uk_track in UK_RACECOURSES:
            if uk_track in track_lower:
                return 'uk', uk_track
        
        return 'other', None
    
    def get_race_urls_for_date(self, date):
        """Get all race URLs for a specific date"""
        print(f"\nCollecting race URLs for {date}")
        
        # Visit the results page for the date
        url = f"{self.base_url}/results/{date}"
        self.driver.get(url)
        time.sleep(5)  # Allow page to load
        
        # Parse the page
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        irish_count = 0
        uk_count = 0
        
        # Find the main divs with the specific CSS classes
        main_divs = soup.find_all('div', class_='css-175oi2r r-14lw9ot r-13awgt0 r-16y2uox')
        
        self.debug_print(f"Found {len(main_divs)} main divs with target CSS classes")
        
        for main_div in main_divs:
            # Find all a href elements within this div
            links = main_div.find_all('a', href=True)
            
            for link in links:
                href = link.get('href', '')
                link_text = link.text.strip()
                
                # Skip links that contain "replay" (case insensitive)
                if 'replay' in href.lower() or 'replay' in link_text.lower():
                    self.debug_print(f"Skipping replay link: {href}")
                    continue
                
                # Skip links that don't contain "Full Result" in link text
                if 'full result' not in link_text.lower():
                    self.debug_print(f"Skipping non-full result link: {link_text} - {href}")
                    continue
                
                # Get the full URL
                full_url = urljoin(self.base_url, href)
                
                # Check everything for track names - URL, link text, everything
                all_text = f"{href} {link_text}".lower()
                
                # Check if it's Irish
                is_irish = any(irish_track in all_text for irish_track in IRISH_RACECOURSES)
                
                # Check if it's UK  
                is_uk = any(uk_track in all_text for uk_track in UK_RACECOURSES)
                
                # If it's either Irish or UK, keep it
                if is_irish or is_uk:
                    # Figure out which track name we found
                    found_track = "Unknown"
                    if is_irish:
                        for irish_track in IRISH_RACECOURSES:
                            if irish_track in all_text:
                                found_track = irish_track.replace('-', ' ').title()
                                break
                    elif is_uk:
                        for uk_track in UK_RACECOURSES:
                            if uk_track in all_text:
                                found_track = uk_track.replace('-', ' ').title()
                                break
                    
                    # Look for race time
                    time_match = re.search(r'\b\d{1,2}:\d{2}\b', all_text)
                    race_time = time_match.group() if time_match else ''
                    
                    race_data = {
                        'date': date,
                        'track': found_track,
                        'time': race_time,
                        'url': full_url,
                        'link_text': link_text
                    }
                    
                    if is_irish:
                        self.irish_races.append(race_data)
                        irish_count += 1
                        self.debug_print(f"KEPT Irish race: {found_track} - {full_url}")
                    else:  # UK
                        self.uk_races.append(race_data)
                        uk_count += 1
                        self.debug_print(f"KEPT UK race: {found_track} - {full_url}")
                else:
                    self.debug_print(f"REJECTED (no track match): {link_text} - {href}")
        
        print(f"  Found {irish_count} Irish races and {uk_count} UK races")
    
    def collect_urls_date_range(self, start_date, end_date):
        """Main collection method for date range"""
        dates = self.generate_date_range(start_date, end_date)
        total_dates = len(dates)
        
        print(f"Collecting race URLs from {start_date} to {end_date}")
        print(f"Total days to process: {total_dates}")
        
        for i, date in enumerate(dates, 1):
            print(f"\nProgress: {i}/{total_dates} days")
            self.get_race_urls_for_date(date)
            
            # Small delay between dates to avoid overwhelming the server
            if i < total_dates:
                time.sleep(2)
    
    def save_to_csv(self, irish_filename='irish_races.csv', uk_filename='uk_races.csv'):
        """Save results to CSV files"""
        # Save Irish races
        if self.irish_races:
            irish_df = pd.DataFrame(self.irish_races)
            irish_df.to_csv(irish_filename, index=False)
            print(f"\nSaved {len(self.irish_races)} Irish race URLs to {irish_filename}")
        else:
            print("\nNo Irish races found")
        
        # Save UK races
        if self.uk_races:
            uk_df = pd.DataFrame(self.uk_races)
            uk_df.to_csv(uk_filename, index=False)
            print(f"Saved {len(self.uk_races)} UK race URLs to {uk_filename}")
        else:
            print("No UK races found")
    
    def close(self):
        """Close the WebDriver"""
        self.driver.quit()


def main():
    """Main execution function"""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Racing TV URL Collector for Irish and UK Racecourses')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    parser.add_argument('--irish-output', type=str, default='irish_races.csv', help='Output CSV filename for Irish races')
    parser.add_argument('--uk-output', type=str, default='uk_races.csv', help='Output CSV filename for UK races')
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        datetime.strptime(args.start_date, '%Y-%m-%d')
        datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        print("Error: Dates must be in YYYY-MM-DD format")
        return
    
    # Initialize collector
    collector = RacingTVURLCollector(headless=args.headless, debug=args.debug)
    
    try:
        # Collect URLs for the date range
        collector.collect_urls_date_range(args.start_date, args.end_date)
        
        # Save results
        collector.save_to_csv(args.irish_output, args.uk_output)
        
        # Print summary
        print(f"\n=== Collection Summary ===")
        print(f"Date range: {args.start_date} to {args.end_date}")
        print(f"Total Irish races: {len(collector.irish_races)}")
        print(f"Total UK races: {len(collector.uk_races)}")
        print(f"Total races: {len(collector.irish_races) + len(collector.uk_races)}")
        
    except Exception as e:
        print(f"Error during collection: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always close the driver
        collector.close()


if __name__ == "__main__":
    main()


"""
==============================
         Use case CLI
==============================
# Basic usage (collects URLs between dates)
python racing_url_collector.py --start-date 2020-01-01 --end-date 2025-06-11

# With custom output filenames
python racing_url_collector.py --start-date 2023-01-01 --end-date 2023-12-31 --irish-output irish_2023.csv --uk-output uk_2023.csv

# Run in headless mode (no browser window)
python racing_url_collector.py --start-date 2024-01-01 --end-date 2024-12-31 --headless

# Enable debug mode for troubleshooting
python racing_url_collector.py --start-date 2025-06-01 --end-date 2025-06-11 --debug
"""
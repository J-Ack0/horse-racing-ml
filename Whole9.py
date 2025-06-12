import argparse
import pandas as pd
import time
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse
import os
import logging
import json
import gc  # For garbage collection

class CSVLinkScraper:
    def __init__(self, headless=True, delay=2, batch_size=100):
        """
        Initialize the CSV Link Scraper
        
        Args:
            headless (bool): Run browser in headless mode (default: True)
            delay (int): Delay between requests in seconds (default: 2)
            batch_size (int): Number of URLs to process per batch (default: 100)
        """
        self.delay = delay
        self.batch_size = batch_size
        self.scraped_data = []  # Will now store individual horse records
        self.failed_urls = []
        self.progress_file = "scraping_progress.json"
        self.batch_output_prefix = "batch_horses"
        
        # Setup logging
        self.setup_logging()
        
        # Setup Chrome options
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        # Add memory optimization options
        chrome_options.add_argument("--memory-pressure-off")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        
        # Setup requests session for BS4 parsing
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'scraper_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def save_progress(self, batch_num, total_batches, processed_urls, failed_urls):
        """Save current progress to a JSON file"""
        progress = {
            'last_batch_completed': batch_num,
            'total_batches': total_batches,
            'total_urls_processed': processed_urls,
            'total_failed_urls': len(failed_urls),
            'failed_urls': failed_urls,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        
        self.logger.info(f"Progress saved: Batch {batch_num}/{total_batches} completed")
    
    def load_progress(self):
        """Load previous progress if it exists"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                self.logger.info(f"Previous progress found: Last completed batch {progress['last_batch_completed']}")
                return progress
            except Exception as e:
                self.logger.error(f"Error loading progress file: {e}")
        return None
    
    def load_csv(self, csv_file):
        """
        Load URLs from CSV file
        
        Args:
            csv_file (str): Path to CSV file containing URLs
            
        Returns:
            pd.DataFrame: DataFrame with URLs and metadata
        """
        try:
            df = pd.read_csv(csv_file)
            self.logger.info(f"Loaded {len(df)} rows from {csv_file}")
            
            # Validate that 'url' column exists
            if 'url' not in df.columns:
                raise ValueError("CSV must contain a 'url' column")
            
            # Remove any empty URLs
            df = df.dropna(subset=['url'])
            df = df[df['url'].str.strip() != '']
            
            self.logger.info(f"Found {len(df)} valid URLs to scrape")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            raise
    
    def get_initial_page_load(self, url):
        """
        Load page with Selenium to handle JavaScript/dynamic content
        
        Args:
            url (str): URL to load
            
        Returns:
            str: Page source after JavaScript execution, or None if failed
        """
        try:
            self.logger.info(f"Loading page with Selenium: {url}")
            self.driver.get(url)
            
            # Wait for page to load - wait a bit longer for racing pages
            time.sleep(3)  # Increased from 2 to 3 seconds for racing pages
            
            # Optional: Wait for specific racing elements to load
            try:
                self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".css-175oi2r.r-14lw9ot.r-13awgt0.r-16y2uox")))
                self.logger.info(f"Racing main block detected for: {url}")
            except TimeoutException:
                self.logger.warning(f"Racing main block not found within timeout for: {url}")
                # Continue anyway - page might still have content
            
            page_source = self.driver.page_source
            self.logger.info(f"Successfully loaded page: {url}")
            return page_source
            
        except TimeoutException:
            self.logger.error(f"Timeout loading page: {url}")
            return None
        except WebDriverException as e:
            self.logger.error(f"WebDriver error loading page {url}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error loading page {url}: {e}")
            return None
    
    def parse_jockey_and_claim(self, jockey_text):
        """
        Parse jockey name and extract claim if present
        
        Args:
            jockey_text (str): Raw jockey text (e.g., "J: P Hanlon(7lb)")
            
        Returns:
            tuple: (cleaned_jockey_name, claim_value)
        """
        if jockey_text == 'N/A' or not jockey_text:
            return 'N/A', 0
        
        # Remove "J:" prefix if present
        jockey_text = jockey_text.replace("J:", "").strip()
        
        # Extract claim using regex - looking for pattern like (7lb) or (5lb)
        claim_pattern = r'\((\d+)lb\)'
        match = re.search(claim_pattern, jockey_text)
        
        if match:
            claim_value = int(match.group(1))
            # Remove the claim from the jockey name
            jockey_name = re.sub(claim_pattern, '', jockey_text).strip()
        else:
            claim_value = 0  # Default to 0 if no claim
            jockey_name = jockey_text
        
        return jockey_name, claim_value
    
    def clean_trainer_name(self, trainer_text):
        """
        Clean trainer name by removing "T:" prefix
        
        Args:
            trainer_text (str): Raw trainer text (e.g., "T: John Smith")
            
        Returns:
            str: Cleaned trainer name
        """
        if trainer_text == 'N/A' or not trainer_text:
            return 'N/A'
        
        # Remove "T:" prefix if present
        return trainer_text.replace("T:", "").strip()
    
    def extract_going_and_distance(self, soup):
        """
        Extract going and distance from the 2nd h3 tag
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            
        Returns:
            dict: Dictionary with going and distance
        """
        result = {
            'going': 'N/A',
            'distance': 'N/A'
        }
        
        try:
            h3_tags = soup.find_all('h3')
            if len(h3_tags) >= 2:
                second_h3_text = h3_tags[1].get_text(strip=True)
                self.logger.info(f"Found 2nd h3 tag: {second_h3_text}")
                
                # Extract going - everything before the first integer
                going_match = re.match(r'^([^\d]+)', second_h3_text)
                if going_match:
                    result['going'] = going_match.group(1).strip()
                
                # Extract distance - pattern like "2m 2f 96y"
                # Can have any combination of Xm, Xf, Xy
                distance_pattern = r'(\d+m(?:\s+\d+f)?(?:\s+\d+y)?|\d+f(?:\s+\d+y)?|\d+y)'
                distance_match = re.search(distance_pattern, second_h3_text)
                if distance_match:
                    result['distance'] = distance_match.group(1)
                
                self.logger.info(f"Extracted - Going: {result['going']}, Distance: {result['distance']}")
            else:
                self.logger.warning("Less than 2 h3 tags found on page")
                
        except Exception as e:
            self.logger.error(f"Error extracting going and distance: {e}")
        
        return result
    
    def extract_race_info(self, soup):
        """
        Extract track name, date, and time from the page
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            
        Returns:
            dict: Dictionary with track_name, race_date, and race_time
        """
        race_info = {
            'track_name': 'N/A',
            'race_date': 'N/A',
            'race_time': 'N/A'
        }
        
        try:
            # Try to find race header information
            # This will depend on the actual page structure
            # Common patterns to look for:
            
            # Pattern 1: Look for header elements that might contain track/date/time
            header_elements = soup.select('h1, h2, h3, .race-header, .race-title')
            for header in header_elements:
                text = header.get_text(strip=True)
                # Look for date patterns (e.g., "12 Jun 2024", "12/06/2024")
                date_patterns = [
                    r'\d{1,2}\s+\w+\s+\d{4}',  # "12 Jun 2024"
                    r'\d{1,2}/\d{1,2}/\d{4}',   # "12/06/2024"
                    r'\d{4}-\d{2}-\d{2}'        # "2024-06-12"
                ]
                for pattern in date_patterns:
                    date_match = re.search(pattern, text)
                    if date_match:
                        race_info['race_date'] = date_match.group()
                        break
                
                # Look for time patterns (e.g., "14:30", "2:30 PM")
                time_patterns = [
                    r'\d{1,2}:\d{2}(?:\s*[AP]M)?',  # "14:30" or "2:30 PM"
                ]
                for pattern in time_patterns:
                    time_match = re.search(pattern, text, re.IGNORECASE)
                    if time_match:
                        race_info['race_time'] = time_match.group()
                        break
            
            # Pattern 2: Look for specific divs/spans with race info
            # Look for elements that might contain track name
            track_candidates = soup.select('[class*="track"], [class*="venue"], [class*="course"]')
            for elem in track_candidates:
                text = elem.get_text(strip=True)
                if text and len(text) > 2 and len(text) < 50:  # Reasonable track name length
                    race_info['track_name'] = text
                    break
            
            # Pattern 3: Try to extract from page title
            title_tag = soup.find('title')
            if title_tag and race_info['track_name'] == 'N/A':
                title_text = title_tag.get_text(strip=True)
                # Often racing pages have titles like "Ascot - 14:30 - 12 Jun 2024"
                # Try to extract track name (first part before dash)
                parts = title_text.split('-')
                if parts:
                    potential_track = parts[0].strip()
                    if len(potential_track) > 2 and len(potential_track) < 50:
                        race_info['track_name'] = potential_track
            
            self.logger.info(f"Extracted race info - Track: {race_info['track_name']}, Date: {race_info['race_date']}, Time: {race_info['race_time']}")
            
        except Exception as e:
            self.logger.error(f"Error extracting race info: {e}")
        
        return race_info
    
    def check_for_first_place(self, horse_records):
        """
        Check if we found the first place horse and try to find it if missing
        
        Args:
            horse_records (list): List of horse records
            
        Returns:
            bool: True if first place horse found
        """
        # Check if any horse has position 1, 1st, or 1ST
        first_place_found = any(
            record.get('race_position') and 
            any(pos in str(record.get('race_position', '')).upper() for pos in ['1ST', '1'])
            for record in horse_records
        )
        
        if first_place_found:
            self.logger.info("✓ First place horse found successfully")
        else:
            self.logger.warning("⚠ First place horse NOT found in the extracted data")
        
        return first_place_found
    
    def find_missing_first_place(self, soup, existing_records):
        """
        Try to find the first place horse using alternative methods
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            existing_records (list): Already extracted records
            
        Returns:
            dict or None: First place horse record if found
        """
        self.logger.info("Attempting to find missing first place horse...")
        
        try:
            # Method 1: Look for any element containing "1st" or "Winner"
            winner_indicators = ['1st', 'Winner', 'WINNER', 'First']
            
            for indicator in winner_indicators:
                # Search for elements containing the indicator
                elements = soup.find_all(text=re.compile(indicator, re.IGNORECASE))
                
                for element in elements:
                    parent = element.parent
                    # Try to find the horse name in nearby elements
                    if parent:
                        # Look for nearby <a> tags that might contain horse name
                        nearby_links = parent.find_all('a', limit=5)
                        if nearby_links:
                            # Assume first link is horse name
                            horse_name = nearby_links[0].get_text(strip=True)
                            
                            # Check if this horse is already in our records
                            if not any(r.get('horse_name') == horse_name for r in existing_records):
                                self.logger.info(f"Found potential winner via {indicator}: {horse_name}")
                                
                                # Create a basic record for the winner
                                winner_record = {
                                    'horse_name': horse_name,
                                    'race_position': '1st',
                                    'lost_by_length': '0',
                                    'jockey': 'N/A',
                                    'claims': 0,
                                    'trainer': 'N/A',
                                    'age': 'N/A',
                                    'weight': 'N/A',
                                    'rating': 'N/A',
                                    'found_via_alternative_method': True
                                }
                                
                                # Try to extract additional info from nearby elements
                                if len(nearby_links) > 1:
                                    raw_jockey = nearby_links[1].get_text(strip=True)
                                    jockey_name, claim_value = self.parse_jockey_and_claim(raw_jockey)
                                    winner_record['jockey'] = jockey_name
                                    winner_record['claims'] = claim_value
                                
                                if len(nearby_links) > 2:
                                    raw_trainer = nearby_links[2].get_text(strip=True)
                                    winner_record['trainer'] = self.clean_trainer_name(raw_trainer)
                                
                                return winner_record
            
            # Method 2: Look specifically for elements with winning styles
            # Check for elements with gold/yellow backgrounds or special styling
            styled_elements = soup.find_all(style=re.compile('background|gold|yellow|winner', re.IGNORECASE))
            
            for elem in styled_elements:
                links = elem.find_all('a')
                if links:
                    horse_name = links[0].get_text(strip=True)
                    if not any(r.get('horse_name') == horse_name for r in existing_records):
                        self.logger.info(f"Found potential winner via styling: {horse_name}")
                        
                        winner_record = {
                            'horse_name': horse_name,
                            'race_position': '1st',
                            'lost_by_length': '0',
                            'jockey': 'N/A',
                            'claims': 0,
                            'trainer': 'N/A',
                            'age': 'N/A',
                            'weight': 'N/A',
                            'rating': 'N/A',
                            'found_via_alternative_method': True
                        }
                        
                        if len(links) > 1:
                            raw_jockey = links[1].get_text(strip=True)
                            jockey_name, claim_value = self.parse_jockey_and_claim(raw_jockey)
                            winner_record['jockey'] = jockey_name
                            winner_record['claims'] = claim_value
                        
                        if len(links) > 2:
                            raw_trainer = links[2].get_text(strip=True)
                            winner_record['trainer'] = self.clean_trainer_name(raw_trainer)
                        
                        return winner_record
            
        except Exception as e:
            self.logger.error(f"Error in alternative first place search: {e}")
        
        return None
    
    def parse_with_bs4(self, html_content, url, row_data=None):
        """
        Parse HTML content with BeautifulSoup and return individual horse records
        
        Args:
            html_content (str): HTML content to parse
            url (str): Original URL for context
            row_data (dict): Original CSV row data
            
        Returns:
            list: List of individual horse records
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract race information (track, date, time)
            race_info = self.extract_race_info(soup)
            
            # Extract going and distance
            going_distance_info = self.extract_going_and_distance(soup)
            
            # Extract basic page information
            page_info = {
                'url': url,
                'title': self.extract_title(soup),
                'scraped_at': datetime.now().isoformat(),
                'status': 'success',
                'track_name': race_info['track_name'],
                'race_date': race_info['race_date'],
                'race_time': race_info['race_time'],
                'going': going_distance_info['going'],
                'distance': going_distance_info['distance']
            }
            
            # Add original CSV data if provided
            if row_data:
                for key, value in row_data.items():
                    if key != 'url':  # Don't override the URL
                        page_info[f'original_{key}'] = value
            
            # Extract individual horse records
            horse_records = self.extract_horse_records(soup, url, page_info)
            
            # Check if we found the first place horse
            first_place_found = self.check_for_first_place(horse_records)
            
            # If first place not found, try alternative methods
            if not first_place_found and horse_records:
                self.logger.warning(f"First place horse missing for {url}, attempting alternative search...")
                winner_record = self.find_missing_first_place(soup, horse_records)
                
                if winner_record:
                    # Add page info to winner record
                    winner_record.update(page_info)
                    # Insert at the beginning of the list
                    horse_records.insert(0, winner_record)
                    self.logger.info("✓ Successfully found and added first place horse via alternative method")
                else:
                    self.logger.error(f"❌ Could not find first place horse for {url} despite alternative search")
            
            # Add first_place_found flag to all records
            for record in horse_records:
                record['first_place_found'] = self.check_for_first_place(horse_records)
            
            self.logger.info(f"Successfully parsed {len(horse_records)} horse records from: {url}")
            return horse_records
            
        except Exception as e:
            self.logger.error(f"Error parsing HTML for {url}: {e}")
            # Return a single error record instead of individual horses
            error_record = {
                'url': url,
                'title': None,
                'scraped_at': datetime.now().isoformat(),
                'status': 'parse_error',
                'error': str(e),
                'horse_name': 'N/A',
                'race_position': 'N/A',
                'lost_by_length': 'N/A',
                'jockey': 'N/A',
                'claims': 0,
                'trainer': 'N/A',
                'age': 'N/A',
                'weight': 'N/A',
                'rating': 'N/A',
                'track_name': 'N/A',
                'race_date': 'N/A',
                'race_time': 'N/A',
                'going': 'N/A',
                'distance': 'N/A',
                'first_place_found': False
            }
            
            # Add original CSV data if provided
            if row_data:
                for key, value in row_data.items():
                    if key != 'url':
                        error_record[f'original_{key}'] = value
            
            return [error_record]
    
    def extract_title(self, soup):
        """Extract page title"""
        title_tag = soup.find('title')
        return title_tag.get_text(strip=True) if title_tag else None
    
    def extract_horse_records(self, soup, url, page_info):
        """
        Extract individual horse records from the page
        
        Args:
            soup (BeautifulSoup): Parsed HTML
            url (str): Original URL
            page_info (dict): Basic page information
            
        Returns:
            list: List of individual horse records
        """
        horse_records = []
        
        try:
            # Racing-specific extraction logic
            self.logger.info(f"Extracting racing data from: {url}")
            
            # Grab only the first main block
            main_blocks = soup.select(".css-175oi2r.r-14lw9ot.r-13awgt0.r-16y2uox")
            if not main_blocks:
                self.logger.warning(f"No main blocks found for {url}")
                # Return empty list if no racing data found
                return []
            
            main_block = main_blocks[0]
            self.logger.info(f"Processing main block #1 for {url}")
            
            # More flexible approach to finding horse sub-blocks
            # Look for divs with the exact classes, regardless of additional attributes like style
            sub_blocks = []
            
            # Method 1: Direct CSS selector (original)
            sub_blocks_direct = main_block.select(".css-175oi2r.r-1snhixr.r-qklmqi.r-n7gxbd.r-glunga.r-5iw40x")
            
            # Method 2: Find all divs and filter by class
            all_divs = main_block.find_all("div", recursive=True)
            required_classes = ["css-175oi2r", "r-1snhixr", "r-qklmqi", "r-n7gxbd", "r-glunga", "r-5iw40x"]
            
            for div in all_divs:
                div_classes = div.get("class", [])
                # Check if all required classes are present
                if all(cls in div_classes for cls in required_classes):
                    # Avoid duplicates
                    if div not in sub_blocks_direct:
                        sub_blocks.append(div)
            
            # NEW: Method 3 - Look for the 2nd child which might contain first place
            # Based on the selector path provided
            if len(main_block.find_all("div", recursive=False)) > 1:
                second_child = main_block.find_all("div", recursive=False)[1]
                self.logger.info("Checking 2nd child of main block for first place horse...")
                
                # Look for horse entries with different class structure
                alternative_horse_blocks = second_child.select(".css-175oi2r.r-18u37iz")
                for alt_block in alternative_horse_blocks:
                    # Check if this looks like a horse entry (has links inside)
                    if alt_block.find_all("a"):
                        self.logger.info(f"Found alternative horse block structure with {len(alt_block.find_all('a'))} links")
                        sub_blocks.insert(0, alt_block)  # Insert at beginning as it might be first place
            
            # NEW: Method 4 - Look for any div that contains horse-like structure
            # Sometimes the classes might be completely different
            potential_horse_blocks = main_block.find_all("div", recursive=True)
            for block in potential_horse_blocks:
                # Check if it has the structure of a horse entry:
                # - Has links (for horse name, jockey, trainer)
                # - Has position/length info
                if (block.find_all("a") and 
                    block not in sub_blocks and
                    len(block.find_all("a")) >= 1):  # At least horse name
                    
                    # Check if it has position information
                    position_divs = block.select(".css-175oi2r.r-41uqdh")
                    if position_divs:
                        # Check position text
                        position_children = position_divs[0].select(".css-146c3p1")
                        if position_children:
                            pos_text = position_children[0].get_text(strip=True) if position_children else ""
                            if any(x in pos_text.upper() for x in ['1ST', '1']):
                                self.logger.info(f"Found likely first place horse via structure analysis: {block.find_all('a')[0].get_text(strip=True) if block.find_all('a') else 'Unknown'}")
                                sub_blocks.insert(0, block)  # Insert at beginning
                                break
            
            # Combine all methods, removing duplicates
            seen_divs = set()
            final_sub_blocks = []
            
            # Process all found blocks
            for block in sub_blocks:
                block_id = id(block)
                if block_id not in seen_divs:
                    seen_divs.add(block_id)
                    final_sub_blocks.append(block)
            
            sub_blocks = final_sub_blocks
            
            # Check specifically for first place horse with background styling
            first_place_candidates = main_block.find_all("div", 
                style=lambda value: value and "background-color" in value if value else False)
            
            # Add first place candidates to the beginning if not already included
            for candidate in first_place_candidates:
                if candidate not in sub_blocks and candidate.find_all("a"):
                    sub_blocks.insert(0, candidate)  # Insert at beginning
                    self.logger.info(f"Found potential first place horse with background styling")
            
            self.logger.info(f"Found {len(sub_blocks)} horse entries for {url} using multiple detection methods")
            
            # Process each horse as a separate record
            for idx, sub_block in enumerate(sub_blocks, 1):
                self.logger.info(f"Processing sub block #{idx} for {url}")
                
                # Check if this might be the winner (has background-color style)
                style_attr = sub_block.get('style', '')
                is_potential_winner = 'background-color' in style_attr
                
                # Create individual horse record with page info
                horse_record = page_info.copy()
                horse_record.update({
                    'horse_position_in_data': idx,  # Position in scraped data
                    'horse_name': 'N/A',
                    'jockey': 'N/A',
                    'claims': 0,  # Default to 0 for no claim
                    'trainer': 'N/A',
                    'race_position': 'N/A',        # Actual finishing position
                    'lost_by_length': 'N/A',      # Distance behind winner
                    'age': 'N/A',
                    'weight': 'N/A', 
                    'rating': 'N/A',
                    'jockey_trainer_raw_texts': [],
                    'has_background_style': is_potential_winner  # Track if this might be the winner
                })
                
                try:
                    # Horse name, Jockey & Trainer: from specific div with <a> tags
                    horse_jockey_trainer_div = sub_block.select_one(".css-175oi2r.r-4ajl7u")
                    a_tags = []  # Initialize for metadata tracking
                    
                    if horse_jockey_trainer_div:
                        # Get all <a> tags within this div
                        a_tags = horse_jockey_trainer_div.select("a")
                        
                        # First <a> tag = horse name
                        if len(a_tags) > 0:
                            horse_record['horse_name'] = a_tags[0].get_text(strip=True)
                        
                        # Second <a> tag = jockey (with potential claim)
                        if len(a_tags) > 1:
                            raw_jockey = a_tags[1].get_text(strip=True)
                            jockey_name, claim_value = self.parse_jockey_and_claim(raw_jockey)
                            horse_record['jockey'] = jockey_name
                            horse_record['claims'] = claim_value
                        
                        # Third <a> tag = trainer
                        if len(a_tags) > 2:
                            raw_trainer = a_tags[2].get_text(strip=True)
                            horse_record['trainer'] = self.clean_trainer_name(raw_trainer)
                        
                        # Store raw texts for debugging
                        horse_record['jockey_trainer_raw_texts'] = [a.get_text(strip=True) for a in a_tags]
                        
                        self.logger.info(f"Found {len(a_tags)} <a> tags - Horse: {horse_record['horse_name']}, Jockey: {horse_record['jockey']} (claim: {horse_record.get('claims', 0)}), Trainer: {horse_record['trainer']}")
                    else:
                        self.logger.warning(f"Horse/jockey/trainer div not found in sub block {idx}")
                        horse_record['jockey_trainer_raw_texts'] = []
                    
                    # Age, Weight, Rating: 1st, 2nd, 3rd elements of this class
                    awr_elements = sub_block.select(".css-175oi2r.r-13awgt0")
                    horse_record['age'] = awr_elements[0].get_text(strip=True) if len(awr_elements) > 0 else "N/A"
                    horse_record['weight'] = awr_elements[1].get_text(strip=True) if len(awr_elements) > 1 else "N/A"
                    horse_record['rating'] = awr_elements[2].get_text(strip=True) if len(awr_elements) > 2 else "N/A"
                    
                    # Position and Lost by Length: from main div with specific children
                    position_length_div = sub_block.select_one(".css-175oi2r.r-41uqdh")
                    if position_length_div:
                        # Find child elements with css-146c3p1 class
                        position_length_children = position_length_div.select(".css-146c3p1")
                        
                        # Child 1: Race position
                        if len(position_length_children) > 0:
                            horse_record['race_position'] = position_length_children[0].get_text(strip=True)
                        
                        # Child 2: Lost by length
                        if len(position_length_children) > 1:
                            horse_record['lost_by_length'] = position_length_children[1].get_text(strip=True)
                        
                        self.logger.info(f"Position/Length - Found {len(position_length_children)} children: Position: {horse_record['race_position']}, Lost by: {horse_record['lost_by_length']}")
                    else:
                        self.logger.warning(f"Position/length div not found for horse {horse_record['horse_name']}")
                    
                    # If this horse has background styling and position is "1" or "1st", it's likely the winner
                    if is_potential_winner and horse_record['race_position'] in ['1', '1st', '1ST']:
                        self.logger.info(f"Confirmed winner: {horse_record['horse_name']}")
                    
                    # Additional metadata for debugging
                    horse_record['awr_elements_count'] = len(awr_elements)
                    horse_record['horse_jt_a_tags_count'] = len(a_tags)
                    horse_record['jockey_trainer_raw'] = " | ".join(horse_record['jockey_trainer_raw_texts'])
                    
                    self.logger.info(f"Extracted: {horse_record['horse_name']} - Position: {horse_record['race_position']}, Lost by: {horse_record['lost_by_length']}, Jockey: {horse_record['jockey']} (claim: {horse_record.get('claims', 0)}), Trainer: {horse_record['trainer']}, Age: {horse_record['age']}, Weight: {horse_record['weight']}, Rating: {horse_record['rating']}")
                    
                except Exception as e:
                    self.logger.error(f"Error extracting data from sub block {idx} for {url}: {e}")
                    horse_record['extraction_error'] = str(e)
                
                # Add this horse record to the list
                horse_records.append(horse_record)
            
            # Add race summary info to each horse record
            total_horses = len(horse_records)
            valid_horses = len([r for r in horse_records if r['horse_name'] != 'N/A'])
            
            for record in horse_records:
                record['total_horses_in_race'] = total_horses
                record['valid_horses_in_race'] = valid_horses
                record['main_blocks_found'] = len(main_blocks)
                record['sub_blocks_count'] = len(sub_blocks)
            
            self.logger.info(f"Successfully extracted {valid_horses}/{total_horses} horse records from {url}")
            
        except Exception as e:
            self.logger.error(f"Error in racing data extraction for {url}: {e}")
            # Return a single error record
            error_record = page_info.copy()
            error_record.update({
                'extraction_error': str(e),
                'horse_name': 'N/A',
                'race_position': 'N/A',
                'lost_by_length': 'N/A',
                'jockey': 'N/A',
                'claims': 0,
                'trainer': 'N/A',
                'age': 'N/A',
                'weight': 'N/A',
                'rating': 'N/A',
                'total_horses_in_race': 0,
                'valid_horses_in_race': 0,
                'first_place_found': False
            })
            horse_records = [error_record]
        
        return horse_records
    
    def scrape_url(self, url, row_data=None):
        """
        Scrape a single URL and return individual horse records
        
        Args:
            url (str): URL to scrape
            row_data (dict): Original row data from CSV
            
        Returns:
            list: List of individual horse records
        """
        try:
            # Step 1: Load page with Selenium
            html_content = self.get_initial_page_load(url)
            
            if html_content is None:
                error_record = {
                    'url': url,
                    'status': 'load_error',
                    'scraped_at': datetime.now().isoformat(),
                    'error': 'Failed to load page with Selenium',
                    'horse_name': 'N/A',
                    'race_position': 'N/A',
                    'lost_by_length': 'N/A',
                    'jockey': 'N/A',
                    'claims': 0,
                    'trainer': 'N/A',
                    'age': 'N/A',
                    'weight': 'N/A',
                    'rating': 'N/A',
                    'track_name': 'N/A',
                    'race_date': 'N/A',
                    'race_time': 'N/A',
                    'going': 'N/A',
                    'distance': 'N/A',
                    'first_place_found': False
                }
                
                # Add original CSV data if provided
                if row_data:
                    for key, value in row_data.items():
                        if key != 'url':
                            error_record[f'original_{key}'] = value
                
                return [error_record]
            
            # Step 2: Parse with BeautifulSoup and get individual horse records
            horse_records = self.parse_with_bs4(html_content, url, row_data)
            
            return horse_records
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            error_record = {
                'url': url,
                'status': 'error',
                'scraped_at': datetime.now().isoformat(),
                'error': str(e),
                'horse_name': 'N/A',
                'race_position': 'N/A',
                'lost_by_length': 'N/A',
                'jockey': 'N/A',
                'claims': 0,
                'trainer': 'N/A',
                'age': 'N/A',
                'weight': 'N/A',
                'rating': 'N/A',
                'track_name': 'N/A',
                'race_date': 'N/A',
                'race_time': 'N/A',
                'going': 'N/A',
                'distance': 'N/A',
                'first_place_found': False
            }
            
            # Add original CSV data if provided
            if row_data:
                for key, value in row_data.items():
                    if key != 'url':
                        error_record[f'original_{key}'] = value
            
            return [error_record]
    
    def scrape_batch(self, batch_df, batch_num, total_batches):
        """
        Scrape a batch of URLs
        
        Args:
            batch_df (pd.DataFrame): DataFrame containing batch of URLs
            batch_num (int): Current batch number
            total_batches (int): Total number of batches
            
        Returns:
            list: List of horse records from the batch
        """
        batch_size = len(batch_df)
        batch_data = []
        batch_failed_urls = []
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"STARTING BATCH {batch_num}/{total_batches} ({batch_size} URLs)")
        self.logger.info(f"{'='*60}\n")
        
        for index, row in batch_df.iterrows():
            url = row['url']
            relative_index = index % batch_size + 1
            
            self.logger.info(f"Batch {batch_num} - URL {relative_index}/{batch_size}: {url}")
            
            # Convert row to dictionary for context
            row_data = row.to_dict()
            
            # Scrape the URL and get individual horse records
            horse_records = self.scrape_url(url, row_data)
            
            # Add batch information to each record
            for record in horse_records:
                record['batch_number'] = batch_num
                record['url_index_in_batch'] = relative_index
            
            # Add all horse records to batch data
            batch_data.extend(horse_records)
            
            # Track failed URLs
            if any(record.get('status') != 'success' for record in horse_records):
                batch_failed_urls.append(url)
            
            # Delay between requests
            if relative_index < batch_size:  # Don't delay after last URL in batch
                self.logger.info(f"Waiting {self.delay} seconds before next request...")
                time.sleep(self.delay)
        
        # Update overall failed URLs list
        self.failed_urls.extend(batch_failed_urls)
        
        successful_urls = batch_size - len(batch_failed_urls)
        total_horses = len(batch_data)
        
        self.logger.info(f"\nBatch {batch_num} complete - URLs: Success: {successful_urls}, Failed: {len(batch_failed_urls)}, Horses extracted: {total_horses}")
        
        return batch_data
    
    def save_batch_results(self, batch_data, batch_num):
        """
        Save batch results to CSV
        
        Args:
            batch_data (list): Horse records from the batch
            batch_num (int): Batch number
            
        Returns:
            str: Path to saved batch file
        """
        if not batch_data:
            self.logger.warning(f"No data to save for batch {batch_num}")
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_file = f"{self.batch_output_prefix}_batch_{batch_num:04d}_{timestamp}.csv"
        
        try:
            # Define the specific columns to save
            columns_to_save = [
                'track_name',
                'race_date',
                'race_time',
                'going',
                'distance',
                'horse_name',
                'jockey',
                'claims',
                'trainer',
                'race_position',
                'lost_by_length',
                'age',
                'weight',
                'rating',
                'first_place_found',
                'batch_number',
                'url_index_in_batch'
            ]
            
            df = pd.DataFrame(batch_data)
            
            # Select only the specified columns and save
            df_filtered = df[columns_to_save]
            df_filtered.to_csv(batch_file, index=False)
            
            self.logger.info(f"Batch {batch_num} results saved to {batch_file}")
            
            return batch_file
            
        except Exception as e:
            self.logger.error(f"Error saving batch {batch_num} results: {e}")
            return None
    
    def combine_batch_files(self, batch_files, final_output_file):
        """
        Combine all batch files into a single final output file
        
        Args:
            batch_files (list): List of batch file paths
            final_output_file (str): Final output file path
        """
        try:
            # Read all batch files and combine
            all_data = []
            for batch_file in batch_files:
                if os.path.exists(batch_file):
                    batch_df = pd.read_csv(batch_file)
                    all_data.append(batch_df)
                    self.logger.info(f"Loaded {len(batch_df)} records from {batch_file}")
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # Remove batch-specific columns for final output
                columns_to_remove = ['batch_number', 'url_index_in_batch']
                final_columns = [col for col in combined_df.columns if col not in columns_to_remove]
                
                combined_df[final_columns].to_csv(final_output_file, index=False)
                self.logger.info(f"Combined {len(combined_df)} total records into {final_output_file}")
                
                # Optionally delete individual batch files
                # for batch_file in batch_files:
                #     if os.path.exists(batch_file):
                #         os.remove(batch_file)
                #         self.logger.info(f"Deleted batch file: {batch_file}")
                
                return len(combined_df)
            else:
                self.logger.warning("No batch files found to combine")
                return 0
                
        except Exception as e:
            self.logger.error(f"Error combining batch files: {e}")
            return 0
    
    def scrape_all_urls_batch(self, csv_df, resume=False):
        """
        Scrape all URLs from CSV in batches
        
        Args:
            csv_df (pd.DataFrame): DataFrame containing URLs and metadata
            resume (bool): Whether to resume from previous progress
            
        Returns:
            list: List of all batch output files
        """
        total_urls = len(csv_df)
        total_batches = (total_urls + self.batch_size - 1) // self.batch_size  # Ceiling division
        
        self.logger.info(f"Total URLs: {total_urls}, Batch size: {self.batch_size}, Total batches: {total_batches}")
        
        # Check for previous progress
        start_batch = 1
        if resume:
            progress = self.load_progress()
            if progress:
                start_batch = progress['last_batch_completed'] + 1
                self.failed_urls = progress.get('failed_urls', [])
                self.logger.info(f"Resuming from batch {start_batch}")
        
        batch_files = []
        total_processed_urls = (start_batch - 1) * self.batch_size
        
        # Process each batch
        for batch_num in range(start_batch, total_batches + 1):
            # Calculate batch indices
            start_idx = (batch_num - 1) * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_urls)
            
            # Get batch DataFrame
            batch_df = csv_df.iloc[start_idx:end_idx]
            
            # Scrape the batch
            batch_data = self.scrape_batch(batch_df, batch_num, total_batches)
            
            # Save batch results
            batch_file = self.save_batch_results(batch_data, batch_num)
            if batch_file:
                batch_files.append(batch_file)
            
            # Update progress
            total_processed_urls += len(batch_df)
            self.save_progress(batch_num, total_batches, total_processed_urls, self.failed_urls)
            
            # Clear memory after each batch
            self.scraped_data = []  # Clear main data
            batch_data = None
            gc.collect()  # Force garbage collection
            
            self.logger.info(f"Memory cleared after batch {batch_num}")
            
            # Optional: Restart browser after every N batches to prevent memory issues
            if batch_num % 10 == 0 and batch_num < total_batches:
                self.logger.info("Restarting browser to clear memory...")
                self.driver.quit()
                time.sleep(5)  # Give it time to fully close
                
                # Reinitialize driver
                chrome_options = Options()
                if self.driver.capabilities.get('goog:chromeOptions', {}).get('args'):
                    for arg in self.driver.capabilities['goog:chromeOptions']['args']:
                        chrome_options.add_argument(arg)
                
                self.driver = webdriver.Chrome(options=chrome_options)
                self.wait = WebDriverWait(self.driver, 10)
                self.logger.info("Browser restarted successfully")
        
        return batch_files
    
    def get_summary(self):
        """Get scraping summary statistics"""
        # This method is now less useful for batch processing
        # but kept for compatibility
        return {
            'total_failed_urls': len(self.failed_urls),
            'failed_urls': self.failed_urls
        }
    
    def close(self):
        """Close browser and cleanup"""
        if self.driver:
            self.driver.quit()
        if self.session:
            self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Scrape URLs from a CSV file using Selenium + BeautifulSoup with Batch Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --csv urls.csv
  %(prog)s --csv urls.csv --batch-size 50 --output final_horses.csv
  %(prog)s --csv urls.csv --resume --headless
  %(prog)s --csv urls.csv --delay 5 --verbose
        """
    )
    
    parser.add_argument(
        "--csv", "-c",
        required=True,
        help="Path to CSV file containing URLs (must have 'url' column)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Final output CSV filename (default: auto-generated with timestamp)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100,
        help="Number of URLs to process per batch (default: 100)"
    )
    
    parser.add_argument(
        "--delay", "-d",
        type=int,
        default=2,
        help="Delay between requests in seconds (default: 2)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (default: False)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous progress if available"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def main():
    """Main function with CLI functionality"""
    args = parse_arguments()
    
    # Validate input file
    if not os.path.exists(args.csv):
        print(f"Error: CSV file '{args.csv}' not found")
        return
    
    try:
        with CSVLinkScraper(headless=args.headless, delay=args.delay, batch_size=args.batch_size) as scraper:
            # Load CSV
            csv_df = scraper.load_csv(args.csv)
            
            # Scrape all URLs in batches
            batch_files = scraper.scrape_all_urls_batch(csv_df, resume=args.resume)
            
            # Combine all batch files into final output
            if args.output is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                final_output = f"scraped_horses_combined_{timestamp}.csv"
            else:
                final_output = args.output
            
            total_records = scraper.combine_batch_files(batch_files, final_output)
            
            # Save failed URLs
            if scraper.failed_urls:
                failed_file = final_output.replace('.csv', '_failed_urls.txt')
                with open(failed_file, 'w') as f:
                    for url in scraper.failed_urls:
                        f.write(f"{url}\n")
                print(f"\nFailed URLs saved to {failed_file}")
            
            # Show summary
            print("\n" + "="*50)
            print("SCRAPING COMPLETE:")
            print("="*50)
            print(f"Total records combined: {total_records}")
            print(f"Failed URLs: {len(scraper.failed_urls)}")
            print(f"Final output saved to: {final_output}")
            print(f"Batch files created: {len(batch_files)}")
            
            # Clean up progress file
            if os.path.exists(scraper.progress_file):
                os.remove(scraper.progress_file)
                print(f"\nProgress file cleaned up")
    
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        print("Progress has been saved. Use --resume flag to continue from where you left off.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

    #python3 Whole9.py --csv urls.csv --batch-size 50 --output Part2.csv --headless
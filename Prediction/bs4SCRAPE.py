import requests
from bs4 import BeautifulSoup
import time
import csv
import re

# === CONFIG ===
START_URL = "https://www.racingpost.com/"  # Replace with the real URL
CSV_FILENAME = "race_data.csv"
# Add headers to mimic a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# === FETCH HTML CONTENT ===
def fetch_html(url):
    """Fetch HTML content from a URL using requests instead of Selenium"""
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text
    except Exception as e:
        print(f"[ERROR] Failed to fetch URL {url}: {e}")
        return None

# Helper function to convert relative URLs to absolute URLs
def make_absolute_url(url, base_url):
    """Convert relative URLs to absolute URLs"""
    if url and not url.startswith('http'):
        return base_url.rstrip('/') + '/' + url.lstrip('/')
    return url

# === STAGE 1: Extract race meeting links ===
def get_links_from_meeting_boxes(html):
    """Extract race links from the meeting boxes using BeautifulSoup"""
    if not html:
        return []
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        # Look for meeting boxes by class name
        boxes = soup.find_all(class_="sc-36f6c09-8")
        all_links = []

        for box in boxes:
            links = box.find_all("a")
            hrefs = [link.get("href") for link in links if link.get("href")]
            all_links.extend(hrefs)

        # If no links found with the specific class, try a more general approach
        if not all_links:
            # Find all links that might be race links
            links = soup.find_all("a", href=re.compile(r"/racecards/"))
            all_links = [link.get("href") for link in links if link.get("href")]
        
        # Convert relative URLs to absolute URLs
        all_links = [make_absolute_url(link, START_URL) for link in all_links]
        
        return all_links

    except Exception as e:
        print(f"[ERROR] Failed to extract links: {e}")
        return []

# === STAGE 2: Scrape race + runner data ===
def scrape_race(html, url):
    """Scrape race and runner data using BeautifulSoup instead of Selenium"""
    if not html:
        return {"url": url, "error": "Failed to fetch page"}
    
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # Extract race time
        race_time_element = soup.find(class_="RC-courseHeader__time")
        race_time = race_time_element.text.strip() if race_time_element else "Unknown"

        # Extract header text
        header_element = soup.find("h1")
        header_text = header_element.text.strip() if header_element else "Unknown"

        # Extract race date
        race_date_element = soup.find(class_="RC-courseHeader__dateHolder")
        race_date = race_date_element.text.strip() if race_date_element else "Unknown"
        
        # Try to extract race details from h1 for better classification
        class_pattern = re.compile(r'Class (\d+)')
        prize_pattern = re.compile(r'(\d+K)')
        distance_pattern = re.compile(r'(\d+[fm]\d*[fy]?)')
        
        class_match = class_pattern.search(header_text) if header_text else None
        prize_match = prize_pattern.search(header_text) if header_text else None
        distance_match = distance_pattern.search(header_text) if header_text else None
        
        race_class_from_header = class_match.group(1) if class_match else None
        race_prize_from_header = prize_match.group(1) if prize_match else None
        race_distance_from_header = distance_match.group(0) if distance_match else None
        
        # Extract going with better fallback options
        going = None
        info_rows = soup.find_all(class_="RC-headerBox__infoRow")
        if len(info_rows) >= 3:
            going_element = info_rows[2].find(class_="RC-headerBox__infoRow__content")
            going = going_element.text.strip() if going_element else None
        
        if not going:
            # Fallback to any RC-headerBox__infoRow__content that contains going information
            going_elements = soup.find_all(class_="RC-headerBox__infoRow__content")
            for element in going_elements:
                text = element.text.strip() if element else ""
                if text and any(keyword in text.lower() for keyword in ["good", "firm", "soft", "heavy", "standard"]):
                    going = text
                    break
        
        if not going:
            # Last resort: try to find any RC-headerBox__infoRow__content
            going_element = soup.find(class_="RC-headerBox__infoRow__content")
            going = going_element.text.strip() if going_element else "Unknown"
        
        # Extract distance with better class selectors and regex parsing
        distance = None
        course_details = soup.find(class_="RC-cardHeader__courseDetails")
        if course_details:
            course_text = course_details.text.strip()
            
            # Use regex to extract the distance pattern from the text
            distance_matches = distance_pattern.findall(course_text)
            
            if distance_matches:
                # Use the first match as the distance
                distance = distance_matches[0]
            else:
                # If no match found, use the full text for further processing
                distance = course_text
        
        if not distance:
            # Try alternative with desktop version of the class
            course_details = soup.find(class_="RC-cardHeader__courseDetails--desktop")
            if course_details:
                distance = course_details.text.strip()
                
                # Try to extract distance with regex if we got text
                if distance:
                    distance_matches = distance_pattern.findall(distance)
                    if distance_matches:
                        distance = distance_matches[0]
        
        if not distance:
            # Fall back to the original distance selectors
            distance_element = soup.find("strong", class_="RC-cardHeader__distance", attrs={"data-test-selector": "RC-header__raceDistanceRound"})
            if distance_element:
                distance = distance_element.text.strip()
            else:
                # Try generic class name as last resort
                distance_element = soup.find(class_="RC-cardHeader__distance")
                distance = distance_element.text.strip() if distance_element else "Unknown"
        
        # Extract race class with fallback options
        race_class = None
        race_class_element = soup.find("span", attrs={"data-test-selector": "RC-header__raceClass"})
        if race_class_element:
            race_class = race_class_element.text.strip()
        else:
            # Try to find any element that might contain class information
            class_elements = soup.find_all(lambda tag: tag.name and 
                               (tag.get('class') and any(cls in tag.get('class') for cls in ['class', 'Class'])) or 
                               (tag.has_attr('data-test-selector') and any(cls in tag['data-test-selector'] for cls in ['class', 'Class'])))
            
            for element in class_elements:
                if element.text and "class" in element.text.lower():
                    race_class = element.text.strip()
                    break
            
            if not race_class:
                race_class = "Unknown"

        runner_data = []
        # Updated to target the runner card wrappers based on the HTML structure
        runner_cards = soup.find_all(class_="RC-runnerCardWrapper")

        for card in runner_cards:
            try:
                # Extract horse name from the link with the specific class
                name_element = card.find("a", class_="RC-runnerName")
                name = name_element.text.strip() if name_element else "Unknown"
                
                # Extract horse URL for additional information
                horse_url = name_element.get("href") if name_element else None
                if horse_url:
                    horse_url = make_absolute_url(horse_url, START_URL)
                
                # Extract TS value using data-test-selector
                ts_element = card.find("span", attrs={"data-test-selector": "RC-cardPage-runnerTs"})
                ts = ts_element.text.strip() if ts_element else None
                
                # Extract RPR value using data-test-selector
                rpr_element = card.find("span", attrs={"data-test-selector": "RC-cardPage-runnerRpr"})
                rpr = rpr_element.text.strip() if rpr_element else None
                
                # Get price if available (may be null)
                price_element = card.find(class_="RC-runnerRowPriceWrapper")
                price = price_element.text.strip() if price_element else None
                
                # Extract horse age
                age = None
                age_element = card.find("span", class_="RC-runnerAge", attrs={"data-test-selector": "RC-cardPage-runnerAge"})
                if age_element:
                    age = age_element.text.strip()
                else:
                    # Try a fallback method
                    age_elements = card.find_all(lambda tag: tag.name and 
                                       (tag.get('class') and any(cls in tag.get('class') for cls in ['Age', 'age'])) or 
                                       (tag.has_attr('data-test-selector') and any(cls in tag['data-test-selector'] for cls in ['Age', 'age'])))
                    
                    for element in age_elements:
                        if element.text and element.text.strip().isdigit():
                            age = element.text.strip()
                            break
                
                # Get jockey information using the correct selectors
                jockey_name = None
                jockey_url = None
                
                # Use the exact data-test-selector shown in your HTML example
                jockey_element = card.find("a", attrs={"data-test-selector": "RC-cardPage-runnerJockey-name"})
                if jockey_element:
                    jockey_name = jockey_element.text.strip()
                    jockey_url = jockey_element.get("href")
                    if jockey_url:
                        jockey_url = make_absolute_url(jockey_url, START_URL)
                else:
                    # Try a fallback method
                    jockey_elements = card.find_all("a", class_="RC-runnerInfo__name")
                    if jockey_elements:
                        jockey_name = jockey_elements[0].text.strip()
                        jockey_url = jockey_elements[0].get("href")
                        if jockey_url:
                            jockey_url = make_absolute_url(jockey_url, START_URL)
                
                # Get trainer information using the correct selectors
                trainer_name = None
                trainer_url = None
                
                # Use data-test-selector for trainer
                trainer_element = card.find("a", attrs={"data-test-selector": "RC-cardPage-runnerTrainer-name"})
                if trainer_element:
                    trainer_name = trainer_element.text.strip()
                    trainer_url = trainer_element.get("href")
                    if trainer_url:
                        trainer_url = make_absolute_url(trainer_url, START_URL)
                else:
                    # Try a fallback method
                    trainer_elements = card.find_all("a", class_="RC-runnerInfo__name")
                    if len(trainer_elements) > 1:
                        trainer_name = trainer_elements[1].text.strip()
                        trainer_url = trainer_elements[1].get("href")
                        if trainer_url:
                            trainer_url = make_absolute_url(trainer_url, START_URL)

                # Weight (carried)
                weight = "NA"
                wgt_elem = card.find("span", class_="RC-runnerWgt__carried")
                if wgt_elem:
                    # '9 1' → '9-1'
                    weight = "-".join(wgt_elem.text.split())

                # Jockey claim / allowance
                claim = "NA"
                claim_elem = card.find("span", attrs={"data-test-selector": "RC-cardPage-runnerJockey-allowance"})
                if claim_elem:
                    claim = claim_elem.text.strip()
                
                runner_data.append({
                    "name": name,
                    "horse_url": horse_url,
                    "ts": ts,
                    "rpr": rpr,
                    "price": price,
                    "age": age,
                    "jockey_name": jockey_name,
                    "jockey_url": jockey_url,
                    "trainer_name": trainer_name,
                    "trainer_url": trainer_url,
                    "weight": weight,
                    "claim": claim,
                })

            except Exception as e:
                print(f"Error processing runner card: {e}")
                runner_data.append({
                    "name": "Error",
                    "horse_url": None,
                    "ts": None,
                    "rpr": None,
                    "price": None,
                    "age": None,
                    "jockey_name": None,
                    "jockey_url": None,
                    "trainer_name": None,
                    "trainer_url": None,
                    "weight": "NA",
                    "claim": "NA",
                    "error": str(e)
                })

        return {
            "url": url,
            "time": race_time,
            "header": header_text,
            "date": race_date,
            "going": going,
            "distance": distance,
            "class": race_class,
            "runners": runner_data
        }

    except Exception as e:
        print(f"[ERROR] Scraping failed on {url}: {e}")
        return {"url": url, "error": str(e)}

# Process all race URLs
def process_race_urls(urls):
    """Process each race URL with requests/BeautifulSoup instead of Selenium"""
    results = []
    for url in urls:
        print(f"➡️  Visiting: {url}")
        try:
            html = fetch_html(url)
            if html:
                result = scrape_race(html, url)
                results.append(result)
            else:
                print(f"[ERROR] Failed to fetch HTML for {url}")
                results.append({"url": url, "error": "Failed to fetch HTML"})
        except Exception as e:
            print(f"[ERROR] Failed to load or scrape {url}: {e}")
            results.append({"url": url, "error": str(e)})
    return results

# Save race data to CSV - function remains the same as the original script
# Save race data to CSV - updated with remapped column names
def save_results_to_csv(results, filename):
    # Updated fieldnames using the rename mapping
    fieldnames = [
        "url", "race_time", "track_name", "race_date", "going", "distance", "class",
        "horse_name", "horse_url", "runner_ts", "rating", "price",
        "age", "jockey", "jockey_url",
        "trainer", "trainer_url",
        "weight", "claims"
    ]

    # Create a list to hold all rows before deduplication
    all_rows = []
    
    # Process all races and runners
    for race in results:
        if "runners" in race:
            for runner in race["runners"]:
                row = {
                    "url": race.get("url"),
                    "race_time": race.get("time"),  # time -> race_time
                    "track_name": race.get("header"),  # header -> track_name
                    "race_date": race.get("date"),  # date -> race_date
                    "going": race.get("going"),
                    "distance": race.get("distance"),
                    "class": race.get("class"),
                    "horse_name": runner.get("name"),  # runner_name -> horse_name
                    "horse_url": runner.get("horse_url"),
                    "runner_ts": runner.get("ts"),
                    "rating": runner.get("rpr"),  # runner_rpr -> rating
                    "price": runner.get("price"),
                    "age": runner.get("age"),  # runner_age -> age
                    "jockey": runner.get("jockey_name"),  # jockey_name -> jockey
                    "jockey_url": runner.get("jockey_url"),
                    "trainer": runner.get("trainer_name"),  # trainer_name -> trainer
                    "trainer_url": runner.get("trainer_url"),
                    "weight": runner.get("weight"),  # runner_weight -> weight
                    "claims": runner.get("claim"),  # jockey_claim -> claims
                }
                all_rows.append(row)
    
    # Remove duplicate rows
    unique_rows = []
    seen = set()
    
    for row in all_rows:
        # Create a tuple of values that we want to check for uniqueness
        # Using horse_name, horse_url, and url as keys for uniqueness
        key = (row["horse_name"], row["horse_url"], row["url"])
        
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    
    print(f"Removed {len(all_rows) - len(unique_rows)} duplicate rows")
    
    # Write the deduplicated rows to CSV
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in unique_rows:
            writer.writerow(row)

    print(f"✅ Saved {len(results)} races with {len(unique_rows)} unique runners to {filename}")

# Main function - modified to use BeautifulSoup approach
def main():
    print("Starting Racing Post scraper with BeautifulSoup...")

    try:
        # STAGE 1: Get all race links - modified to work with BS4 instead of Selenium
        print("Stage 1: Fetching race links...")
        
        # First try the homepage to find race links
        html = fetch_html(START_URL)
        if not html:
            print("Failed to fetch homepage. Exiting.")
            return
        
        race_links = get_links_from_meeting_boxes(html)
        
        # If we don't find any links on the homepage, try the racecards page
        if not race_links:
            print("No race links found on homepage, trying racecards page...")
            racecards_url = START_URL + "racecards"  # Common URL pattern
            racecards_html = fetch_html(racecards_url)
            
            if racecards_html:
                race_links = get_links_from_meeting_boxes(racecards_html)
            
            # If still no links, try a few other common URL patterns
            if not race_links:
                print("No race links found on racecards page, trying other URL patterns...")
                for pattern in ["races", "race-cards", "meetings", "today"]:
                    url = START_URL + pattern
                    html = fetch_html(url)
                    if html:
                        links = get_links_from_meeting_boxes(html)
                        if links:
                            race_links = links
                            break
        
        print(f"Found {len(race_links)} race links")
        
        if not race_links:
            print("Could not find any race links. Exiting.")
            return

        # STAGE 2: Process each race URL and extract data
        print("Stage 2: Scraping race data...")
        race_results = process_race_urls(race_links)
        
        # Save data to CSV
        save_results_to_csv(race_results, CSV_FILENAME)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    print("Scraper finished")

if __name__ == "__main__":
    main()
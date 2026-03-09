import requests
from bs4 import BeautifulSoup

def get_links_directly_from_tag(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    base_url = "https://www.racingpost.com"
    
    try:
        print(f"Searching for <a> tags containing 'data-race-is-over'...")
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all <a> tags that specifically have this data attribute
        race_links = soup.find_all('a', attrs={"data-race-is-over": True})
        
        results = []
        for a in race_links:
            href = a.get('href')
            if href:
                full_url = href if href.startswith('http') else base_url + href
                
                # Extract the status from the tag for your info
                status = a.get('data-race-is-over')
                
                results.append(full_url)
                print(f"Found Link: {full_url} (Is Over: {status})")

        # Deduplicate
        unique_links = list(dict.fromkeys(results))
        print(f"\nTotal Unique Links Found: {len(unique_links)}")
        
        return unique_links

    except Exception as e:
        print(f"Error: {e}")
        return []

# Run
all_urls = get_links_directly_from_tag("https://www.racingpost.com/racecards/tomorrow")
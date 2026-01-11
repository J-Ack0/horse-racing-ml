import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from datetime import datetime
import os

# --- UTILITIES ---

def clean_text(text):
    """Removes newlines, tabs, and collapses multiple spaces into one."""
    if not text:
        return ""
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_pos_digit(pos_str):
    """Extracts '1' from '1 (8)' or returns the original string."""
    match = re.search(r'\d+', str(pos_str))
    return match.group(0) if match else pos_str

def get_headers():
    """Returns a robust set of headers to avoid 406 errors."""
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.racingpost.com/',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

# --- SCRAPER LOGIC ---

def get_finished_race_links(session, target_date=None):
    url = f"https://www.racingpost.com/racecards/{target_date}/" if target_date else "https://www.racingpost.com/racecards/"
    base_url = "https://www.racingpost.com"
    
    try:
        print(f"--- Fetching finished races from {url} ---")
        response = session.get(url, headers=get_headers(), timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        race_links = soup.find_all('a', class_='RC-meetingItem__link', attrs={"data-race-is-over": "1"})
        
        results = []
        for a in race_links:
            href = a.get('href')
            if href:
                href = href.replace('/racecards/', '/results/')
                results.append(href if href.startswith('http') else base_url + href)
        
        unique_links = list(dict.fromkeys(results))
        print(f"Found {len(unique_links)} completed races to audit.\n")
        return unique_links
    except Exception as e:
        print(f"Error fetching links: {e}")
        return []

def scrape_actual_results(session, url):
    try:
        response = session.get(url, headers=get_headers(), timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        track_el = soup.select_one('[data-test-selector="RC-courseHeader__name"]') or soup.select_one('.rp-raceHeader__courseName')
        time_el = soup.select_one('[data-test-selector="RC-courseHeader__time"]') or soup.select_one('.rp-raceHeader__time')
        
        track = clean_text(track_el.text) if track_el else "Unknown Track"
        time_val = clean_text(time_el.text) if time_el else "Unknown Time"
        
        race_rows = []
        runner_rows = soup.find_all('tr', class_='rp-horseTable__mainRow', attrs={"data-test-selector": "table-row"})
        
        for row in runner_rows:
            name_el = row.find('a', {'data-test-selector': 'link-horseName'})
            pos_el = row.find('span', {'data-test-selector': 'text-horsePosition'})
            
            if name_el:
                h_name = clean_text(name_el.text)
                raw_pos = clean_text(pos_el.text) if pos_el else "N/A"
                
                race_rows.append({
                    "track_name": track,
                    "time": time_val,
                    "horse_name": h_name,
                    "actual_position": extract_pos_digit(raw_pos)
                })
        return race_rows
    except Exception as e:
        print(f"  [!] Error scraping results for {url}: {e}")
        return []

# --- AUDIT PIPELINE ---

def run_loss_function_pipeline(date_str):
    session = requests.Session()
    
    links = get_finished_race_links(session, date_str)
    all_actuals = []
    
    for link in links:
        print(f"Scraping: {link}")
        results = scrape_actual_results(session, link)
        if results:
            all_actuals.extend(results)
        time.sleep(2) 
    
    if not all_actuals:
        print("No actual results found to audit.")
        return

    df_actuals = pd.DataFrame(all_actuals)

    # Load Predictions
    pred_filename = f"Inference_Outputs/Output_{date_str}.csv"
    if not os.path.exists(pred_filename):
        print(f"Prediction file {pred_filename} not found.")
        return
    
    df_preds = pd.read_csv(pred_filename)

    # 1. Merge (on horse_name)
    merged_df = pd.merge(
        df_preds, 
        df_actuals[['horse_name', 'actual_position']], 
        on=['horse_name'], 
        how='left'
    )

    # 2. Composite Key: Race_id
    merged_df['Race_id'] = merged_df['track_name'] + "_" + date_str + "_" + merged_df['time']

    # 3. Calculate Inferred Position
    prob_col = df_preds.columns[-1] 
    
    merged_df = merged_df.sort_values(by=['Race_id', prob_col], ascending=[True, False])
    merged_df['inferred_position'] = merged_df.groupby('Race_id')[prob_col].rank(
        ascending=False, method='first'
    ).astype(int)

    # 4. Accuracy Log Calculation (with Abandoned Race Filtering)
    summary_data = []
    
    for race_id, group in merged_df.groupby('Race_id'):
        
        # --- LOGICAL FILTER ---
        # If the race was abandoned or we didn't scrape it, 'actual_position' 
        # will either be all NaN (from merge) or won't contain a '1'.
        # We check if there is at least one horse with position '1' in the race.
        actual_winner_row = group[group['actual_position'] == '1']
        
        if actual_winner_row.empty:
            # This race has no ground truth results. Skip from Summary/Strike Rate.
            continue

        # Find our predicted winner (Inferred Pos 1)
        top_preds = group[group['inferred_position'] == 1]
        if top_preds.empty: 
            continue
        
        top_pred = top_preds.iloc[0]
        actual_winner_name = actual_winner_row.iloc[0]['horse_name']
        
        hit = 0
        if top_pred['horse_name'].strip().upper() == actual_winner_name.strip().upper():
            hit = 1
        
        summary_data.append({
            "Race_id": race_id,
            "Predicted_Winner": top_pred['horse_name'],
            "Actual_Winner": actual_winner_name,
            "Hit": hit
        })

    df_summary = pd.DataFrame(summary_data)

    # 5. Save Files
    output_dir = "Inference_Actuals"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    audit_path = os.path.join(output_dir, f"{date_str.replace('-', '_')}_Actual.csv")
    merged_df.to_csv(audit_path, index=False)

    output_2 = "Inference_Summary"
    if not os.path.exists(output_2): os.makedirs(output_2)
    log_path = os.path.join(output_2, f"{date_str.replace('-', '_')}_Summary_Log.csv")
    df_summary.to_csv(log_path, index=False)

    print(f"\n--- Audit Complete ---")
    print(f"Detailed Audit: {audit_path}")
    print(f"Summary Log: {log_path}")
    
    if not df_summary.empty:
        # Strike rate calculation only uses 'valid' races now
        strike_rate = df_summary['Hit'].mean()
        total_races = len(df_summary)
        hits = df_summary['Hit'].sum()
        
        print(f"Valid Races Scraped: {total_races}")
        print(f"Total Wins: {hits}")
        print(f"Strike Rate: {strike_rate:.2%}")
    else:
        print("No valid races with finishing results were found for analytics.")

if __name__ == "__main__":
    today_str = datetime.now().strftime('%Y-%m-%d')
    run_loss_function_pipeline(today_str)
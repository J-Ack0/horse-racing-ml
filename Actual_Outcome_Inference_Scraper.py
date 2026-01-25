import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from datetime import datetime, timedelta
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
    """Extracts '1' from '1 (8)' or returns the original string - always returns string."""
    match = re.search(r'\d+', str(pos_str))
    result = match.group(0) if match else str(pos_str)
    # Ensure it's always returned as a string
    return str(result)

def fractional_to_decimal(frac_str):
    """Convert fractional odds to decimal odds"""
    if pd.isna(frac_str) or frac_str == '' or frac_str == 'N/A':
        return None
    
    frac_str = str(frac_str).strip()
    
    # Handle special cases
    if frac_str.lower() == 'evens':
        return 2.0
    
    try:
        # Handle X/Y format
        if '/' in frac_str:
            parts = frac_str.split('/')
            if len(parts) == 2:
                num = float(parts[0])
                den = float(parts[1])
                return (num / den) + 1.0
        # Already decimal
        return float(frac_str)
    except Exception as e:
        print(f"Warning: Could not convert odds '{frac_str}': {e}")
        return None

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

        soup = BeautifulSoup(response.text, "html.parser")

        now = datetime.now()
        cutoff = now - timedelta(minutes=30)

        race_links = soup.find_all("a", class_="RC-meetingItem__link")
        results = []

        for a in race_links:
            race_time_str = a.get("data-race-time")
            href = a.get("href")

            if not race_time_str or not href:
                continue

            race_date = (
                datetime.strptime(target_date, "%Y-%m-%d").date()
                if target_date
                else now.date()
            )

            race_time = datetime.strptime(race_time_str, "%H:%M").time()
            race_datetime = datetime.combine(race_date, race_time)

            if race_datetime <= cutoff:
                href = href.replace("/racecards/", "/results/")
                results.append(href if href.startswith("http") else base_url + href)

        unique_links = list(dict.fromkeys(results))
        print(f"Found {len(unique_links)} completed races to audit.\n")
        return unique_links

    except Exception as e:
        print(f"Error fetching links: {e}")
        return []


def scrape_actual_results(session, url, index=0):
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

        if index == 1:
            return []

        if hasattr(e, "response") and e.response is not None:
            if e.response.status_code == 406:
                return scrape_actual_results(session, url, 1)

        return []


# --- BETTING ROI CALCULATION (UPDATED FOR NEW SCHEMA) ---

def calculate_bet_roi(row):
    """
    Calculate profit/loss for a single bet using NEW SCHEMA
    
    New schema uses:
    - kelly_stake (stake amount)
    - odds_live (fractional format like "5/2")
    - actual_position (string like "1", "2", etc.)
    """
    if pd.isna(row['kelly_stake']):
        return 0.0
    
    stake = float(row['kelly_stake'])
    
    # Convert fractional odds to decimal
    decimal_odds = fractional_to_decimal(row['odds_live'])
    if decimal_odds is None or decimal_odds <= 1.0:
        return 0.0
    
    # Check if bet won (position 1) - ROBUST COMPARISON
    actual_pos = row.get('actual_position', None)
    
    # Handle both string and numeric, strip whitespace
    if pd.isna(actual_pos):
        won = False
    else:
        # Convert to string, strip whitespace, compare
        pos_str = str(actual_pos).strip()
        won = (pos_str == '1' or pos_str == '1.0')
    
    if won:
        # Profit = stake * (odds - 1)
        return stake * (decimal_odds - 1.0)
    else:
        # Loss = -stake
        return -stake


# --- AUDIT PIPELINE (UPDATED FOR NEW SCHEMA) ---

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

    # Load Betting Recommendations (NEW PATH)
    bet_filename = f"Inference_Bets/Bets_{date_str}.csv"
    has_bets = os.path.exists(bet_filename)
    
    if has_bets:
        df_bets = pd.read_csv(bet_filename)
        print(f"\n✓ Loaded {len(df_bets)} betting recommendations")
        
        # NEW SCHEMA: Rename horse_name_odds to horse_name for easier merging
        if 'horse_name_odds' in df_bets.columns:
            df_bets['horse_name'] = df_bets['horse_name_odds']
        
        print(f"Bet columns: {df_bets.columns.tolist()}")
    else:
        print(f"\n⚠️  No betting file found: {bet_filename}")
        df_bets = pd.DataFrame()

    # 1. Merge predictions with actuals (on horse_name)
    merged_df = pd.merge(
        df_preds, 
        df_actuals[['horse_name', 'actual_position']], 
        on=['horse_name'], 
        how='left'
    )

    # 2. Composite Key: Race_id
    merged_df['Race_id'] = merged_df['track_name'] + "_" + date_str + "_" + merged_df['time']

    # 3. Calculate Inferred Position
    prob_col = 'win_probability' if 'win_probability' in df_preds.columns else df_preds.columns[-1]
    
    merged_df = merged_df.sort_values(by=['Race_id', prob_col], ascending=[True, False])
    merged_df['inferred_position'] = merged_df.groupby('Race_id')[prob_col].rank(
        ascending=False, method='first'
    ).astype(int)

    # 4. Merge betting data if available (UPDATED FOR NEW SCHEMA)
    if has_bets and not df_bets.empty:
        # Normalize horse names for better matching
        merged_df['horse_name_norm'] = merged_df['horse_name'].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
        df_bets['horse_name_norm'] = df_bets['horse_name'].str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)
        
        # DEBUG: Print sample normalized names
        print("\n--- Sample normalized horse names (predictions) ---")
        print(merged_df[['horse_name', 'horse_name_norm']].head(3))
        print("\n--- Sample normalized horse names (bets) ---")
        print(df_bets[['horse_name', 'horse_name_norm']].head(3))
        
        # NEW SCHEMA: Use correct column names
        merge_columns = ['horse_name_norm', 'kelly_stake', 'odds_live', 'edge', 'ev', 'action']
        available_columns = ['horse_name_norm'] + [col for col in merge_columns[1:] if col in df_bets.columns]
        
        merged_df = pd.merge(
            merged_df,
            df_bets[available_columns],
            on='horse_name_norm',
            how='left'
        )
        
        # DEBUG: Check how many bets were matched
        matched_bets = merged_df['kelly_stake'].notna().sum()
        print(f"\n✓ Matched {matched_bets} bets to predictions")
        
        # Calculate ROI for bets (NEW FUNCTION)
        merged_df['bet_roi'] = merged_df.apply(calculate_bet_roi, axis=1)
        
        # Flag if this was a bet
        merged_df['was_bet'] = merged_df['kelly_stake'].notna()
    else:
        merged_df['was_bet'] = False
        merged_df['bet_roi'] = 0.0
        merged_df['kelly_stake'] = None
        merged_df['action'] = None

    # 5. Accuracy Log Calculation (with Abandoned Race Filtering)
    summary_data = []
    
    for race_id, group in merged_df.groupby('Race_id'):
        
        # Check if race has valid results
        actual_winner_row = group[group['actual_position'] == '1']
        
        if actual_winner_row.empty:
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
        
        # Betting analysis
        race_bets = group[group['was_bet'] == True]
        bet_placed = len(race_bets) > 0
        
        # Check if any of our bets won - ROBUST COMPARISON
        if bet_placed:
            # Convert actual_position to string, strip whitespace, check for '1'
            bet_won = False
            for _, bet_row in race_bets.iterrows():
                actual_pos = bet_row.get('actual_position', None)
                if not pd.isna(actual_pos):
                    pos_str = str(actual_pos).strip()
                    if pos_str == '1' or pos_str == '1.0':
                        bet_won = True
                        break
        else:
            bet_won = False
        
        total_staked = race_bets['kelly_stake'].sum() if bet_placed else 0.0
        total_roi = race_bets['bet_roi'].sum() if bet_placed else 0.0
        
        # DEBUG: Print details for races where we bet
        if bet_placed:
            print(f"\n--- Race: {race_id} ---")
            print(f"Bets placed: {len(race_bets)}")
            print(f"Bet horses: {race_bets['horse_name'].tolist()}")
            print(f"Odds: {race_bets['odds_live'].tolist()}")
            print(f"Stakes: {race_bets['kelly_stake'].tolist()}")
            print(f"Actual positions (raw): {race_bets['actual_position'].tolist()}")
            print(f"Actual positions (type): {[type(p).__name__ for p in race_bets['actual_position'].tolist()]}")
            print(f"Actual winner: {actual_winner_name}")
            print(f"Bet won: {bet_won}")
            print(f"Total staked: £{total_staked:.2f}")
            print(f"Total ROI: £{total_roi:.2f}")
            
            # Extra diagnostic for position checking
            for _, bet_row in race_bets.iterrows():
                horse = bet_row['horse_name']
                pos = bet_row['actual_position']
                pos_str = str(pos).strip() if not pd.isna(pos) else 'NaN'
                print(f"  → {horse}: position={repr(pos)} (type={type(pos).__name__}), string={repr(pos_str)}, is_winner={pos_str in ['1', '1.0']}")
        
        summary_data.append({
            "Race_id": race_id,
            "Predicted_Winner": top_pred['horse_name'],
            "Actual_Winner": actual_winner_name,
            "Prediction_Hit": hit,
            "Bet_Placed": 1 if bet_placed else 0,
            "Bet_Won": 1 if bet_won else 0,
            "Total_Staked": total_staked,
            "Money_Made": total_roi
        })

    df_summary = pd.DataFrame(summary_data)

    # 6. Save Files
    output_dir = "Inference_Actuals"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    audit_path = os.path.join(output_dir, f"{date_str.replace('-', '_')}_Actual.csv")
    merged_df.to_csv(audit_path, index=False)

    output_2 = "Inference_Summary"
    if not os.path.exists(output_2): os.makedirs(output_2)
    log_path = os.path.join(output_2, f"{date_str.replace('-', '_')}_Summary_Log.csv")
    df_summary.to_csv(log_path, index=False)

    print(f"\n{'='*100}")
    print(f"AUDIT COMPLETE - {date_str}")
    print(f"{'='*100}")
    print(f"Detailed Audit: {audit_path}")
    print(f"Summary Log: {log_path}")
    
    if not df_summary.empty:
        # Strike rate calculation
        strike_rate = df_summary['Prediction_Hit'].mean()
        total_races = len(df_summary)
        prediction_hits = df_summary['Prediction_Hit'].sum()
        
        print(f"\n{'='*100}")
        print("PREDICTION PERFORMANCE")
        print(f"{'='*100}")
        print(f"Valid Races: {total_races}")
        print(f"Prediction Wins: {int(prediction_hits)}")
        print(f"Strike Rate: {strike_rate:.2%}")
        
        # Betting performance
        if has_bets and df_summary['Bet_Placed'].sum() > 0:
            total_bets = df_summary['Bet_Placed'].sum()
            total_bet_wins = df_summary['Bet_Won'].sum()
            total_staked = df_summary['Total_Staked'].sum()
            total_profit = df_summary['Money_Made'].sum()
            roi_pct = (total_profit / total_staked * 100) if total_staked > 0 else 0
            
            print(f"\n{'='*100}")
            print("BETTING PERFORMANCE")
            print(f"{'='*100}")
            print(f"Races with Bets: {int(total_bets)}")
            print(f"Bets Won: {int(total_bet_wins)}")
            print(f"Bet Win Rate: {(total_bet_wins/total_bets*100):.2f}%" if total_bets > 0 else "N/A")
            print(f"Total Staked: £{total_staked:.2f}")
            print(f"Total Profit/Loss: {'£' if total_profit >= 0 else '-£'}{abs(total_profit):.2f}")
            print(f"ROI: {roi_pct:+.2f}%")
            
            # Detailed bet results
            print(f"\n{'='*100}")
            print("DETAILED BET RESULTS")
            print(f"{'='*100}")
            print(f"{'Race':<40} {'Bet?':<8} {'Won?':<8} {'Staked':<12} {'Profit/Loss':<12}")
            print(f"{'-'*100}")
            
            for _, row in df_summary.iterrows():
                if row['Bet_Placed']:
                    bet_status = "YES" if row['Bet_Placed'] else "NO"
                    won_status = "WIN" if row['Bet_Won'] else "LOSS"
                    profit_str = f"£{row['Money_Made']:+.2f}"
                    
                    print(
                        f"{row['Race_id'][:39]:<40} "
                        f"{bet_status:<8} "
                        f"{won_status:<8} "
                        f"£{row['Total_Staked']:.2f}      "
                        f"{profit_str:<12}"
                    )
        
        print(f"\n{'='*100}")
    else:
        print("No valid races with finishing results were found for analytics.")

if __name__ == "__main__":
    today_str = datetime.now().strftime('%Y-%m-%d')
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    run_loss_function_pipeline(yesterday_str)
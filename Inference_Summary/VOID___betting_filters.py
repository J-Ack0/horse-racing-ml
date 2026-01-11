import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple
import time

# ============================================================
#                    CONFIGURATION
# ============================================================

API_KEY = 'YOUR_ODDS_API_KEY'  # Replace with your actual API key
API_BASE_URL = 'https://api.the-odds-api.com/v4'

# Betting thresholds
MIN_EDGE = 0.12  # Minimum 12% edge to consider betting
MIN_EV = 0.10    # Minimum 10% expected value
CLV_THRESHOLD = 1.03  # Target 3%+ CLV

# Request tracking
MONTHLY_REQUEST_LIMIT = 450  # Reserve 50 for testing/errors
requests_used = 0

# ============================================================
#                    UTILITY FUNCTIONS
# ============================================================

def clean_track_name(track_name: str) -> str:
    """Remove suffixes in parentheses like (AW), (Polytrack)"""
    if not isinstance(track_name, str):
        return track_name
    
    # Remove anything in parentheses
    cleaned = re.sub(r'\s*\([^)]*\)', '', track_name).strip()
    return cleaned

def parse_race_datetime(row) -> Tuple[datetime, str]:
    """
    Parse the messy date format from CSV
    
    Input: 
    - track_name: "Chelmsford (AW)"
    - race_date: "11 Jan 2026Polytrack"
    - time: "4:12"
    
    Output: (datetime object, clean_track_name)
    """
    # Extract date from race_date column (before any text like "Polytrack")
    date_str = re.match(r'^([\d]+\s+[A-Za-z]+\s+[\d]{4})', str(row['race_date']))
    if not date_str:
        return None, None
    
    date_str = date_str.group(1)  # e.g., "11 Jan 2026"
    
    # Extract time
    time_str = str(row['time'])  # e.g., "4:12"
    
    # Combine into full datetime string
    datetime_str = f"{date_str} {time_str}"
    
    try:
        # Parse to datetime
        race_datetime = datetime.strptime(datetime_str, "%d %b %Y %H:%M")
        
        # Clean track name
        clean_track = clean_track_name(row['track_name'])
        
        return race_datetime, clean_track
    except Exception as e:
        print(f"Error parsing datetime: {e} | Input: {datetime_str}")
        return None, None

def load_predictions(csv_path: str) -> pd.DataFrame:
    """Load and clean prediction CSV"""
    print(f"Loading predictions from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    
    # Parse datetime and clean track names
    df[['race_datetime', 'clean_track']] = df.apply(
        lambda row: pd.Series(parse_race_datetime(row)), 
        axis=1
    )
    
    # Drop rows with parse errors
    df = df.dropna(subset=['race_datetime', 'clean_track'])
    
    # Remove duplicate races (same track + time)
    df['race_key'] = df['clean_track'] + '_' + df['race_datetime'].astype(str)
    
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['race_key', 'horse_name'])
    after_dedup = len(df)
    
    print(f"✓ Loaded {after_dedup} predictions ({before_dedup - after_dedup} duplicates removed)")
    print(f"✓ Date range: {df['race_datetime'].min()} to {df['race_datetime'].max()}")
    
    return df

# ============================================================
#                    ODDS API FUNCTIONS
# ============================================================

def fetch_all_races(sport='horse_racing_uk', regions='uk', 
                   commence_time_from=None, commence_time_to=None) -> Dict:
    """
    Fetch all races from The Odds API
    
    Returns: dict with race data and request count
    """
    global requests_used
    
    params = {
        'apiKey': API_KEY,
        'regions': regions,
        'markets': 'h2h',  # Win market
        'oddsFormat': 'decimal',
        'bookmakers': 'paddypower,williamhill,ladbrokes,bet365,skybet'
    }
    
    # Optional time filters
    if commence_time_from:
        params['commenceTimeFrom'] = commence_time_from
    if commence_time_to:
        params['commenceTimeTo'] = commence_time_to
    
    url = f"{API_BASE_URL}/sports/{sport}/odds"
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Track request usage
        requests_remaining = int(response.headers.get('x-requests-remaining', 0))
        requests_used = MONTHLY_REQUEST_LIMIT - requests_remaining
        
        print(f"✓ API Request successful | Requests used: {requests_used}/{MONTHLY_REQUEST_LIMIT}")
        
        return {
            'data': response.json(),
            'requests_remaining': requests_remaining,
            'timestamp': datetime.now()
        }
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request failed: {e}")
        return {'data': [], 'requests_remaining': 0, 'timestamp': datetime.now()}

def parse_odds_response(api_response: Dict) -> pd.DataFrame:
    """
    Parse Odds API response into structured DataFrame
    
    Returns: DataFrame with columns:
    - track_name, race_datetime, horse_name, bookmaker, odds
    """
    races_data = []
    
    for event in api_response['data']:
        # Extract race info
        track_name = event.get('home_team', 'Unknown')
        race_datetime = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
        
        # Extract odds from each bookmaker
        for bookmaker in event.get('bookmakers', []):
            bookmaker_name = bookmaker['title']
            
            for market in bookmaker.get('markets', []):
                if market['key'] == 'h2h':  # Win market
                    for outcome in market['outcomes']:
                        races_data.append({
                            'track_name': track_name,
                            'race_datetime': race_datetime,
                            'horse_name': outcome['name'],
                            'bookmaker': bookmaker_name,
                            'decimal_odds': outcome['price']
                        })
    
    df = pd.DataFrame(races_data)
    
    if not df.empty:
        print(f"✓ Parsed {len(df)} odds entries from {len(api_response['data'])} races")
    
    return df

# ============================================================
#                    ANALYSIS FUNCTIONS
# ============================================================

def calculate_best_odds(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each horse, find the best available odds across all bookmakers
    """
    if odds_df.empty:
        return pd.DataFrame()
    
    best_odds = odds_df.groupby(['track_name', 'race_datetime', 'horse_name']).agg({
        'decimal_odds': 'max',  # Best (highest) odds
        'bookmaker': lambda x: odds_df.loc[x.index[odds_df.loc[x.index, 'decimal_odds'].idxmax()], 'bookmaker']
    }).reset_index()
    
    best_odds.rename(columns={'bookmaker': 'best_bookmaker'}, inplace=True)
    
    return best_odds

def match_predictions_to_odds(predictions_df: pd.DataFrame, 
                             odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match model predictions to bookmaker odds
    
    Returns: Merged DataFrame with predictions and odds
    """
    # Get best odds for each horse
    best_odds = calculate_best_odds(odds_df)
    
    # Normalize horse names for matching (lowercase, strip spaces)
    predictions_df['horse_name_norm'] = predictions_df['horse_name'].str.lower().str.strip()
    best_odds['horse_name_norm'] = best_odds['horse_name'].str.lower().str.strip()
    
    # Normalize track names
    predictions_df['track_norm'] = predictions_df['clean_track'].str.lower().str.strip()
    best_odds['track_norm'] = best_odds['track_name'].str.lower().str.strip()
    
    # Merge on track + horse name
    merged = predictions_df.merge(
        best_odds,
        left_on=['track_norm', 'horse_name_norm'],
        right_on=['track_norm', 'horse_name_norm'],
        how='inner',
        suffixes=('_pred', '_odds')
    )
    
    print(f"✓ Matched {len(merged)} horses with odds data")
    
    return merged

def calculate_ev_and_edge(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Expected Value (EV) and Edge
    
    EV = (win_probability * decimal_odds) - 1
    Edge = win_probability - implied_probability
    Implied Probability = 1 / decimal_odds
    """
    # Calculate implied probability from odds
    merged_df['implied_prob'] = 1 / merged_df['decimal_odds']
    
    # Calculate edge (model prob - market prob)
    merged_df['edge'] = merged_df['win_probability'] - merged_df['implied_prob']
    
    # Calculate Expected Value
    merged_df['ev'] = (merged_df['win_probability'] * merged_df['decimal_odds']) - 1
    
    # Calculate ROI percentage
    merged_df['roi_pct'] = merged_df['ev'] * 100
    
    return merged_df

def identify_value_bets(analysis_df: pd.DataFrame, 
                       min_edge=MIN_EDGE, 
                       min_ev=MIN_EV) -> pd.DataFrame:
    """
    Filter for value betting opportunities
    """
    value_bets = analysis_df[
        (analysis_df['edge'] > min_edge) & 
        (analysis_df['ev'] > min_ev)
    ].copy()
    
    # Sort by EV descending
    value_bets = value_bets.sort_values('ev', ascending=False)
    
    return value_bets

def calculate_kelly_stake(win_prob: float, decimal_odds: float, 
                         bankroll: float = 1000, fraction: float = 0.25) -> float:
    """
    Calculate Kelly Criterion stake size
    
    Kelly = (bp - q) / b
    where:
    - b = decimal_odds - 1
    - p = win_probability
    - q = 1 - p
    
    Using fractional Kelly (0.25 = quarter Kelly) for safety
    """
    b = decimal_odds - 1
    p = win_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    
    # Apply fractional Kelly for bankroll protection
    fractional_kelly = kelly * fraction
    
    # Convert to stake amount (as % of bankroll)
    stake = max(0, fractional_kelly * bankroll)
    
    return stake

# ============================================================
#                    REPORTING FUNCTIONS
# ============================================================

def generate_morning_report(analysis_df: pd.DataFrame, 
                           snapshot_time: datetime) -> str:
    """Generate morning snapshot report"""
    
    report = []
    report.append("="*80)
    report.append(f"MORNING SNAPSHOT - {snapshot_time.strftime('%Y-%m-%d %H:%M')}")
    report.append("="*80)
    report.append(f"Total Races Analyzed: {analysis_df['race_key'].nunique()}")
    report.append(f"Total Horses Matched: {len(analysis_df)}")
    report.append(f"Requests Used: {requests_used}/{MONTHLY_REQUEST_LIMIT}")
    report.append("")
    
    # Value opportunities
    value_bets = identify_value_bets(analysis_df)
    report.append(f"VALUE OPPORTUNITIES FOUND: {len(value_bets)}")
    report.append("-"*80)
    
    if not value_bets.empty:
        report.append(f"{'Track':<20} {'Horse':<20} {'Time':<8} {'Model':<8} {'Odds':<8} {'Edge':<8} {'EV':<8} {'Bookmaker':<15}")
        report.append("-"*80)
        
        for _, row in value_bets.head(10).iterrows():
            report.append(
                f"{row['clean_track'][:19]:<20} "
                f"{row['horse_name'][:19]:<20} "
                f"{row['race_datetime_pred'].strftime('%H:%M'):<8} "
                f"{row['win_probability']:.3f}    "
                f"{row['decimal_odds']:.2f}     "
                f"{row['edge']:+.3f}   "
                f"{row['ev']:+.3f}   "
                f"{row['best_bookmaker']:<15}"
            )
    else:
        report.append("No value bets found meeting criteria.")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)

def generate_prerace_report(analysis_df: pd.DataFrame, 
                           morning_odds: pd.DataFrame,
                           snapshot_time: datetime) -> str:
    """Generate pre-race comparison report with CLV tracking"""
    
    report = []
    report.append("="*80)
    report.append(f"PRE-RACE SNAPSHOT (T-15) - {snapshot_time.strftime('%Y-%m-%d %H:%M')}")
    report.append("="*80)
    
    # Calculate odds movement
    if not morning_odds.empty:
        movement = analysis_df.merge(
            morning_odds[['horse_name_norm', 'track_norm', 'decimal_odds']],
            on=['horse_name_norm', 'track_norm'],
            suffixes=('_current', '_morning')
        )
        
        movement['odds_change_pct'] = (
            (movement['decimal_odds_current'] - movement['decimal_odds_morning']) / 
            movement['decimal_odds_morning'] * 100
        )
        
        movement['clv'] = movement['decimal_odds_morning'] / movement['decimal_odds_current']
        
        report.append(f"ODDS MOVEMENT ANALYSIS")
        report.append("-"*80)
        report.append(f"{'Horse':<20} {'Morning':<10} {'Current':<10} {'Change':<10} {'CLV':<8} {'Signal':<15}")
        report.append("-"*80)
        
        for _, row in movement.iterrows():
            signal = "✓ LENGTHENED" if row['odds_change_pct'] > 5 else "✗ SHORTENED" if row['odds_change_pct'] < -5 else "→ STABLE"
            
            report.append(
                f"{row['horse_name'][:19]:<20} "
                f"{row['decimal_odds_morning']:.2f}        "
                f"{row['decimal_odds_current']:.2f}        "
                f"{row['odds_change_pct']:+.1f}%      "
                f"{row['clv']:.3f}   "
                f"{signal:<15}"
            )
    
    report.append("")
    
    # Final betting recommendations
    value_bets = identify_value_bets(analysis_df)
    report.append(f"FINAL BETTING RECOMMENDATIONS: {len(value_bets)}")
    report.append("-"*80)
    
    if not value_bets.empty:
        report.append(f"{'Horse':<20} {'Odds':<8} {'Edge':<8} {'EV':<8} {'Kelly':<10} {'Action':<15}")
        report.append("-"*80)
        
        for _, row in value_bets.iterrows():
            kelly_stake = calculate_kelly_stake(row['win_probability'], row['decimal_odds'])
            action = "BET" if row['edge'] > 0.15 else "SMALL BET"
            
            report.append(
                f"{row['horse_name'][:19]:<20} "
                f"{row['decimal_odds']:.2f}     "
                f"{row['edge']:+.3f}   "
                f"{row['ev']:+.3f}   "
                f"£{kelly_stake:.2f}     "
                f"{action:<15}"
            )
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)

# ============================================================
#                    MAIN WORKFLOW
# ============================================================

def run_morning_snapshot(predictions_df: pd.DataFrame, 
                        target_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run morning snapshot (6 AM)
    
    Returns: (analysis_df, morning_odds_df)
    """
    print("\n" + "="*80)
    print("RUNNING MORNING SNAPSHOT (6 AM)")
    print("="*80 + "\n")
    
    # Fetch all races for the day
    if target_date:
        commence_from = f"{target_date}T00:00:00Z"
        commence_to = f"{target_date}T23:59:59Z"
    else:
        commence_from = None
        commence_to = None
    
    api_response = fetch_all_races(
        commence_time_from=commence_from,
        commence_time_to=commence_to
    )
    
    # Parse odds
    odds_df = parse_odds_response(api_response)
    
    if odds_df.empty:
        print("⚠️ No odds data returned from API")
        return pd.DataFrame(), pd.DataFrame()
    
    # Match predictions to odds
    analysis_df = match_predictions_to_odds(predictions_df, odds_df)
    
    # Calculate EV and edge
    analysis_df = calculate_ev_and_edge(analysis_df)
    
    # Generate report
    report = generate_morning_report(analysis_df, api_response['timestamp'])
    print(report)
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    with open(f'reports/morning_snapshot_{timestamp}.txt', 'w') as f:
        f.write(report)
    
    # Save data
    analysis_df.to_csv(f'reports/morning_analysis_{timestamp}.csv', index=False)
    
    return analysis_df, odds_df

def run_prerace_snapshot(predictions_df: pd.DataFrame, 
                        morning_odds_df: pd.DataFrame,
                        race_time: datetime,
                        minutes_before: int = 15) -> pd.DataFrame:
    """
    Run pre-race snapshot (T-15 minutes)
    """
    print("\n" + "="*80)
    print(f"RUNNING PRE-RACE SNAPSHOT (T-{minutes_before})")
    print("="*80 + "\n")
    
    # Calculate time window
    target_time = race_time - timedelta(minutes=minutes_before)
    window_start = race_time - timedelta(minutes=30)
    window_end = race_time + timedelta(minutes=30)
    
    # Fetch odds for specific time window
    api_response = fetch_all_races(
        commence_time_from=window_start.isoformat() + 'Z',
        commence_time_to=window_end.isoformat() + 'Z'
    )
    
    # Parse odds
    odds_df = parse_odds_response(api_response)
    
    if odds_df.empty:
        print("⚠️ No odds data returned from API")
        return pd.DataFrame()
    
    # Match predictions to odds
    analysis_df = match_predictions_to_odds(predictions_df, odds_df)
    
    # Calculate EV and edge
    analysis_df = calculate_ev_and_edge(analysis_df)
    
    # Generate report with CLV
    report = generate_prerace_report(analysis_df, morning_odds_df, api_response['timestamp'])
    print(report)
    
    # Save report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    with open(f'reports/prerace_snapshot_{timestamp}.txt', 'w') as f:
        f.write(report)
    
    analysis_df.to_csv(f'reports/prerace_analysis_{timestamp}.csv', index=False)
    
    return analysis_df

# ============================================================
#                    MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Load predictions
    predictions_file = "Inference_Outputs/Output_2026-01-11.csv"
    predictions_df = load_predictions(predictions_file)
    
    # Run morning snapshot
    morning_analysis, morning_odds = run_morning_snapshot(
        predictions_df, 
        target_date="2026-01-11"
    )
    
    # Identify high-value races for T-15 monitoring
    if not morning_analysis.empty:
        value_races = identify_value_bets(morning_analysis)
        
        if not value_races.empty:
            print(f"\n✓ {len(value_races)} value opportunities identified")
            print(f"✓ Monitoring {value_races['race_key'].nunique()} races for pre-race updates")
            
            # Example: Run pre-race snapshot for first high-value race
            # In production, you'd schedule these based on race times
            first_race_time = value_races.iloc[0]['race_datetime_pred']
            
            print(f"\n[Example] Running pre-race analysis for {first_race_time}")
            prerace_analysis = run_prerace_snapshot(
                predictions_df,
                morning_odds,
                first_race_time
            )
    
    print(f"\n✓ Complete! Total API requests used: {requests_used}/{MONTHLY_REQUEST_LIMIT}")
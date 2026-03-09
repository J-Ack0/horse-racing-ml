import pandas as pd
import numpy as np
import re
import joblib
import os
import warnings
from datetime import datetime, timedelta
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

# ============================================================
#                      LOGGING UTILITY
# ============================================================

def write_inference_log(target_date, input_path, total_rows, nan_counts, horse_details, log_dir="Inference_Logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{target_date}_logs.txt")
    
    with open(log_filename, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(f"INFERENCE LOG FOR EVENT DATE: {target_date}\n")
        f.write(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input File: {input_path}\n")
        f.write("="*60 + "\n\n")
        
        f.write("### DATA QUALITY SUMMARY\n")
        f.write(f"Total Entries Processed: {total_rows}\n")
        f.write("Missing Values (NaNs) found in key features (before zero-fill):\n")
        if nan_counts.sum() == 0:
            f.write(" - No missing values found. Data is clean.\n")
        else:
            for col, count in nan_counts.items():
                if count > 0:
                    f.write(f" - {col}: {count}\n")
        f.write("\n")
        
        f.write("### HORSE PREDICTION SUMMARY\n")
        f.write(f"{'Horse Name':<25} | {'Runs Found':<10} | {'Win Prob':<10} | {'Source':<10}\n")
        f.write("-" * 65 + "\n")
        for h in horse_details:
            f.write(f"{h['name'][:24]:<25} | {h['runs']:<10} | {h['prob']:<10.4f} | {h['source']:<10}\n")
            
    print(f"✓ Log file generated: {log_filename}")

# ============================================================
#                      HELPER FUNCTIONS
# ============================================================

def clean_names(name):
    if not isinstance(name, str): return name
    name = name.replace('.', '').strip()
    parts = name.split()
    if len(parts) < 2: return name.title()
    initials = [p[0].upper() + '.' for p in parts[:-1]]
    last_name = parts[-1]
    return ' '.join(initials + [last_name])

def wilson_score(wins, total, confidence=0.95):
    if total == 0: return 0.0
    z = {0.95: 1.96, 0.99: 2.576, 0.90: 1.645}[confidence]
    phat = wins / total
    denominator = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
    return (centre - margin) / denominator

def extract_clean_going(raw_going):
    GOING_SCALE = ["Firm", "Good to Firm", "Good", "Good to Yielding", "Yielding", "Yielding to Soft", "Soft", "Soft to Heavy", "Heavy", "Standard"]
    if not isinstance(raw_going, str): return None
    raw = str(raw_going).lower().replace('-', ' ').strip()
    for going in sorted(GOING_SCALE, key=lambda g: -len(g)):
        pattern = r'\b' + re.escape(going.lower()) + r'\b'
        if re.search(pattern, raw): return going
    return None

def normalize_distance(dist_str: str) -> float:
    dist_str = str(dist_str).strip()
    if dist_str in ['nan', 'NaN', 'N/A', '-', '', 'None']: return 0.0
    pattern = r'^\s*(?:(?P<miles>\d+(?:\.\d+)?)\s*m)?\s*(?:(?P<furlongs>\d+(?:\.\d+)?)\s*f)?\s*(?:(?P<yards>\d+(?:\.\d+)?)\s*y)?\s*$'
    m = re.match(pattern, dist_str, flags=re.IGNORECASE)
    if not m: return 0.0
    miles = float(m.group('miles') or 0)
    furlongs = float(m.group('furlongs') or 0)
    yards = float(m.group('yards') or 0)
    return float(miles * 1760 + furlongs * 220 + yards)

def normalize_weight(w_str: str) -> float:
    """Fixed to handle spaces like '11 -12'"""
    w_str = str(w_str).strip()
    if w_str in ['-', '', 'N/A', 'nan', 'NaN', 'None']: return 126.0
    
    # Remove spaces before attempting regex
    w_str = w_str.replace(' ', '')
    
    stone_pounds_match = re.match(r'^(\d+)-(\d+)', w_str)
    if stone_pounds_match:
        return int(stone_pounds_match.group(1)) * 14 + int(stone_pounds_match.group(2))
    match = re.match(r'^(\d+(?:\.\d+)?)', w_str)
    if match: return float(match.group(1))
    return 126.0

# ============================================================
#                  CORE AUTOMATION PIPELINE
# ============================================================
def run_automated_pipeline(target_date_str, model_path, scaler_path, historic_path, lookup_path):
    input_file = os.path.join("Inference_Inputs", f"Input_{target_date_str}.csv")
    output_dir = "Inference_Outputs"
    output_file = os.path.join(output_dir, f"Output_{target_date_str}.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: No input file found for {target_date_str}")
        return

    os.makedirs(output_dir, exist_ok=True)
    df_new = pd.read_csv(input_file)
    
    # 1. Lookup Loader
    lookup_df = None
    if os.path.exists(lookup_path):
        lookup_df = pd.read_csv(lookup_path)
        if 'race_date' in lookup_df.columns:
            lookup_df['race_date'] = pd.to_datetime(lookup_df['race_date'], format='mixed', errors='coerce')
            lookup_df = lookup_df.sort_values('race_date').groupby('horse_name').last().reset_index()
        lookup_df.set_index('horse_name', inplace=True)

    # 2. History Loader
    df_hist = pd.read_csv(historic_path)
    df_hist['race_date'] = pd.to_datetime(df_hist['race_date'], errors='coerce')
    
    if 'race_position' in df_hist.columns:
        df_hist['label'] = (df_hist['race_position'] == 1).astype(int)
    else:
        df_hist['label'] = 0
    
    df_new['is_new'] = True
    df_hist['is_new'] = False
    df_new['label'] = 0
    
    full_df = pd.concat([df_hist, df_new], ignore_index=True)
    
    # 3. Preprocessing (FIXED: Added Age Bins here to prevent KeyError)
    full_df['distance'] = full_df['distance'].apply(normalize_distance)
    full_df['weight'] = full_df['weight'].apply(normalize_weight)
    full_df['rating'] = pd.to_numeric(full_df['rating'], errors='coerce').fillna(70)
    full_df['age'] = pd.to_numeric(full_df['age'], errors='coerce').fillna(4)
    full_df['claims'] = pd.to_numeric(full_df['claims'], errors='coerce').fillna(0)
    full_df['J_Name'] = full_df['jockey'].apply(clean_names).str.replace('.', '', regex=False)
    full_df['T_Name'] = full_df['trainer'].apply(clean_names).str.replace('.', '', regex=False)
    full_df['j_t'] = full_df['J_Name'].astype(str) + " | " + full_df['T_Name'].astype(str)
    full_df['going_clean'] = full_df['going'].apply(extract_clean_going)

    # --- AGE BIN CREATION (Moved here for safety) ---
    full_df['is_young_horse'] = (full_df['age'] <= 3).astype(int)
    full_df['is_prime_age'] = ((full_df['age'] >= 4) & (full_df['age'] <= 6)).astype(int)
    full_df['is_veteran'] = (full_df['age'] >= 7).astype(int)

    # 4. Contextual Features
    full_df['jt_runs'] = full_df.groupby('j_t').cumcount()
    full_df['jt_wins'] = full_df.groupby('j_t')['label'].cumsum() - full_df['label']
    full_df['wilson_score'] = full_df.apply(lambda row: wilson_score(row['jt_wins'], row['jt_runs']), axis=1)
    
    full_df['track_stripped'] = full_df['track_name'].str.split('|').str[0].str.strip()
    full_df['track_total_runs'] = full_df.groupby('track_stripped').cumcount()
    full_df['track_total_wins'] = full_df.groupby('track_stripped')['label'].cumsum() - full_df['label']
    
    full_df['track_win_rate'] = np.where(full_df['track_total_runs'] > 0, full_df['track_total_wins'] / full_df['track_total_runs'], 0.0)
    full_df['track_cum_distance'] = full_df.groupby('track_stripped')['distance'].cumsum() - full_df['distance']
    
    median_distance = full_df['distance'].median()
    full_df['track_avg_distance'] = np.where(full_df['track_total_runs'] > 0, full_df['track_cum_distance'] / full_df['track_total_runs'], median_distance)

    # 5. Horse Stats & History Tracking
    pred_idx = full_df[full_df['is_new']].index
    horse_cols = ['EMA_Form', 'performance_on_going', 'days_since_last', 'recent_win_rate', 
                  'career_wins', 'career_races', 'career_win_rate']
    for c in horse_cols:
        if c not in full_df.columns:
            full_df[c] = 0.0

    log_horse_data = []
    # Lists to store history info for final output
    hist_counts = []
    is_found_list = []

    print("Processing Horse Stats...")
    for idx in tqdm(pred_idx):
        h_name = full_df.at[idx, 'horse_name']
        going = full_df.at[idx, 'going_clean']
        source = "Fallback"
        runs_found = 0
        
        # Priority 1: Check Lookup Index
        if lookup_df is not None and h_name in lookup_df.index:
            for col in horse_cols:
                if col in lookup_df.columns:
                    val = lookup_df.loc[h_name, col]
                    full_df.at[idx, col] = val if pd.notna(val) else 0.0
            source = "Lookup"
            runs_found = int(lookup_df.loc[h_name, 'career_races']) if 'career_races' in lookup_df.columns else 0
        
        # Priority 2: Check Historical File
        else:
            h_history = full_df[(full_df['horse_name'] == h_name) & (~full_df['is_new'])].sort_values('race_date')

            if not h_history.empty:
                runs_found = len(h_history)
                full_df.at[idx, 'career_wins'] = h_history['label'].sum()
                full_df.at[idx, 'career_races'] = runs_found
                full_df.at[idx, 'career_win_rate'] = full_df.at[idx, 'career_wins'] / runs_found
                full_df.at[idx, 'recent_win_rate'] = h_history['label'].tail(5).mean()
                
                g_hist = h_history[h_history['going_clean'] == going]
                full_df.at[idx, 'performance_on_going'] = g_hist['label'].mean() if not g_hist.empty else 0.0
                
                if pd.notna(h_history['race_date'].iloc[-1]):
                    target_race_date = full_df.at[idx, 'race_date']
                    last_race_date = h_history['race_date'].iloc[-1]
                    days_diff = (pd.to_datetime(target_race_date) - pd.to_datetime(last_race_date)).days
                    full_df.at[idx, 'days_since_last'] = max(0, days_diff)
                
                full_df.at[idx, 'EMA_Form'] = 0.5 
            else:
                full_df.at[idx, 'EMA_Form'] = 0.3
                full_df.at[idx, 'days_since_last'] = 365
                runs_found = 0
        
        # Save history tracking data
        hist_counts.append(runs_found)
        is_found_list.append(runs_found > 0)
        log_horse_data.append({"name": h_name, "runs": runs_found, "source": source})

    # 6. Prediction
    feature_cols = [
        'EMA_Form', 'wilson_score', 'jt_runs', 'jt_wins',
        'performance_on_going',
        'days_since_last', 'track_win_rate', 'track_avg_distance',
        'is_young_horse', 'is_prime_age', 'is_veteran', 
        'recent_win_rate', 'career_wins', 'career_races', 'career_win_rate',
        'distance', 'claims', 'age', 'weight', 'rating'
    ]
    
    X = full_df.loc[pred_idx, feature_cols]
    
    nan_summary = X.isna().sum()
    X = X.fillna({
        'EMA_Form': 0.3, 'wilson_score': 0.0, 'jt_runs': 0, 'jt_wins': 0,
        'performance_on_going': 0.0, 'days_since_last': 365, 'track_win_rate': 0.0,
        'track_avg_distance': median_distance, 'is_young_horse': 0, 'is_prime_age': 1,
        'is_veteran': 0, 'recent_win_rate': 0.0, 'career_wins': 0, 'career_races': 0,
        'career_win_rate': 0.0, 'distance': 0.0, 'claims': 0, 'age': 4, 'weight': 126.0, 'rating': 70
    })
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    # 7. Finalize Output
    final_output = df_new.copy()
    final_output['win_probability'] = probs
    
    # NEW: Add history metadata to output CSV
    final_output['history_count'] = hist_counts
    final_output['is_found_in_history'] = is_found_list

    for i, p in enumerate(probs):
        log_horse_data[i]['prob'] = p

    final_output = final_output.sort_values(['track_name', 'win_probability'], ascending=[True, False])
    final_output.to_csv(output_file, index=False)
    
    write_inference_log(target_date_str, input_file, len(df_new), nan_summary, log_horse_data)
    print(f"✓ Success! Predictions saved to: {output_file}")

# ============================================================
#                      EXECUTION LOGIC
# ============================================================

if __name__ == "__main__":
    tomorrow = datetime.now() + timedelta(days=1)
    tomorrow_str = tomorrow.strftime("%Y-%m-%d")
    
    PARAMS = {
        "target_date_str": tomorrow_str,
        "model_path": "Pt2/models/xgb_model.joblib",
        "scaler_path": "Pt2/models/xgb_scaler.joblib",
        "historic_path": "merged_v2_horses_labeled.csv",
        "lookup_path": "V2_engineered_features.csv"
    }
    
    run_automated_pipeline(**PARAMS)
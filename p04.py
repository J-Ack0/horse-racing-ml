#XGB MODEL

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier  # Added XGBoost
import joblib  # Added for saving model
# import torch # Removed TabNet
import scipy.stats as stats
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm 
import os
 # <-- NEW: Import tqdm
warnings.filterwarnings('ignore')

# ============================================================
#                    FEATURE ENGINEERING
# ============================================================
# (All your feature engineering functions: clean_names, wilson_score,
# extract_clean_going, normalize_distance, calculate_form_features,
# normalize_weight, and process_features remain here, unchanged)

def clean_names(name):
    """Clean jockey/trainer names to standard format"""
    if not isinstance(name, str):
        return name
    
    name = name.replace('.', '').strip()
    parts = name.split()
    
    if len(parts) < 2:
        return name.title()
    
    initials = [p[0].upper() + '.' for p in parts[:-1]]
    last_name = parts[-1]
    
    return ' '.join(initials + [last_name])

def wilson_score(wins, total, confidence=0.95):
    """Calculate Wilson score for confidence interval"""
    if total == 0:
        return 0.0
    z = {0.95: 1.96, 0.99: 2.576, 0.90: 1.645}[confidence]
    phat = wins / total
    denominator = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
    return (centre - margin) / denominator

def extract_clean_going(raw_going):
    """Extract standardized going condition"""
    GOING_SCALE = [
        "Firm", "Good to Firm", "Good", "Good to Yielding",
        "Yielding", "Yielding to Soft", "Soft", "Soft to Heavy",
        "Heavy", "Standard"
    ]
    
    if not isinstance(raw_going, str):
        return None

    raw = raw_going.lower().replace('-', ' ').strip()

    for going in sorted(GOING_SCALE, key=lambda g: -len(g)):
        pattern = r'\b' + re.escape(going.lower()) + r'\b'
        if re.search(pattern, raw):
            return going

    tokens = re.findall(r'\b\w+\b', raw)
    for token in tokens:
        for going in GOING_SCALE:
            if token == going.lower():
                return going

    return None

def normalize_distance(dist_str: str) -> float:
    """
    Convert a horse-race distance string (e.g. '2m 6f 74y') into total yards.
    Handles NaN and missing values.
    """
    dist_str = str(dist_str).strip()
    
    # Handle missing/invalid values
    if dist_str in ['nan', 'NaN', 'N/A', '-', '', 'None']:
        return np.nan
    
    pattern = (
    r'^\s*'
    r'(?:(?P<miles>\d+(?:\.\d+)?)\s*m)?\s*'
    r'(?:(?P<furlongs>\d+(?:\.\d+)?)\s*f)?\s*'
    r'(?:(?P<yards>\d+(?:\.\d+)?)\s*y)?'
    r'\s*$'
)
    
    m = re.match(pattern, dist_str, flags=re.IGNORECASE)
    if not m:
        print(f"Warning: Invalid distance format: '{dist_str}' - returning NaN")
        return np.nan
    
    miles = float(m.group('miles') or 0)
    furlongs = float(m.group('furlongs') or 0)
    yards = float(m.group('yards') or 0)
    return float(miles * 1760 + furlongs * 220 + yards)

def calculate_form_features(races):
    """Calculate EMA form features for a horse's race history"""
    if len(races) < 2:
        return {}
    
    races = races.sort_values('race_date_clean').reset_index(drop=True)
    
    form_scores = []
    for _, race in races.iterrows():
        if pd.isna(race['race_position_clean']) or pd.isna(race['rating_clean']):
            form_scores.append(np.nan)
            continue
            
        try:
            position = float(race['race_position_clean'])
            rating = float(race['rating_clean']) if not pd.isna(race['rating_clean']) else 70
            
            norm_pos = max(0, 1 - (position - 1) / 19)
            norm_rating = rating / 130
            
            form_score = 0.6 * norm_pos + 0.4 * norm_rating
            form_scores.append(form_score)
        except:
            form_scores.append(np.nan)
    
    features = {}
    if len(form_scores) >= 2:
        ema = form_scores[0] if not pd.isna(form_scores[0]) else 0.5
        for score in form_scores[1:]:
            if not pd.isna(score):
                ema = 0.3 * score + 0.7 * ema
        features['EMA_Form'] = ema
    
    return features

def normalize_weight(w_str: str) -> float:
    """
    Normalize British weight format to pounds.
    
    Handles formats like:
    - '10-13' → 10 stone 13 pounds → 153 pounds
    - '11-0tp' → 11 stone 0 pounds → 154 pounds (ignoring suffix)
    - '10-13tb1' → 10 stone 13 pounds → 153 pounds (ignoring suffix)
    - '-' or empty → returns np.nan
    """
    w_str = str(w_str).strip()
    
    if w_str in ['-', '', 'N/A', 'nan', 'NaN', 'None']:
        return np.nan
    
    # British stone-pounds format: "stone-pounds" optionally followed by letters
    stone_pounds_match = re.match(r'^(\d+)-(\d+)', w_str)
    if stone_pounds_match:
        stone = int(stone_pounds_match.group(1))
        pounds = int(stone_pounds_match.group(2))
        return stone * 14 + pounds  # 1 stone = 14 pounds
    
    # Fallback: extract leading number (assuming already in pounds)
    match = re.match(r'^(\d+(?:\.\d+)?)', w_str)
    if match:
        return float(match.group(1))
    
    print(f"Warning: Invalid weight format: '{w_str}' - returning NaN")
    return np.nan

def process_features(df):
    """Main feature processing pipeline - Rewritten for point-in-time (non-leaky) features"""
    print("Starting feature engineering...")

    def _extract_race_date(track_name):
        """Extract race date from track_name column"""
        if not isinstance(track_name, str):
            return None
        
        # Isolate the date part of the string
        new = track_name.split('|', 1)[-1]
        
        # Clean string: "22nd June 2024 16:05" -> "22 June 2024 16:05"
        cleaned_string = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', new).strip()

        try:
            # Try parsing with time
            date_object = datetime.strptime(cleaned_string, "%d %B %Y %H:%M")
            return date_object
        except ValueError:
            try:
                # Try parsing without time (if format is "22 June 2024")
                date_object = datetime.strptime(cleaned_string, "%d %B %Y")
                return date_object
            except ValueError:
                # Handle other potential formats or return None
                return None

    print("Parsing and sorting by race date...")
    # Use your new parser
    df['race_date_clean'] = df['track_name'].apply(_extract_race_date)
    
    # Drop rows where date couldn't be parsed
    df = df.dropna(subset=['race_date_clean'])
    
    # CRITICAL: Sort by date and reset the index
    # All future calculations depend on this temporal order.
    df = df.sort_values('race_date_clean').reset_index(drop=True)
    print(f"Sorted {len(df)} rows by date.")

    # ============================================================
    # 1. ROW-WISE FEATURES (NO LEAKAGE)
    # These operations only look at the current row.
    # ============================================================

    if 'distance' in df.columns:
        print("Normalizing distance values...")
        df['distance'] = df['distance'].apply(normalize_distance)
    
    if 'weight' in df.columns:
        print("Normalizing weight values...")
        df['weight'] = df['weight'].apply(normalize_weight)

    numeric_columns = ['age', 'claims']
    if 'rating_clean' in df.columns:
        numeric_columns.append('rating_clean')
    elif 'rating' in df.columns:
        numeric_columns.append('rating')
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print("Applying slow .apply() operations (tracking progress)...")
    tqdm.pandas(desc="Cleaning Names") 
    df['J_Name'] = df['jockey'].progress_apply(clean_names).str.replace('.','', regex=False)
    df['T_Name'] = df['trainer'].progress_apply(clean_names).str.replace('.','', regex=False)
    
    df['track_stripped'] = df['track_name'].str.split('|').str[0].str.strip()
    df['j_t'] = df['J_Name'].astype(str) + " | " + df['T_Name'].astype(str)
    
    # Create label
    df['label'] = (df['race_position_clean'] == 1).astype(int)
    
    tqdm.pandas(desc="Clean Going")
    df['going'] = df['going'].str.split('(').str[0].str.strip()
    df['going'] = df['going'].progress_apply(extract_clean_going)

    # ============================================================
    # 2. POINT-IN-TIME (NON-LEAKY) AGGREGATE FEATURES
    # These use groupby() + cumsum() / cumcount() on the sorted df.
    # This calculates stats using ONLY data from rows *before* the current one.
    # ============================================================

    print("Calculating point-in-time jockey-trainer stats...")
    # 'cumcount()' is the number of *previous* runs (0 for the first run)
    df['jt_runs'] = df.groupby('j_t').cumcount()
    
    # 'cumsum() - df['label']' is the sum of wins *before* the current race
    df['jt_wins'] = df.groupby('j_t')['label'].cumsum() - df['label']
    
    print("Calculating point-in-time wilson score...")
    tqdm.pandas(desc="Wilson Score (Non-Leaky)")
    df['wilson_score'] = df.progress_apply(
        lambda row: wilson_score(row['jt_wins'], row['jt_runs']),
        axis=1
    )
    
    print("Calculating point-in-time performance on going...")
    # Calculate wins/runs for each horse on each specific going type
    df['horse_going_runs'] = df.groupby(['horse_name', 'going']).cumcount()
    df['horse_going_wins'] = df.groupby(['horse_name', 'going'])['label'].cumsum() - df['label']
    
    # Calculate the horse's *past* win rate on the *current* going
    df['performance_on_going'] = (df['horse_going_wins'] / df['horse_going_runs']).fillna(0)
    
    print("Calculating point-in-time track-specific features...")
    # Total past runs at this track
    df['track_total_runs'] = df.groupby('track_stripped').cumcount()
    
    # Total past wins at this track
    df['track_total_wins'] = df.groupby('track_stripped')['label'].cumsum() - df['label']
    
    # Past win rate at this track (using a safe 0 for the first race)
    df['track_win_rate'] = (df['track_total_wins'] / df['track_total_runs']).fillna(0)
    
    # Past average distance at this track
    # .cumsum() - df['distance'] gives cumulative *past* distance
    df['track_cum_distance'] = df.groupby('track_stripped')['distance'].cumsum() - df['distance']
    df['track_avg_distance'] = (df['track_cum_distance'] / df['track_total_runs']).fillna(df['distance'].median()) # fillna with median


    # ============================================================
    # 3. CORRECTLY IMPLEMENTED ROLLING FEATURES (YOURS)
    # These loops were already correct and non-leaky.
    # ============================================================
    
    print("Calculating EMA Form...")
    df['EMA_Form'] = 0.5  # Default value
    # This loop is correct because it passes temporally-sorted data
    # to calculate_form_features, which then calculates a rolling EMA.
    for horse in tqdm(df['horse_name'].unique(), desc="EMA Form"):
        # The main df is already sorted, so horse_races will be too
        horse_races = df[df['horse_name'] == horse].copy()
        if len(horse_races) >= 2:
            form_features = calculate_form_features(horse_races)
            if 'EMA_Form' in form_features:
                df.loc[df['horse_name'] == horse, 'EMA_Form'] = form_features['EMA_Form']
    
    print("Calculating historical performance features...")
    # df is already sorted by 'race_date_clean'
    
    df['recent_win_rate'] = 0.0
    df['career_wins'] = 0
    df['career_races'] = 0
    df['career_win_rate'] = 0.0
    df['days_since_last'] = 0 # Note: This feature is not calculated in your loop
    
    # This loop is perfectly non-leaky. It iterates through each horse's
    # sorted races and only looks at 'prior_indices'.
    for horse in tqdm(df['horse_name'].unique(), desc="Career Stats"):
        horse_mask = df['horse_name'] == horse
        horse_indices = df[horse_mask].index.tolist() # Indices are already sorted
        
        for i, idx in enumerate(horse_indices):
            if i > 0:
                prior_indices = horse_indices[:i]
                prior_labels = df.loc[prior_indices, 'label']
                
                career_wins = prior_labels.sum()
                career_races = len(prior_labels)
                career_win_rate = career_wins / career_races if career_races > 0 else 0
                
                recent_labels = prior_labels.tail(5)
                recent_win_rate = recent_labels.mean() if len(recent_labels) > 0 else 0
                
                df.loc[idx, 'recent_win_rate'] = recent_win_rate
                df.loc[idx, 'career_wins'] = career_wins
                df.loc[idx, 'career_races'] = career_races
                df.loc[idx, 'career_win_rate'] = career_win_rate

    # ============================================================
    # 4. ROW-WISE FEATURES (POST-CALCULATION)
    # ============================================================
    
    # Removed leaky 'weight_percentile'. The raw 'weight' feature
    # is already included in 'raw_features' by prepare_data_for_tabnet
    
    if 'age' in df.columns:
        df['is_young_horse'] = (df['age'] <= 3).astype(int)
        df['is_prime_age'] = ((df['age'] >= 4) & (df['age'] <= 6)).astype(int)
        df['is_veteran'] = (df['age'] >= 7).astype(int)
    
    # Drop any rows that still have NaNs in critical columns (post-processing)
    final_cols = numeric_columns + ['distance', 'weight', 'wilson_score', 'recent_win_rate', 'career_win_rate']
    final_cols = [c for c in final_cols if c in df.columns]
    df = df.dropna(subset=final_cols)
    print(f"Final dataset size after feature engineering and NaN removal: {len(df)}")
    
    print("Feature engineering complete!")
    return df
# ============================================================
#                    DATA PREPARATION
# ============================================================

def prepare_data_for_tabnet(df): # Renamed function, was prepare_data_for_tabnet
    """Prepare data for model training"""
    
    # Define feature columns
    high_correlation_features = [
        'EMA_Form', 'wilson_score', 'jt_runs', 'jt_wins'
    ]
    
    # Add performance_on_{going_type} features
    going_features = [col for col in df.columns if col.startswith('performance_on_')]
    high_correlation_features.extend(going_features)
    
    additional_features = [
        'days_since_last', 'track_win_rate', 'track_avg_distance',
        'weight_percentile', 'is_young_horse', 'is_prime_age', 'is_veteran',
        'recent_win_rate', 'career_wins', 'career_races', 'career_win_rate'
    ]
    
    # Raw numeric features
    raw_features = ['distance', 'claims', 'age', 'weight']
    # Add rating_clean or rating if available
    if 'rating_clean' in df.columns:
        raw_features.append('rating_clean')
    elif 'rating' in df.columns:
        raw_features.append('rating')
    
    # Check which features exist in the dataframe
    all_features = []
    for feat in high_correlation_features + additional_features + raw_features:
        if feat in df.columns:
            all_features.append(feat)
    
    print(f"Total features for training: {len(all_features)}")
    print(f"High correlation features: {[f for f in high_correlation_features if f in df.columns]}")
    
    # Handle missing values
    for col in all_features:
        if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
            # Use median for numeric columns, but handle empty columns
            if df[col].notna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
    
    # Prepare feature matrix
    df = df.sort_values('race_date_clean')
    X = df[all_features].values
    y = df['label'].values
    
    # Create attention mask (1 for high correlation features, 0 for others)
    # This is specific to TabNet, but we pass it along. XGBoost will ignore it.
    attention_mask = np.zeros(len(all_features))
    for i, feat in enumerate(all_features):
        if feat in high_correlation_features:
            attention_mask[i] = 1
    
    return X, y, all_features, attention_mask

# ============================================================
#                    XGBOOST TRAINING
# ============================================================

def train_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """
    Train XGBoost model and use validation set for early stopping.
    
    Args:
        X_train, y_train: Training data and labels.
        X_val, y_val: Validation data and labels for early stopping.
        feature_names: List of feature names (for verbose output).
        
    Returns:
        Trained XGBClassifier model and the fitted StandardScaler.
    """
    
    print("\nStarting XGBoost training...")
    
    # 1. Scaling Numeric Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 2. XGBoost Parameters
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'use_label_encoder': False, # Suppress warning
        'n_estimators': 2000,       # Max number of boosting rounds
        'learning_rate': 0.01,      # Step size shrinkage
        'max_depth': 5,             # Max depth of a tree
        'subsample': 0.8,           # Subsample ratio of the training instance
        'colsample_bytree': 0.8,    # Subsample ratio of columns when constructing each tree
        'random_state': 42,
        'tree_method': 'hist',      # Use histogram-based algorithm for speed
        'n_jobs': -1                # Use all available cores
    }
    
    # 3. Model Initialization
    model = XGBClassifier(**xgb_params)
    
    # 4. Model Training with Early Stopping
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=10                 # Print metrics every 100 boosting rounds
    )
    
    print(f"\nXGBoost training finished in sum boosting rounds.")
    print(f"Best Validation AUC: {model.score}")
    
    return model, scaler

# ============================================================
#                    UPDATED MAIN PIPELINE
# ============================================================

# ============================================================
#                    UPDATED MAIN PIPELINE
# ============================================================

def main_xgboost():
    """Main training pipeline with XGBoost and threshold testing"""
    
    # Check if engineered features already exist
    engineered_features_path = "Pt2/engineered_features.csv"
    
    if os.path.exists(engineered_features_path):
        print(f"Found existing engineered features at '{engineered_features_path}'")
        print("Loading pre-computed features...")
        try:
            df = pd.read_csv(engineered_features_path)
            print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
            
            # Verify essential columns exist
            if 'label' not in df.columns:
                print("Error: 'label' column not found in engineered features.")
                print("Recomputing features from scratch...")
                df = pd.read_csv("Pt2/merged_horse_racing_data.csv")
                df = process_features(df)
                df.to_csv(engineered_features_path, index=False)
                print("✓ Features recomputed and saved!")
        except Exception as e:
            print(f"Error loading engineered features: {e}")
            print("Attempting to recompute from raw data...")
            df = pd.read_csv("Pt2/merged_horse_racing_data.csv")
            df = process_features(df)
            df.to_csv(engineered_features_path, index=False)
            print("✓ Features recomputed and saved!")
    else:
        # Load raw data
        print("Loading raw data...")
        try:
            df = pd.read_csv("Pt2/merged_horse_racing_data.csv")
        except FileNotFoundError:
            print("Error: 'Pt2/merged_horse_racing_data.csv' not found.")
            print("Please ensure the data file is in the correct directory.")
            return None, None, None, None, None
        
        # Process features
        print("Computing features (this will take ~15 minutes)...")
        df = process_features(df)
        print(f"\nSaving engineered features to '{engineered_features_path}'...")
        df.to_csv(engineered_features_path, index=False)
        print("✓ Features saved!")
    
    # ============================================================
    #                    THRESHOLD TESTING & ANALYSIS
    # ============================================================
    # (All your threshold testing functions: test_thresholds, 
    # plot_precision_recall_curve, find_optimal_threshold, 
    # and print_threshold_analysis remain here, unchanged)
    #
    # --- START OF THE FIX ---
    # The code below was missing. It needs to run *after* 'df' is loaded/created.
    
    print("\nPreparing data for modeling...")
    # XGBoost doesn't use the attention_mask, so we can ignore it with '_'
    X, y, feature_names, _ = prepare_data_for_tabnet(df) 

    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Positive class (wins) rate: {y.mean():.2%}")

    # 1. Split into Train+Val and Test sets (e.g., 80% Train+Val, 20% Test)
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 2. Split Train+Val into Train and Validation sets (e.g., 80/20 split of the 80%)
        # This makes the validation set 20% of 80% = 16% of the total data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
        )
    except ValueError as e:
        print(f"\nWarning: Stratification failed (likely too few samples of one class): {e}")
        print("Splitting data without stratification...")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=42
        )

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # 3. Train the XGBoost model
    model, scaler = train_xgboost(X_train, y_train, X_val, y_val, feature_names)

    # 4. Save the model and scaler
    print("\nSaving model and scaler...")
    os.makedirs("Pt2/models", exist_ok=True)
    joblib.dump(model, "Pt2/models/xgb_model.joblib")
    joblib.dump(scaler, "Pt2/models/xgb_scaler.joblib")
    print("✓ Model and scaler saved to 'Pt2/models/'")

    # 5. Evaluate on the Test set
    print("\nEvaluating model on hold-out test set...")
    X_test_scaled = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] # Probabilities for the positive class (1)
    
    # 6. Analyze Thresholds
    threshold_results = test_thresholds(y_test, y_proba)
    
    # Plot Precision-Recall Curve
    plot_precision_recall_curve(y_test, y_proba, save_path='Pt2/xgb_precision_recall.png')
    print("✓ Precision-Recall curve saved to 'Pt2/xgb_precision_recall.png'")
    
    # Print detailed threshold analysis
    optimal_thresholds = print_threshold_analysis(threshold_results, y_test, y_proba)

    # 7. Get Feature Importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    print("\nTOP 20 FEATURES:")
    print(importance_df.head(20).to_string())
    importance_df.to_csv("Pt2/xgb_feature_importance.csv", index=False)
    print("\n✓ Feature importance saved to 'Pt2/xgb_feature_importance.csv'")

    # 8. Return all the expected components
    # This now matches what the __main__ block expects
    return model, feature_names, importance_df, threshold_results, optimal_thresholds
    
    # --- END OF THE FIX ---



# ============================================================
#                    THRESHOLD TESTING & ANALYSIS
# ============================================================
# (All your threshold testing functions: test_thresholds, 
# plot_precision_recall_curve, find_optimal_threshold, 
# and print_threshold_analysis remain here, unchanged)

def test_thresholds(y_true, y_proba, thresholds=None):
    """
    Test different probability thresholds and calculate metrics
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        thresholds: List of thresholds to test (default: 0.1 to 0.9 in steps of 0.05)
    
    Returns:
        DataFrame with threshold analysis results
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Positive predictive value and negative predictive value
        ppv = precision  # Same as precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'predicted_positive': tp + fp,
            'predicted_negative': tn + fn
        })
    
    return pd.DataFrame(results)

def plot_precision_recall_curve(y_true, y_proba, save_path='precision_recall_curve.png'):
    """
    Plot precision-recall curve
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)
    
    plt.figure(figsize=(10, 8))
    
    # Main precision-recall curve
    plt.subplot(2, 1, 1)
    plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precision and recall vs threshold
    plt.subplot(2, 1, 2)
    # Note: thresholds array is one element shorter than precision/recall
    plt.plot(thresholds, precision[:-1], label='Precision', linewidth=2)
    plt.plot(thresholds, recall[:-1], label='Recall', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return avg_precision

def find_optimal_threshold(threshold_results, metric='f1_score'):
    """
    Find optimal threshold based on specified metric
    
    Args:
        threshold_results: DataFrame from test_thresholds function
        metric: Metric to optimize ('f1_score', 'precision', 'recall', 'accuracy')
    
    Returns:
        Optimal threshold value and corresponding metrics
    """
    optimal_idx = threshold_results[metric].idxmax()
    optimal_row = threshold_results.loc[optimal_idx]
    
    return optimal_row['threshold'], optimal_row

def print_threshold_analysis(threshold_results, y_true, y_proba):
    """Print comprehensive threshold analysis"""
    print("\n" + "="*80)
    print("                    THRESHOLD ANALYSIS RESULTS")
    print("="*80)
    
    # Find optimal thresholds for different metrics
    metrics = ['f1_score', 'precision', 'recall', 'accuracy']
    optimal_thresholds = {}
    
    for metric in metrics:
        threshold, row = find_optimal_threshold(threshold_results, metric)
        optimal_thresholds[metric] = (threshold, row)
    
    # Print optimal thresholds summary
    print("\nOPTIMAL THRESHOLDS BY METRIC:")
    print("-" * 50)
    for metric, (threshold, row) in optimal_thresholds.items():
        print(f"{metric.upper():12}: {threshold:.3f} "
              f"(P={row['precision']:.3f}, R={row['recall']:.3f}, "
              f"F1={row['f1_score']:.3f}, Acc={row['accuracy']:.3f})")
    
    # Print detailed results for key thresholds
    print(f"\nDETAILED THRESHOLD ANALYSIS:")
    print("-" * 50)
    key_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print(f"{'Thresh':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Acc':>6} "
          f"{'Spec':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'Pred+':>5}")
    print("-" * 80)
    
    for threshold in key_thresholds:
        # Rounding to handle floating point inaccuracies
        row = threshold_results[np.isclose(threshold_results['threshold'], threshold)]
        if not row.empty:
            row = row.iloc[0]
            print(f"{row['threshold']:6.2f} {row['precision']:6.3f} {row['recall']:6.3f} "
                  f"{row['f1_score']:6.3f} {row['accuracy']:6.3f} {row['specificity']:6.3f} "
                  f"{int(row['tp']):4d} {int(row['fp']):4d} {int(row['tn']):4d} {int(row['fn']):4d} "
                  f"{int(row['predicted_positive']):5d}")
    
    # Calculate baseline metrics
    baseline_precision = np.mean(y_true)  # Random precision would be proportion of positives
    print(f"\nBASELINE METRICS:")
    print("-" * 50)
    print(f"Random Baseline Precision: {baseline_precision:.3f}")
    print(f"Average Precision Score: {average_precision_score(y_true, y_proba):.3f}")
    print(f"AUC-ROC Score: {roc_auc_score(y_true, y_proba):.3f}")
    
    # Recommendations
    print(f"\nRECOMMENDations:")
    print("-" * 50)
    f1_threshold = optimal_thresholds['f1_score'][0]
    precision_threshold = optimal_thresholds['precision'][0]
    recall_threshold = optimal_thresholds['recall'][0]
    
    print(f"• For balanced performance: Use threshold {f1_threshold:.3f} (optimizes F1-score)")
    print(f"• For high precision (fewer false positives): Use threshold {precision_threshold:.3f}")
    print(f"• For high recall (catch more winners): Use threshold {recall_threshold:.3f}")
    
    return optimal_thresholds

# ============================================================
#                    SCRIPT EXECUTION
# ============================================================
if __name__ == "__main__":
    # Change the call to main_xgboost to run the new pipeline
    results = main_xgboost()

    # --- FIX APPLIED HERE ---
    # Check if results is None (indicating a failure/early exit in main_xgboost)
    # if results is None:
    #     print("\nPipeline execution failed, likely due to missing data file or an error in main_xgboost.")
    # # If results is not None, proceed to check if all its components are not None
    if all(r is not None for r in results):
        model, features, importance, threshold_results, optimal_thresholds = results
        print("\nPipeline execution successful.")
    else:
        # This handles the case where results is an iterable, but some items are None
        print("\nPipeline execution failed, likely due to missing data file or partially missing results.")
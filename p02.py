import re
import pandas as pd
import numpy as np
import glob
import os
import pickle
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
# Global flag for pretraining availability
PRETRAINING_AVAILABLE = False
try:
    from pytorch_tabnet.pretraining import TabNetPretrainer
    PRETRAINING_AVAILABLE = True
    print("✅ TabNetPretrainer available - pretraining enabled")
except ImportError:
    print("⚠️ TabNetPretrainer not available - will skip pretraining")
    PRETRAINING_AVAILABLE = False
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

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

def normalize_lost_by_length(lost_str: str) -> float:
    """
    Convert lost by length string to numeric value.
    
    Examples:
    - 'N/A' → 0.0 (winner)
    - '7 l' → 7.0
    - '9 1/2 l' → 9.5
    - '1 1/4 l' → 1.25
    - 'Nose' → 0.05
    - 'Head' → 0.1
    - 'Neck' → 0.2
    - 'Short Head' → 0.08
    """
    lost_str = str(lost_str).strip().lower()
    
    if lost_str in ['n/a', '-', '', 'nan', 'none']:
        return 0.0  # Assume winners have 0 loss
    
    # Handle racing terminology
    if 'nose' in lost_str:
        return 0.05  # Very small margin
    elif 'head' in lost_str:
        if 'short' in lost_str:
            return 0.08
        else:
            return 0.1   # Small margin
    elif 'neck' in lost_str:
        return 0.2
    elif 'nk' in lost_str:  # Common abbreviation for neck
        return 0.2
    elif 'shd' in lost_str:  # Short head abbreviation
        return 0.08
    elif 'hd' in lost_str:   # Head abbreviation
        return 0.1
    
    # Remove 'l' suffix and clean
    lost_str = re.sub(r'\s*l\s*$', '', lost_str, flags=re.IGNORECASE)
    
    # Handle fractions like "9 1/2" or "1/4"
    fraction_match = re.match(r'^(\d+)?\s*(\d+)/(\d+)$', lost_str)
    if fraction_match:
        whole = int(fraction_match.group(1) or 0)
        numerator = int(fraction_match.group(2))
        denominator = int(fraction_match.group(3))
        return whole + (numerator / denominator)
    
    # Handle pure numbers
    number_match = re.match(r'^(\d+(?:\.\d+)?)$', lost_str)
    if number_match:
        return float(number_match.group(1))
    
    print(f"Warning: Unrecognized lost_by_length format: '{lost_str}' - returning 0.0")
    return 0.0

def clean_names(name):
    """
    Clean jockey/trainer names to standardized format.
    From the EDA notebook - converts names to initials + last name format.
    """
    if not isinstance(name, str):
        return name
    
    name = name.replace('.', '').strip()
    parts = name.split()
    
    if len(parts) < 2:
        return name.title()
    
    initials = [p[0].upper() + '.' for p in parts[:-1]]
    
    # Keep last name as-is (preserves apostrophes, caps)
    last_name = parts[-1]
    
    return ' '.join(initials + [last_name])

def wilson_score(wins, total, confidence=0.95):
    """
    Calculate Wilson score for confidence interval of success rate.
    From the EDA notebook - provides a better estimate for win rate with small samples.
    """
    if total == 0:
        return 0.0
    z = {0.95: 1.96, 0.99: 2.576, 0.90: 1.645}[confidence]
    phat = wins / total
    denominator = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
    return (centre - margin) / denominator

def safe_numeric_convert(series, column_name):
    """
    Safely convert a series to numeric, handling various missing value formats.
    """
    # Replace common missing value representations
    series = series.astype(str).replace({
        'nan': np.nan, 'NaN': np.nan, 'N/A': np.nan, 
        '-': np.nan, '': np.nan, 'None': np.nan
    })
    
    # Convert to numeric
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # Report conversion stats
    nan_count = numeric_series.isna().sum()
    total_count = len(numeric_series)
    if nan_count > 0:
        print(f"  {column_name}: {nan_count}/{total_count} values converted to NaN")
    
    return numeric_series

def load_and_combine_data(batch_folder="batch"):
    """Load and combine all CSV files from the batch folder."""
    print("Loading all CSV files from batch folder...")
    csv_files = glob.glob(os.path.join(batch_folder, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {batch_folder} folder!")
    
    print(f"Found {len(csv_files)} CSV files:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Load and concatenate all CSV files
    dfs = []
    for file in csv_files:
        try:
            df_temp = pd.read_csv(file)
            df_temp['source_file'] = os.path.basename(file)  # Track source file
            dfs.append(df_temp)
            print(f"  Loaded {len(df_temp)} rows from {os.path.basename(file)}")
        except Exception as e:
            print(f"  ERROR loading {file}: {e}")
    
    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    print(f"\nTotal rows after combining and deduplicating: {len(df)}")
    
    return df, len(csv_files)

def clean_and_prepare_data(df):
    """Clean and prepare the data for model training."""
    print("Normalizing features...")
    
    # Safely convert and normalize distance
    print("Processing distance...")
    df['distance'] = df['distance'].astype(str).apply(normalize_distance)
    distance_nan_count = df['distance'].isna().sum()
    if distance_nan_count > 0:
        print(f"  Distance: {distance_nan_count} NaN values found, filling with median")
        df['distance'] = df['distance'].fillna(df['distance'].median())
    
    # Safely convert and normalize weight  
    print("Processing weight...")
    df['weight'] = df['weight'].astype(str).apply(normalize_weight)
    weight_nan_count = df['weight'].isna().sum()
    if weight_nan_count > 0:
        print(f"  Weight: {weight_nan_count} NaN values found, filling with median")
        df['weight'] = df['weight'].fillna(df['weight'].median())
    
    # Ensure other numeric columns are properly converted
    print("Processing other numeric columns...")
    base_numeric_cols = ['claims', 'age', 'rating']
    for col in base_numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric_convert(df[col], col)
    
    # Fill remaining NaN values in numeric columns with 0 or median
    numeric_cols = [col for col in base_numeric_cols if col in df.columns]
    for col in numeric_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            fill_value = df[col].median() if df[col].median() > 0 else 0
            df[col] = df[col].fillna(fill_value)
            print(f"  {col}: filled {nan_count} NaN values with {fill_value}")
    
    # Handle categorical columns - convert NaN to string 'missing'
    print("Processing categorical columns...")
    cat_cols = ['track_name', 'race_time', 'going', 'horse_name', 'jockey', 'trainer', 'race_position']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({'nan': 'missing', 'NaN': 'missing', 'None': 'missing'})
            missing_count = (df[col] == 'missing').sum()
            if missing_count > 0:
                print(f"  {col}: {missing_count} missing values marked as 'missing'")
    
    return df

def add_jockey_trainer_features(df):
    """
    Add jockey-trainer combination features from the EDA notebook.
    This includes cleaned names, jockey-trainer pairs, wins, runs, and Wilson scores.
    """
    print("\n🏇 Adding Jockey-Trainer Combination Features...")
    
    # First create the label if it doesn't exist
    if 'label' not in df.columns:
        df['label'] = (df['race_position'] == '1st').astype(int)
    
    # Clean jockey and trainer names
    print("Cleaning jockey and trainer names...")
    df['J_Name'] = df['jockey'].apply(clean_names).str.replace('.','', regex=False)
    df['T_Name'] = df['trainer'].apply(clean_names).str.replace('.','', regex=False)
    
    # Clean track names
    df['track_stripped'] = df['track_name'].str.split('|').str[0].str.strip()
    
    # Create jockey-trainer combination
    df['j_t'] = df['J_Name'].astype(str) + " | " + df['T_Name'].astype(str)
    print(f"Created {df['j_t'].nunique()} unique jockey-trainer combinations")
    
    # Calculate JT Wins (actual wins only)
    print("Calculating jockey-trainer wins...")
    jtw = df[df['label'] == 1].groupby('j_t').size().reset_index(name='jt_wins')
    df = df.merge(jtw, on='j_t', how='left')
    df['jt_wins'] = df['jt_wins'].fillna(0).astype(int)
    
    # Calculate Total JT Runs
    print("Calculating jockey-trainer total runs...")
    jt_summary = df['j_t'].value_counts().reset_index()
    jt_summary.columns = ['j_t', 'jt_runs']
    df = df.merge(jt_summary, on='j_t', how='left')
    
    # Calculate Wilson Score for JT combinations
    print("Calculating Wilson scores...")
    df['jt_wilson_score'] = df.apply(
        lambda row: wilson_score(row['jt_wins'], row['jt_runs']),
        axis=1
    )
    
    # Calculate individual jockey statistics
    print("Calculating individual jockey statistics...")
    jockey_wins = df[df['label'] == 1].groupby('J_Name').size().reset_index(name='jockey_wins')
    jockey_runs = df['J_Name'].value_counts().reset_index()
    jockey_runs.columns = ['J_Name', 'jockey_runs']
    
    df = df.merge(jockey_wins, on='J_Name', how='left')
    df = df.merge(jockey_runs, on='J_Name', how='left')
    df['jockey_wins'] = df['jockey_wins'].fillna(0).astype(int)
    
    df['jockey_wilson_score'] = df.apply(
        lambda row: wilson_score(row['jockey_wins'], row['jockey_runs']),
        axis=1
    )
    
    # Calculate individual trainer statistics
    print("Calculating individual trainer statistics...")
    trainer_wins = df[df['label'] == 1].groupby('T_Name').size().reset_index(name='trainer_wins')
    trainer_runs = df['T_Name'].value_counts().reset_index()
    trainer_runs.columns = ['T_Name', 'trainer_runs']
    
    df = df.merge(trainer_wins, on='T_Name', how='left')
    df = df.merge(trainer_runs, on='T_Name', how='left')
    df['trainer_wins'] = df['trainer_wins'].fillna(0).astype(int)
    
    df['trainer_wilson_score'] = df.apply(
        lambda row: wilson_score(row['trainer_wins'], row['trainer_runs']),
        axis=1
    )
    
    # Add win rates as percentages
    df['jt_win_rate'] = (df['jt_wins'] / df['jt_runs']).fillna(0)
    df['jockey_win_rate_clean'] = (df['jockey_wins'] / df['jockey_runs']).fillna(0)
    df['trainer_win_rate_clean'] = (df['trainer_wins'] / df['trainer_runs']).fillna(0)
    
    print(f"✅ Added JT features: jt_wins, jt_runs, jt_wilson_score, individual jockey/trainer stats")
    print(f"   Top JT combination: {df.nlargest(1, 'jt_wilson_score')[['j_t', 'jt_wilson_score']].values[0]}")
    
    return df

def add_advanced_features(df):
    """
    Add more sophisticated features for better prediction
    """
    # Class quality features
    if 'claims' in df.columns:
        df['race_class'] = pd.cut(df['claims'], 
                                 bins=[0, 5000, 15000, 50000, float('inf')], 
                                 labels=['maiden', 'low_class', 'mid_class', 'high_class'])
        # Convert to numeric encoding
        df['race_class_encoded'] = df['race_class'].cat.codes
    
    # Weather/track condition interactions
    if 'going' in df.columns and 'horse_name' in df.columns:
        # Horse performance on different track conditions
        for going_type in df['going'].unique():
            going_mask = df['going'] == going_type
            horse_going_performance = df[going_mask].groupby('horse_name')['label'].mean()
            df[f'performance_on_{going_type}'] = df['horse_name'].map(horse_going_performance).fillna(0)
    
    return df

def create_additional_features(df):
    """Create additional legitimate features for horse racing prediction."""
    print("Creating additional features...")
    
    # First create the label column we'll need for historical features
    if 'label' not in df.columns:
        df['label'] = (df['race_position'] == '1st').astype(int)
    
    # Historical performance features (if we have past data)
    if 'horse_name' in df.columns and 'race_date' in df.columns:
        # Sort by date first (if race_date is available and parseable)
        try:
            df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')
            df = df.sort_values('race_date')
        except:
            # If race_date parsing fails, sort by index as proxy
            df = df.sort_index()
        
        # Initialize new feature columns
        df['recent_win_rate'] = 0.0
        df['career_wins'] = 0
        df['career_races'] = 0
        df['career_win_rate'] = 0.0
        df['days_since_last'] = 0
        
        # For each horse, calculate recent performance metrics
        for horse in df['horse_name'].unique():
            horse_mask = df['horse_name'] == horse
            horse_indices = df[horse_mask].index.tolist()
            
            # Calculate rolling statistics for each race
            for i, idx in enumerate(horse_indices):
                if i > 0:  # Skip first race (no prior history)
                    # Get prior races for this horse
                    prior_indices = horse_indices[:i]
                    prior_labels = df.loc[prior_indices, 'label']
                    
                    # Calculate features based on prior races only
                    career_wins = prior_labels.sum()
                    career_races = len(prior_labels)
                    career_win_rate = career_wins / career_races if career_races > 0 else 0
                    
                    # Recent win rate (last 5 races)
                    recent_labels = prior_labels.tail(5) if len(prior_labels) >= 5 else prior_labels
                    recent_win_rate = recent_labels.mean() if len(recent_labels) > 0 else 0
                    
                    # Update the dataframe
                    df.loc[idx, 'recent_win_rate'] = recent_win_rate
                    df.loc[idx, 'career_wins'] = career_wins
                    df.loc[idx, 'career_races'] = career_races
                    df.loc[idx, 'career_win_rate'] = career_win_rate
        
        print(f"  Added historical performance features for {df['horse_name'].nunique()} horses")
    
    # Track-specific features (using cleaned track names)
    if 'track_stripped' in df.columns:
        track_stats = []
        
        for idx in df.index:
            track = df.loc[idx, 'track_stripped']
            
            # Get other races at this track (excluding current race)
            other_races = df[
                (df['track_stripped'] == track) & 
                (df.index != idx)
            ]
            
            if len(other_races) >= 10:  # Only use if enough history
                track_win_rate = other_races['label'].mean()
                track_avg_distance = other_races['distance'].mean()
            else:
                track_win_rate = df['label'].mean()
                track_avg_distance = df['distance'].mean()
            
            track_stats.append({
                'index': idx,
                'track_win_rate': track_win_rate,
                'track_avg_distance': track_avg_distance
            })
        
        stats_df = pd.DataFrame(track_stats).set_index('index')
        df['track_win_rate'] = stats_df['track_win_rate']
        df['track_avg_distance'] = stats_df['track_avg_distance']
        print(f"  Added features: track_win_rate, track_avg_distance")
    
    # Weight-related features
    if 'weight' in df.columns:
        df['weight_percentile'] = df['weight'].rank(pct=True)
        print(f"  Added feature: weight_percentile")
    
    # Age-related features
    if 'age' in df.columns:
        df['is_young_horse'] = (df['age'] <= 3).astype(int)
        df['is_prime_age'] = ((df['age'] >= 4) & (df['age'] <= 6)).astype(int)
        df['is_veteran'] = (df['age'] >= 7).astype(int)
        print(f"  Added features: is_young_horse, is_prime_age, is_veteran")
    
    return df

def prepare_features_and_labels(df):
    """Prepare features and labels for training."""
    # Label should already be created
    if 'label' not in df.columns:
        df['label'] = (df['race_position'] == '1st').astype(int)
    
    winners_count = df['label'].sum()
    total_races = len(df)
    winner_rate = df['label'].mean()
    
    print(f"Found {winners_count} winners out of {total_races} races ({winner_rate:.1%})")
    
    # Drop leakage columns and any we don't need for features
    # Also drop the intermediate name columns since we'll use the cleaned versions
    drop_cols = ['race_position', 'label', 'race_date', 'first_place_found', 
                 'batch_number', 'url_index_in_batch', 'source_file', 'lost_by_length', 
                 'race_class', 'jockey', 'trainer']  # Drop original jockey/trainer since we have cleaned versions
    
    feature_df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    print(f"🚨 REMOVED 'lost_by_length' and other leakage columns to prevent data leakage")
    
    # Define feature specs - now including the new JT features
    available_cat_features = ['track_name', 'race_time', 'going', 'horse_name', 
                            'J_Name', 'T_Name', 'j_t', 'track_stripped']
    
    available_num_features = ['distance', 'claims', 'age', 'weight', 'rating',
                             'recent_win_rate', 'career_wins', 'career_races', 'career_win_rate', 
                             'days_since_last', 'track_win_rate', 'track_avg_distance', 
                             'weight_percentile', 'is_young_horse', 'is_prime_age', 'is_veteran', 
                             'race_class_encoded',
                             # New JT features
                             'jt_wins', 'jt_runs', 'jt_wilson_score',
                             'jockey_wins', 'jockey_runs', 'jockey_wilson_score',
                             'trainer_wins', 'trainer_runs', 'trainer_wilson_score',
                             'jt_win_rate', 'jockey_win_rate_clean', 'trainer_win_rate_clean']
    
    # Add any performance_on_* columns
    performance_cols = [col for col in feature_df.columns if col.startswith('performance_on_')]
    available_num_features.extend(performance_cols)
    
    cat_features = [col for col in available_cat_features if col in feature_df.columns]
    num_features = [col for col in available_num_features if col in feature_df.columns]
    
    print(f"Categorical features ({len(cat_features)}): {cat_features}")
    print(f"Numerical features ({len(num_features)}): {num_features}")
    
    return feature_df, cat_features, num_features, df['label']

def encode_and_scale_features(feature_df, cat_features, num_features):
    """Encode categorical features and scale numerical features."""
    # Final missing value check
    print("Final missing value check...")
    for col in cat_features:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].astype(str).fillna('missing')
            feature_df[col] = feature_df[col].replace({'nan': 'missing', 'NaN': 'missing', 'None': 'missing'})
    
    for col in num_features:
        if col in feature_df.columns:
            nan_count = feature_df[col].isna().sum()
            if nan_count > 0:
                fill_value = feature_df[col].median() if feature_df[col].median() > 0 else 0
                feature_df[col] = feature_df[col].fillna(fill_value)
                print(f"  Final check - {col}: filled {nan_count} remaining NaN values")
            
            # Ensure no infinite values
            inf_count = np.isinf(feature_df[col]).sum()
            if inf_count > 0:
                feature_df[col] = feature_df[col].replace([np.inf, -np.inf], feature_df[col].median())
                print(f"  {col}: replaced {inf_count} infinite values")
    
    # Encode categoricals
    print("Encoding categorical features...")
    encoders = {}
    for col in cat_features:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))
        encoders[col] = le
        print(f"  {col}: {len(le.classes_)} unique values")
    
    # Scale numericals
    print("Scaling numerical features...")
    scaler = MinMaxScaler()
    if num_features:
        feature_df[num_features] = scaler.fit_transform(feature_df[num_features])
    
    # Final data quality check
    print("\nFinal data quality check:")
    total_missing = feature_df.isnull().sum().sum()
    total_infinite = np.isinf(feature_df.select_dtypes(include=[np.number])).sum().sum()
    print(f"  Total missing values: {total_missing}")
    print(f"  Total infinite values: {total_infinite}")
    
    if total_missing > 0 or total_infinite > 0:
        print("Warning: Still have missing or infinite values - cleaning up...")
        feature_df = feature_df.fillna(0)
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
        feature_df[numeric_columns] = feature_df[numeric_columns].replace([np.inf, -np.inf], 0)
        print("  Cleaned remaining issues")
    
    print(f"Final feature dataframe shape: {feature_df.shape}")
    
    return feature_df, encoders, scaler

def train_tabnet_model(X_train, y_train, X_val, y_val, cat_features, feature_df, X_pretrain=None):
    """Train TabNet model with pretraining and grouped attention on Wilson scores."""
    print("\n🧠 ENHANCED TRAINING WITH WILSON SCORE ATTENTION")
    print("="*60)
    
    # Calculate extreme class weights - heavily penalize missing winners
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    
    # Make the penalty for missing winners even MORE extreme
    winner_weight_multiplier = 10  # Make this higher for even more aggressive winner detection
    class_weights[1] *= winner_weight_multiplier  # Amplify winner weight
    
    print(f"Class weights: No-win={class_weights[0]:.2f}, WIN={class_weights[1]:.2f}")
    print(f"Winner weight is {class_weights[1]/class_weights[0]:.1f}x higher than non-winner weight")
    
    # Create sample weights for training
    sample_weights = np.where(y_train == 1, class_weights[1], class_weights[0])
    
    # Identify categorical indices and dims for TabNet
    print("\nPreparing TabNet parameters...")
    cat_idxs = []
    cat_dims = []
    
    for col in cat_features:
        if col in feature_df.columns:
            idx = feature_df.columns.get_loc(col)
            dim = int(feature_df[col].nunique())
            cat_idxs.append(idx)
            cat_dims.append(dim)
            print(f"  {col}: index={idx}, unique_values={dim}")
    
    # GROUPED ATTENTION: Identify Wilson score related features
    print("\n🎯 Setting up GROUPED ATTENTION for Wilson score features...")
    wilson_features = ['jt_wilson_score', 'jockey_wilson_score', 'trainer_wilson_score',
                      'jt_wins', 'jt_runs', 'jockey_wins', 'jockey_runs', 
                      'trainer_wins', 'trainer_runs']
    
    # Create grouped features list - features that should be attended together
    grouped_features = []
    wilson_group = []
    
    for i, col in enumerate(feature_df.columns):
        if col in wilson_features:
            wilson_group.append(i)
    
    if wilson_group:
        grouped_features.append(wilson_group)
        print(f"✅ Wilson score group: {len(wilson_group)} features grouped for attention")
        print(f"   Features in group: {[feature_df.columns[i] for i in wilson_group]}")
    
    print(f"\nCategorical indices: {cat_idxs}")
    print(f"Categorical dimensions: {cat_dims}")
    
    # STEP 1: Pretraining (if requested and available)
    if X_pretrain is not None and PRETRAINING_AVAILABLE:
        print("\n🔄 STEP 1: UNSUPERVISED PRETRAINING")
        print(f"Pretraining on {len(X_pretrain)} samples to learn feature representations...")
        
        try:
            # Try with grouped features first
            pretrainer = TabNetPretrainer(
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                cat_emb_dim=8,
                n_d=64,
                n_a=64,
                n_steps=5,
                gamma=1.5,
                n_independent=2,
                n_shared=2,
                grouped_features=grouped_features,  # Use grouped attention in pretraining too
                mask_type='entmax',  # or 'sparsemax'
                verbose=1,
                seed=42
            )
            print("✅ Using grouped attention in pretraining")
        except TypeError:
            # Fallback without grouped features if not supported
            print("⚠️ Grouped features not supported in pretraining, using standard pretraining")
            pretrainer = TabNetPretrainer(
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                cat_emb_dim=8,
                n_d=64,
                n_a=64,
                n_steps=5,
                gamma=1.5,
                n_independent=2,
                n_shared=2,
                mask_type='entmax',
                verbose=1,
                seed=42
            )
        
        # Pretrain
        pretrainer.fit(
            X_train=X_pretrain.values,
            eval_set=[X_val.values],
            pretraining_ratio=0.8,  # Predict 80% of features
            max_epochs=50,  # Fewer epochs for pretraining
            patience=10,
            batch_size=1024,  # Larger batch for pretraining
            virtual_batch_size=128
        )
        
        print("✅ Pretraining completed!")
    else:
        if not PRETRAINING_AVAILABLE:
            print("\n⚠️ Pretraining skipped (TabNetPretrainer not available)")
        else:
            print("\n⚠️ Pretraining skipped (no pretraining data provided)")
        pretrainer = None
    
    # STEP 2: Supervised training
    print("\n🎯 STEP 2: SUPERVISED TRAINING with grouped attention...")
    
    try:
        # Try with grouped features first
        clf = TabNetClassifier(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=8,      # Increased embedding size
            n_d=64,             # Increased network width
            n_a=64,             # Increased attention width  
            n_steps=5,          # More decision steps
            gamma=1.5,          # Feature selection regularization
            n_independent=2,    # Number of independent GLU layers
            n_shared=2,         # Number of shared GLU layers
            grouped_features=grouped_features,  # CRITICAL: Use grouped attention!
            verbose=1,
            seed=42
        )
        print("✅ Using grouped attention for Wilson score features")
    except TypeError:
        # Fallback without grouped features if not supported
        print("⚠️ Grouped features not supported, using standard TabNet")
        clf = TabNetClassifier(
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=8,
            n_d=64,
            n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            verbose=1,
            seed=42
        )
    
    # Load pretrained weights if available
    if pretrainer is not None:
        try:
            print("📥 Loading pretrained weights into classifier...")
            clf.load_weights_from_pretrainer(pretrainer)
            print("✅ Successfully loaded pretrained weights")
        except AttributeError:
            print("⚠️ load_weights_from_pretrainer not available, starting from scratch")
            # Alternative: manually copy weights if possible
            try:
                # Try to copy the network weights directly
                clf.network = pretrainer.network
                print("✅ Manually copied network weights from pretrainer")
            except:
                print("⚠️ Could not transfer pretrained weights, continuing without pretraining")
    
    # Train with sample weights to heavily penalize missing winners
    print("\n🚀 Final supervised training with winner-focused loss...")
    clf.fit(
        X_train=X_train.values, 
        y_train=y_train.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_name=['validation'],
        eval_metric=['auc'],
        max_epochs=200,     # More epochs for complex patterns
        patience=25,        # More patience for convergence
        batch_size=512,     # Larger batch size
        virtual_batch_size=128,
        weights=sample_weights  # CRITICAL: Apply heavy winner weights!
    )
    
    # Extract feature importance to verify Wilson score attention
    try:
        if hasattr(clf, 'feature_importances_'):
            print("\n📊 Top features by importance (checking Wilson score impact):")
            feature_names = feature_df.columns
            importances = clf.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(10)
            
            for idx, row in feature_importance_df.iterrows():
                marker = "⭐" if row['feature'] in wilson_features else "  "
                print(f"{marker} {row['feature']}: {row['importance']:.4f}")
    except Exception as e:
        print(f"⚠️ Could not extract feature importances: {e}")
    
    return clf

def save_model_and_components(model, encoders, scaler, feature_columns, cat_features, num_features, 
                             auc_score, best_f1_threshold, best_profit_threshold, save_dir="models"):
    """
    Save the trained model and all preprocessing components for later use.
    """
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(save_dir, f"horse_racing_model_{timestamp}")
    os.makedirs(model_save_path, exist_ok=True)
    
    print(f"\n💾 Saving model and components to: {model_save_path}")
    
    # 1. Save the TabNet model
    tabnet_path = os.path.join(model_save_path, "tabnet_model")
    model.save_model(tabnet_path)
    print(f"✅ TabNet model saved to: {tabnet_path}")
    
    # 2. Save encoders
    encoders_path = os.path.join(model_save_path, "encoders.pkl")
    with open(encoders_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"✅ Label encoders saved to: {encoders_path}")
    
    # 3. Save scaler
    scaler_path = os.path.join(model_save_path, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved to: {scaler_path}")
    
    # 4. Save model metadata and configuration
    metadata = {
        'model_type': 'TabNetClassifier',
        'training_timestamp': timestamp,
        'feature_columns': feature_columns,
        'categorical_features': cat_features,
        'numerical_features': num_features,
        'auc_score': float(auc_score),
        'best_f1_threshold': float(best_f1_threshold),
        'best_profit_threshold': float(best_profit_threshold),
        'total_features': len(feature_columns),
        'categorical_feature_count': len(cat_features),
        'numerical_feature_count': len(num_features)
    }
    
    metadata_path = os.path.join(model_save_path, "model_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✅ Model metadata saved to: {metadata_path}")
    
    # 5. Save human-readable summary
    summary_path = os.path.join(model_save_path, "model_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Horse Racing Prediction Model Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Training Date: {timestamp}\n")
        f.write(f"Model Type: TabNetClassifier\n")
        f.write(f"AUC Score: {auc_score:.4f}\n")
        f.write(f"Best F1 Threshold: {best_f1_threshold:.3f}\n")
        f.write(f"Best Profit Threshold: {best_profit_threshold:.3f}\n\n")
        
        f.write(f"Feature Configuration:\n")
        f.write(f"- Total Features: {len(feature_columns)}\n")
        f.write(f"- Categorical Features: {len(cat_features)}\n")
        f.write(f"- Numerical Features: {len(num_features)}\n\n")
        
        f.write(f"Categorical Features:\n")
        for feat in cat_features:
            f.write(f"  - {feat}\n")
        
        f.write(f"\nNumerical Features:\n")
        for feat in num_features:
            f.write(f"  - {feat}\n")
        
        f.write(f"\nAll Features (in processing order):\n")
        for i, feat in enumerate(feature_columns):
            f.write(f"  {i+1:2d}. {feat}\n")
    
    print(f"✅ Model summary saved to: {summary_path}")
    
    print(f"\n🎉 Model successfully saved! Components:")
    print(f"   📁 Main directory: {model_save_path}")
    print(f"   🤖 TabNet model: tabnet_model/")
    print(f"   🏷️  Encoders: encoders.pkl")
    print(f"   📊 Scaler: scaler.pkl")
    print(f"   📋 Metadata: model_metadata.pkl")
    print(f"   📄 Summary: model_summary.txt")
    
    return model_save_path

def plot_threshold_analysis_full_range(y_val, y_proba):
    """
    Plot comprehensive threshold analysis from 0.0 to 1.0 with finer granularity
    """
    # Full range from 0.0 to 1.0 with smaller steps
    thresholds = np.arange(0.0, 1.01, 0.02)  # 0.0 to 1.0 in steps of 0.02
    
    recalls = []
    precisions = []
    f1_scores = []
    accuracies = []
    profits = []
    
    print(f"\nAnalyzing {len(thresholds)} thresholds from 0.00 to 1.00...")
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        # Handle edge cases where no predictions are made
        if y_pred_thresh.sum() == 0:  # No positive predictions
            recall = 0.0
            precision = 0.0
            f1 = 0.0
            profit_rate = 0.0
        else:
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_thresh).ravel()
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate profit assuming 5:1 odds
            total_bets = tp + fp
            profit = (tp * 5) - total_bets if total_bets > 0 else 0
            profit_rate = profit / total_bets if total_bets > 0 else 0
        
        accuracy = (y_pred_thresh == y_val).mean()
        
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        accuracies.append(accuracy)
        profits.append(profit_rate)
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Recall, Precision, F1
    ax1.plot(thresholds, recalls, 'b-', label='Recall (Sensitivity)', linewidth=2)
    ax1.plot(thresholds, precisions, 'r-', label='Precision', linewidth=2)
    ax1.plot(thresholds, f1_scores, 'g-', label='F1 Score', linewidth=2)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Classification Metrics vs Threshold (Full Range)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    
    # Plot 2: Accuracy
    ax2.plot(thresholds, accuracies, 'purple', linewidth=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Threshold (Full Range)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    
    # Plot 3: Profit Rate
    ax3.plot(thresholds, profits, 'orange', linewidth=2)
    ax3.axhline(y=0, color='red', linestyle='--', label='Break-even')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Profit Rate')
    ax3.set_title('Profit Rate vs Threshold (5:1 odds, Full Range)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.0, 1.0)
    
    # Plot 4: Combined normalized view
    ax4.plot(thresholds, recalls, 'b-', label='Recall', alpha=0.7)
    ax4.plot(thresholds, precisions, 'r-', label='Precision', alpha=0.7)
    ax4.plot(thresholds, f1_scores, 'g-', label='F1', alpha=0.7)
    # Normalize profit rate to 0-1 range for comparison
    profit_normalized = np.array(profits) / max(profits) if max(profits) > 0 else np.array(profits)
    ax4.plot(thresholds, profit_normalized, 'orange', label='Profit Rate (Normalized)', alpha=0.7)
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Normalized Score')
    ax4.set_title('All Metrics Combined (Full Range, Normalized)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.0, 1.0)
    ax4.set_ylim(0.0, 1.0)
    
    plt.tight_layout()
    plt.savefig('threshold_analysis_full_range.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed results (every 10th threshold to avoid spam)
    print("\nDetailed Threshold Analysis (every 10th point):")
    print("Threshold | Recall | Precision | F1 Score | Accuracy | Profit Rate")
    print("-" * 70)
    for i in range(0, len(thresholds), 10):  # Every 10th point
        thresh = thresholds[i]
        print(f"{thresh:9.2f} | {recalls[i]:6.3f} | {precisions[i]:9.3f} | {f1_scores[i]:8.3f} | {accuracies[i]:8.3f} | {profits[i]:11.3f}")
    
    # Find optimal thresholds
    best_f1_idx = np.argmax(f1_scores)
    best_profit_idx = np.argmax(profits)
    
    print(f"\nBest F1 Score: {f1_scores[best_f1_idx]:.3f} at threshold {thresholds[best_f1_idx]:.2f}")
    print(f"Best Profit Rate: {profits[best_profit_idx]:.3f} at threshold {thresholds[best_profit_idx]:.2f}")
    
    return {
        'thresholds': thresholds,
        'recalls': recalls,
        'precisions': precisions,
        'f1_scores': f1_scores,
        'accuracies': accuracies,
        'profits': profits,
        'best_f1_threshold': thresholds[best_f1_idx],
        'best_profit_threshold': thresholds[best_profit_idx],
        'best_f1_score': f1_scores[best_f1_idx],
        'best_profit_rate': profits[best_profit_idx]
    }

def main():
    """
    Main function to orchestrate the complete horse racing prediction pipeline.
    
    Enhanced with:
    - Jockey-Trainer combination features with Wilson scores
    - Optional unsupervised pretraining on full dataset (if available)
    - Grouped attention mechanism for Wilson score features (if supported)
    """
    print("🏇 Starting Horse Racing Prediction Pipeline with Jockey-Trainer Features...")
    print("=" * 60)
    
    try:
        # Step 1: Load and combine data
        print("\n📂 STEP 1: Loading and combining data...")
        df, num_files = load_and_combine_data("batch")
        print(f"✅ Successfully loaded {len(df)} rows from {num_files} files")
        
        # Step 2: Clean and prepare data
        print("\n🧹 STEP 2: Cleaning and preparing data...")
        df = clean_and_prepare_data(df)
        print("✅ Data cleaning completed")
        
        # Step 3: Add jockey-trainer combination features (NEW!)
        print("\n🤝 STEP 3: Adding jockey-trainer combination features...")
        df = add_jockey_trainer_features(df)
        print("✅ Jockey-trainer features added")
        
        # Step 4: Create additional features (includes creating label column)
        print("\n🔧 STEP 4: Creating additional features...")
        df = create_additional_features(df)
        print("✅ Additional features created")
        
        # Step 5: Add advanced features (requires label column to exist)
        print("\n⚡ STEP 5: Adding advanced features...")
        df = add_advanced_features(df)
        print("✅ Advanced features added")
        
        # Step 6: Prepare features and labels
        print("\n🎯 STEP 6: Preparing features and labels...")
        feature_df, cat_features, num_features, labels = prepare_features_and_labels(df)
        print(f"✅ Features prepared: {len(cat_features)} categorical, {len(num_features)} numerical")
        
        # Step 7: Encode and scale features
        print("\n🔄 STEP 7: Encoding and scaling features...")
        feature_df, encoders, scaler = encode_and_scale_features(feature_df, cat_features, num_features)
        print("✅ Feature encoding and scaling completed")
        
        # Step 8: Split data
        print("\n✂️ STEP 8: Splitting data into train/validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            feature_df, labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )
        
        print(f"Training set: {len(X_train)} samples ({y_train.sum()} winners, {y_train.mean():.1%} win rate)")
        print(f"Validation set: {len(X_val)} samples ({y_val.sum()} winners, {y_val.mean():.1%} win rate)")
        
        # Prepare pretraining data (using all available data)
        print("\n🔄 Preparing pretraining data...")
        # Option to disable pretraining by setting to False
        try:
            USE_PRETRAINING = PRETRAINING_AVAILABLE  # Automatically use if available
        except NameError:
            USE_PRETRAINING = False  # Default to False if not defined
            
        if USE_PRETRAINING:
            X_pretrain = feature_df  # Use all data for unsupervised pretraining
            print(f"Pretraining dataset: {len(X_pretrain)} samples (entire dataset)")
        else:
            X_pretrain = None
            print("Skipping pretraining (not available or disabled)")
        
        # Step 9: Train the model with pretraining and grouped attention
        print("\n🤖 STEP 9: Training TabNet model with Wilson score attention...")
        model = train_tabnet_model(X_train, y_train, X_val, y_val, cat_features, feature_df, X_pretrain)
        print("✅ Model training completed")
        
        # Step 10: Make predictions and evaluate
        print("\n📊 STEP 10: Making predictions and evaluating...")
        y_proba = model.predict_proba(X_val.values)[:, 1]  # Get probability of winning
        
        # Basic metrics at default threshold (0.5)
        y_pred_50 = (y_proba >= 0.5).astype(int)
        accuracy_50 = accuracy_score(y_val, y_pred_50)
        auc_score = roc_auc_score(y_val, y_proba)
        
        print(f"AUC Score: {auc_score:.4f}")
        print(f"Accuracy at 0.5 threshold: {accuracy_50:.4f}")
        
        # Classification report at 0.5 threshold
        print("\nClassification Report (threshold = 0.5):")
        print(classification_report(y_val, y_pred_50, target_names=['No Win', 'Win']))
        
        # Step 11: Threshold analysis
        print("\n📈 STEP 11: Running threshold analysis...")
        threshold_results = plot_threshold_analysis_full_range(y_val, y_proba)
        best_f1_threshold = threshold_results['best_f1_threshold']
        best_profit_threshold = threshold_results['best_profit_threshold']

        # Evaluate at optimal thresholds
        print(f"\n🎯 Results at optimal F1 threshold ({best_f1_threshold:.2f}):")
        y_pred_f1 = (y_proba >= best_f1_threshold).astype(int)
        print(classification_report(y_val, y_pred_f1, target_names=['No Win', 'Win']))
        
        print(f"\n💰 Results at optimal profit threshold ({best_profit_threshold:.2f}):")
        y_pred_profit = (y_proba >= best_profit_threshold).astype(int)
        print(classification_report(y_val, y_pred_profit, target_names=['No Win', 'Win']))
        
        # Step 12: Save the model and components
        print("\n💾 STEP 12: Saving model and components...")
        model_save_path = save_model_and_components(
            model=model,
            encoders=encoders,
            scaler=scaler,
            feature_columns=feature_df.columns.tolist(),
            cat_features=cat_features,
            num_features=num_features,
            auc_score=auc_score,
            best_f1_threshold=best_f1_threshold,
            best_profit_threshold=best_profit_threshold
        )
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            print("\n🔍 Top 15 Most Important Features:")
            feature_names = feature_df.columns
            importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(15)
            
            for idx, row in feature_importance_df.iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Summary
        print("\n" + "=" * 60)
        print("🏁 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📈 Final Model Performance:")
        print(f"   • AUC Score: {auc_score:.4f}")
        print(f"   • Best F1 Threshold: {best_f1_threshold:.2f}")
        print(f"   • Best Profit Threshold: {best_profit_threshold:.2f}")
        print(f"   • Total Races Analyzed: {len(df)}")
        print(f"   • Features Used: {len(feature_df.columns)}")
        print(f"   • Unique Jockey-Trainer Combos: {df['j_t'].nunique()}")
        print(f"\n🧠 Training Enhancements Applied:")
        try:
            if PRETRAINING_AVAILABLE and X_pretrain is not None:
                print(f"   ✅ Unsupervised pretraining on full dataset")
            else:
                print(f"   ⚠️  Pretraining skipped (not available)")
        except NameError:
            print(f"   ⚠️  Pretraining status unknown")
        print(f"   • Grouped attention on Wilson score features (if supported)")
        print(f"   • Heavy class weighting for winner detection")
        print(f"   • Jockey-Trainer combination features with Wilson scores")
        print(f"\n💾 Files saved:")
        print(f"   • Model: {model_save_path}")
        print(f"   • Threshold analysis: 'threshold_analysis_full_range.png'")
        
        return {
            'model': model,
            'encoders': encoders,
            'scaler': scaler,
            'feature_columns': feature_df.columns.tolist(),
            'cat_features': cat_features,
            'num_features': num_features,
            'auc_score': auc_score,
            'best_f1_threshold': best_f1_threshold,
            'best_profit_threshold': best_profit_threshold,
            'model_save_path': model_save_path
        }
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("Make sure you have CSV files in the 'batch' folder!")
        return None
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Run the main pipeline
    results = main()
    
    if results:
        print(f"\n🎉 All done! Model and analysis results are ready.")
        print(f"Use the returned 'results' dictionary to access the trained model and parameters.")
    else:
        print(f"\n💥 Pipeline failed. Check the error messages above.")
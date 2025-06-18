import re
import pandas as pd
import numpy as np
import os
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import TabNet
from pytorch_tabnet.tab_model import TabNetClassifier

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

def clean_names(name):
    """
    Clean jockey/trainer names to standardized format.
    """
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
    """
    Calculate Wilson score for confidence interval of success rate.
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

def clean_and_prepare_data(df):
    """Clean and prepare the data - same as training script."""
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

def add_jockey_trainer_features_from_historical(df, historical_df):
    """
    Add jockey-trainer combination features using only historical data.
    This prevents data leakage by not using future race results.
    """
    print("\n🏇 Adding Jockey-Trainer Features from Historical Data...")
    
    # Clean jockey and trainer names in both datasets
    print("Cleaning jockey and trainer names...")
    df['J_Name'] = df['jockey'].apply(clean_names).str.replace('.','', regex=False)
    df['T_Name'] = df['trainer'].apply(clean_names).str.replace('.','', regex=False)
    
    historical_df['J_Name'] = historical_df['jockey'].apply(clean_names).str.replace('.','', regex=False)
    historical_df['T_Name'] = historical_df['trainer'].apply(clean_names).str.replace('.','', regex=False)
    
    # Clean track names
    df['track_stripped'] = df['track_name'].str.split('|').str[0].str.strip()
    historical_df['track_stripped'] = historical_df['track_name'].str.split('|').str[0].str.strip()
    
    # Create jockey-trainer combination
    df['j_t'] = df['J_Name'].astype(str) + " | " + df['T_Name'].astype(str)
    historical_df['j_t'] = historical_df['J_Name'].astype(str) + " | " + historical_df['T_Name'].astype(str)
    
    # Create label for historical data
    if 'label' not in historical_df.columns:
        historical_df['label'] = (historical_df['race_position'] == '1st').astype(int)
    
    print(f"Historical data: {len(historical_df)} races, {historical_df['label'].sum()} wins")
    print(f"New race data: {len(df)} races to predict")
    
    # Calculate JT statistics from historical data only
    print("Calculating jockey-trainer wins from historical data...")
    jtw = historical_df[historical_df['label'] == 1].groupby('j_t').size().reset_index(name='jt_wins')
    jt_runs = historical_df['j_t'].value_counts().reset_index()
    jt_runs.columns = ['j_t', 'jt_runs']
    
    # Merge historical stats with new data
    df = df.merge(jtw, on='j_t', how='left')
    df = df.merge(jt_runs, on='j_t', how='left')
    df['jt_wins'] = df['jt_wins'].fillna(0).astype(int)
    df['jt_runs'] = df['jt_runs'].fillna(0).astype(int)
    
    # Calculate Wilson scores
    print("Calculating Wilson scores...")
    df['jt_wilson_score'] = df.apply(
        lambda row: wilson_score(row['jt_wins'], row['jt_runs']),
        axis=1
    )
    
    # Individual jockey statistics from historical data
    print("Calculating individual jockey statistics...")
    jockey_wins = historical_df[historical_df['label'] == 1].groupby('J_Name').size().reset_index(name='jockey_wins')
    jockey_runs = historical_df['J_Name'].value_counts().reset_index()
    jockey_runs.columns = ['J_Name', 'jockey_runs']
    
    df = df.merge(jockey_wins, on='J_Name', how='left')
    df = df.merge(jockey_runs, on='J_Name', how='left')
    df['jockey_wins'] = df['jockey_wins'].fillna(0).astype(int)
    df['jockey_runs'] = df['jockey_runs'].fillna(0).astype(int)
    
    df['jockey_wilson_score'] = df.apply(
        lambda row: wilson_score(row['jockey_wins'], row['jockey_runs']),
        axis=1
    )
    
    # Individual trainer statistics from historical data
    print("Calculating individual trainer statistics...")
    trainer_wins = historical_df[historical_df['label'] == 1].groupby('T_Name').size().reset_index(name='trainer_wins')
    trainer_runs = historical_df['T_Name'].value_counts().reset_index()
    trainer_runs.columns = ['T_Name', 'trainer_runs']
    
    df = df.merge(trainer_wins, on='T_Name', how='left')
    df = df.merge(trainer_runs, on='T_Name', how='left')
    df['trainer_wins'] = df['trainer_wins'].fillna(0).astype(int)
    df['trainer_runs'] = df['trainer_runs'].fillna(0).astype(int)
    
    df['trainer_wilson_score'] = df.apply(
        lambda row: wilson_score(row['trainer_wins'], row['trainer_runs']),
        axis=1
    )
    
    # Add win rates as percentages
    df['jt_win_rate'] = (df['jt_wins'] / df['jt_runs']).fillna(0)
    df['jockey_win_rate_clean'] = (df['jockey_wins'] / df['jockey_runs']).fillna(0)
    df['trainer_win_rate_clean'] = (df['trainer_wins'] / df['trainer_runs']).fillna(0)
    
    print(f"✅ Added JT features based on historical data")
    print(f"   Found historical data for {(df['jt_runs'] > 0).sum()}/{len(df)} jockey-trainer combinations")
    
    return df

def create_additional_features_from_historical(df, historical_df):
    """Create additional features using historical data only."""
    print("Creating additional features from historical data...")
    
    # Horse historical performance from historical data only
    if 'horse_name' in df.columns and 'horse_name' in historical_df.columns:
        print("Adding horse historical performance features...")
        
        # Initialize new feature columns
        df['recent_win_rate'] = 0.0
        df['career_wins'] = 0
        df['career_races'] = 0
        df['career_win_rate'] = 0.0
        
        # For each horse in new data, get its historical performance
        for horse in df['horse_name'].unique():
            horse_historical = historical_df[historical_df['horse_name'] == horse]
            
            if len(horse_historical) > 0:
                # Calculate historical stats
                career_wins = horse_historical['label'].sum()
                career_races = len(horse_historical)
                career_win_rate = career_wins / career_races if career_races > 0 else 0
                
                # Recent win rate (last 5 historical races)
                recent_races = horse_historical.tail(5)
                recent_win_rate = recent_races['label'].mean() if len(recent_races) > 0 else 0
                
                # Update all rows for this horse
                horse_mask = df['horse_name'] == horse
                df.loc[horse_mask, 'recent_win_rate'] = recent_win_rate
                df.loc[horse_mask, 'career_wins'] = career_wins
                df.loc[horse_mask, 'career_races'] = career_races
                df.loc[horse_mask, 'career_win_rate'] = career_win_rate
        
        print(f"  Added historical performance for {(df['career_races'] > 0).sum()}/{len(df)} horses")
    
    # Track-specific features from historical data
    if 'track_stripped' in df.columns and 'track_stripped' in historical_df.columns:
        print("Adding track-specific features...")
        
        track_stats = []
        for idx in df.index:
            track = df.loc[idx, 'track_stripped']
            
            # Get historical races at this track
            track_historical = historical_df[historical_df['track_stripped'] == track]
            
            if len(track_historical) >= 10:  # Only use if enough history
                track_win_rate = track_historical['label'].mean()
                track_avg_distance = track_historical['distance'].mean()
            else:
                # Use overall historical averages
                track_win_rate = historical_df['label'].mean()
                track_avg_distance = historical_df['distance'].mean()
            
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

def add_advanced_features(df):
    """Add advanced features."""
    # Class quality features
    if 'claims' in df.columns:
        df['race_class'] = pd.cut(df['claims'], 
                                 bins=[0, 5000, 15000, 50000, float('inf')], 
                                 labels=['maiden', 'low_class', 'mid_class', 'high_class'])
        # Convert to numeric encoding
        df['race_class_encoded'] = df['race_class'].cat.codes
    
    return df

def load_model_and_components(model_path):
    """Load the saved TabNet model and preprocessing components."""
    print(f"Loading model components from: {model_path}")
    
    # Load metadata
    metadata_path = os.path.join(model_path, "model_metadata.pkl")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load TabNet model
    tabnet_path = os.path.join(model_path, "tabnet_model.zip")
    model = TabNetClassifier()
    model.load_model(tabnet_path)
    
    # Load encoders
    encoders_path = os.path.join(model_path, "encoders.pkl")
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    
    # Load scaler
    scaler_path = os.path.join(model_path, "scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("✅ Model and components loaded successfully")
    print(f"   Model trained: {metadata['training_timestamp']}")
    print(f"   AUC Score: {metadata['auc_score']:.4f}")
    print(f"   Features: {metadata['total_features']} total")
    
    return model, encoders, scaler, metadata

def prepare_features_for_prediction(df, expected_features, cat_features, num_features):
    """Prepare features for prediction, ensuring all expected features are present."""
    print("Preparing features for prediction...")
    
    # Ensure all expected features are present
    missing_features = []
    for feature in expected_features:
        if feature not in df.columns:
            missing_features.append(feature)
            # Add missing feature with default value
            if feature in cat_features:
                df[feature] = 'missing'
            else:
                df[feature] = 0.0
    
    if missing_features:
        print(f"⚠️  Added {len(missing_features)} missing features with default values:")
        for feat in missing_features[:10]:  # Show first 10
            print(f"   - {feat}")
        if len(missing_features) > 10:
            print(f"   ... and {len(missing_features) - 10} more")
    
    # Select only the expected features in the correct order
    feature_df = df[expected_features].copy()
    
    # Final missing value handling
    for col in cat_features:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].astype(str).fillna('missing')
            feature_df[col] = feature_df[col].replace({'nan': 'missing', 'NaN': 'missing', 'None': 'missing'})
    
    for col in num_features:
        if col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
            feature_df[col] = feature_df[col].fillna(0)
            feature_df[col] = feature_df[col].replace([np.inf, -np.inf], 0)
    
    print(f"✅ Feature preparation completed: {feature_df.shape}")
    return feature_df

def encode_and_scale_features_for_prediction(feature_df, encoders, scaler, cat_features, num_features):
    """Apply the same encoding and scaling used during training."""
    print("Applying saved encoders and scaler...")
    
    # Apply label encoders to categorical features
    for col in cat_features:
        if col in feature_df.columns and col in encoders:
            # Handle unseen categories by mapping them to a default value
            le = encoders[col]
            
            # Create a safe transform function
            def safe_transform(values):
                result = []
                for val in values:
                    if val in le.classes_:
                        result.append(le.transform([val])[0])
                    else:
                        # Assign to class 0 for unseen categories
                        result.append(0)
                        if len(result) <= 5:  # Only warn for first few
                            print(f"   Warning: Unseen category '{val}' in {col}, mapped to 0")
                return np.array(result)
            
            feature_df[col] = safe_transform(feature_df[col].astype(str))
    
    # Apply scaler to numerical features
    if num_features:
        available_num_features = [col for col in num_features if col in feature_df.columns]
        if available_num_features:
            feature_df[available_num_features] = scaler.transform(feature_df[available_num_features])
    
    print("✅ Encoding and scaling completed")
    return feature_df

def predict_race_winners(race_data_path, historical_data_path, model_path, 
                        output_path="race_predictions.csv"):
    """
    Main function to predict race winners.
    
    Args:
        race_data_path: Path to CSV file with new race data to predict
        historical_data_path: Path to CSV file with historical race data
        model_path: Path to saved model directory
        output_path: Path to save predictions CSV
    """
    print("🏇 Starting Horse Racing Prediction...")
    print("=" * 60)
    
    try:
        # Step 1: Load model and components
        print("\n📦 STEP 1: Loading saved model...")
        model, encoders, scaler, metadata = load_model_and_components(model_path)
        
        # Step 2: Load data
        print("\n📂 STEP 2: Loading race data...")
        race_df = pd.read_csv(race_data_path)
        print(f"✅ Loaded {len(race_df)} races to predict from {race_data_path}")
        
        print("\n📂 STEP 3: Loading historical data...")
        historical_df = pd.read_csv(historical_data_path)
        print(f"✅ Loaded {len(historical_df)} historical races from {historical_data_path}")
        
        # Step 3: Clean both datasets
        print("\n🧹 STEP 4: Cleaning race data...")
        race_df = clean_and_prepare_data(race_df)
        
        print("\n🧹 STEP 5: Cleaning historical data...")
        historical_df = clean_and_prepare_data(historical_df)
        
        # Step 4: Add features using historical data
        print("\n🤝 STEP 6: Adding jockey-trainer features...")
        race_df = add_jockey_trainer_features_from_historical(race_df, historical_df)
        
        print("\n🔧 STEP 7: Creating additional features...")
        race_df = create_additional_features_from_historical(race_df, historical_df)
        
        print("\n⚡ STEP 8: Adding advanced features...")
        race_df = add_advanced_features(race_df)
        
        # Step 5: Prepare features for prediction
        print("\n🎯 STEP 9: Preparing features for prediction...")
        expected_features = metadata['feature_columns']
        cat_features = metadata['categorical_features']
        num_features = metadata['numerical_features']
        
        feature_df = prepare_features_for_prediction(race_df, expected_features, cat_features, num_features)
        
        # Step 6: Apply encoding and scaling
        print("\n🔄 STEP 10: Applying encoding and scaling...")
        feature_df = encode_and_scale_features_for_prediction(feature_df, encoders, scaler, cat_features, num_features)
        
        # Step 7: Make predictions
        print("\n🔮 STEP 11: Making predictions...")
        
        # Get prediction probabilities
        y_proba = model.predict_proba(feature_df.values)[:, 1]
        
        # Apply optimal thresholds from training
        best_f1_threshold = metadata['best_f1_threshold']
        best_profit_threshold = metadata['best_profit_threshold']
        
        y_pred_f1 = (y_proba >= best_f1_threshold).astype(int)
        y_pred_profit = (y_proba >= best_profit_threshold).astype(int)
        
        print(f"✅ Predictions completed using:")
        print(f"   - F1 optimal threshold: {best_f1_threshold:.3f}")
        print(f"   - Profit optimal threshold: {best_profit_threshold:.3f}")
        
        # Step 8: Prepare results
        print("\n📊 STEP 12: Preparing results...")
        
        # Create results dataframe
        results_df = race_df[['horse_name', 'jockey', 'trainer', 'track_name', 'race_time']].copy()
        results_df['win_probability'] = y_proba
        results_df['predicted_winner_f1'] = y_pred_f1
        results_df['predicted_winner_profit'] = y_pred_profit
        
        # Add some key features for analysis
        if 'jt_wilson_score' in race_df.columns:
            results_df['jt_wilson_score'] = race_df['jt_wilson_score']
        if 'jockey_wilson_score' in race_df.columns:
            results_df['jockey_wilson_score'] = race_df['jockey_wilson_score']
        if 'trainer_wilson_score' in race_df.columns:
            results_df['trainer_wilson_score'] = race_df['trainer_wilson_score']
        
        # Sort by win probability (highest first)
        results_df = results_df.sort_values('win_probability', ascending=False)
        
        # Step 9: Save results
        print(f"\n💾 STEP 13: Saving results to {output_path}...")
        results_df.to_csv(output_path, index=False)
        
        # Summary statistics
        print("\n📈 PREDICTION SUMMARY:")
        print("=" * 40)
        print(f"Total races analyzed: {len(results_df)}")
        print(f"Predicted winners (F1 threshold): {y_pred_f1.sum()}")
        print(f"Predicted winners (Profit threshold): {y_pred_profit.sum()}")
        print(f"Average win probability: {y_proba.mean():.3f}")
        print(f"Max win probability: {y_proba.max():.3f}")
        print(f"Min win probability: {y_proba.min():.3f}")
        
        print(f"\n🏆 TOP 10 MOST LIKELY WINNERS:")
        print("-" * 80)
        top_10 = results_df.head(10)
        for idx, row in top_10.iterrows():
            f1_marker = "🏅" if row['predicted_winner_f1'] else "  "
            profit_marker = "💰" if row['predicted_winner_profit'] else "  "
            print(f"{f1_marker}{profit_marker} {row['horse_name']:<20} | {row['win_probability']:.3f} | {row['jockey']:<15} | {row['trainer']}")
        
        print(f"\n✅ Predictions saved to: {output_path}")
        print(f"🎯 Use F1 threshold predictions for balanced accuracy")
        print(f"💰 Use profit threshold predictions for betting strategy")
        
        return results_df
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: File not found - {e}")
        print("Make sure all file paths are correct!")
        return None
        
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run predictions."""
    # Configuration
    RACE_DATA_PATH = "race_data.csv"
    HISTORICAL_DATA_PATH = "merged_horse_racing_data.csv"
    MODEL_PATH = "models/horse_racing_model_20250615_232845"
    OUTPUT_PATH = "race_predictions.csv"
    
    print("🏇 Horse Racing Prediction Script")
    print("=" * 50)
    print(f"Race data: {RACE_DATA_PATH}")
    print(f"Historical data: {HISTORICAL_DATA_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    
    # Check if files exist
    if not os.path.exists(RACE_DATA_PATH):
        print(f"❌ Race data file not found: {RACE_DATA_PATH}")
        return
    
    if not os.path.exists(HISTORICAL_DATA_PATH):
        print(f"❌ Historical data file not found: {HISTORICAL_DATA_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model directory not found: {MODEL_PATH}")
        return
    
    # Run predictions
    results = predict_race_winners(
        race_data_path=RACE_DATA_PATH,
        historical_data_path=HISTORICAL_DATA_PATH,
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH
    )
    
    if results is not None:
        print(f"\n🎉 Prediction completed successfully!")
        print(f"Results saved to: {OUTPUT_PATH}")
    else:
        print(f"\n💥 Prediction failed. Check error messages above.")

if __name__ == "__main__":
    main()
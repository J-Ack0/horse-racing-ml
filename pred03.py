import pandas as pd
import numpy as np
import re
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# ============================================================
#                    FEATURE ENGINEERING FUNCTIONS
#                    (Same as training script)
# ============================================================

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
    
    # Compile once, with both VERBOSE and IGNORECASE
    pattern = re.compile(r'''
        ^\s*                                  # leading whitespace
        (?:(?P<miles>\d+(?:\.\d+)?)\s*m)?     # optional “Xm” miles
        \s*
        (?:(?P<furlongs>\d+(?:\.\d+)?)\s*f)?  # optional “Yf” furlongs
        \s*
        (?:(?P<yards>\d+(?:\.\d+)?)\s*y?)?    # optional “Z” or “Zy” yards
        \s*$                                  # trailing whitespace
    ''', flags=re.VERBOSE | re.IGNORECASE)
    
    m = pattern.match(dist_str)
    if not m:
        print(f"Warning: Invalid distance format: '{dist_str}' - returning NaN")
        return np.nan
    
    miles    = float(m.group('miles')    or 0)
    furlongs = float(m.group('furlongs') or 0)
    yards    = float(m.group('yards')    or 0)
    
    # 1 mile = 1760 yards, 1 furlong = 220 yards
    return miles * 1760 + furlongs * 220 + yards

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

# ============================================================
#                    PREDICTION PIPELINE
# ============================================================

class HorseRacingPredictor:
    def __init__(self, model_path, historic_data_path):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the saved TabNet model (.zip file)
            historic_data_path: Path to the historic training data CSV
        """
        self.model_path = model_path
        self.historic_data_path = historic_data_path
        self.model = None
        self.historic_data = None
        self.feature_columns = None
        
    def load_model(self):
        """Load the trained TabNet model"""
        print(f"Loading model from {self.model_path}...")
        self.model = TabNetClassifier()
        self.model.load_model(self.model_path)
        print("Model loaded successfully!")

    def load_historic_data(self):
        """Load and process historic data for feature generation"""
        print(f"Loading historic data from {self.historic_data_path}...")
        self.historic_data = pd.read_csv(self.historic_data_path)

        # Clean and prepare historic data
        self.historic_data['J_Name'] = self.historic_data['jockey'].apply(clean_names).str.replace('.','', regex=False)
        self.historic_data['T_Name'] = self.historic_data['trainer'].apply(clean_names).str.replace('.','', regex=False)
        self.historic_data['j_t'] = self.historic_data['J_Name'].astype(str) + " | " + self.historic_data['T_Name'].astype(str)
        self.historic_data['track_stripped'] = self.historic_data['track_name'].str.split('|').str[0].str.strip()

        # IMPORTANT: Normalize distance values before using them
        if 'distance' in self.historic_data.columns:
            print("Normalizing distance values in historic data...")
            self.historic_data['distance'] = self.historic_data['distance'].apply(normalize_distance)
            print(f"Successfully normalized {self.historic_data['distance'].notna().sum()} distance values")

        # Create label for historic data
        self.historic_data['label'] = (self.historic_data['race_position_clean'] == 1).astype(int)

        # Calculate JT statistics from historic data
        self.jt_stats = self._calculate_jt_statistics()
        self.track_stats = self._calculate_track_statistics()
        self.going_stats = self._calculate_going_statistics()

        print(f"Historic data loaded: {len(self.historic_data)} races")

    def _calculate_track_statistics(self):
        """Calculate track statistics from historic data"""
        # Group by track and calculate statistics
        track_groups = self.historic_data.groupby('track_stripped')

        track_stats = {}
        for track, group in track_groups:
            # Calculate win rate
            win_rate = group['label'].mean()
            
            # Calculate average distance, handling missing values
            distance_values = group['distance'].dropna()
            avg_distance = distance_values.mean() if len(distance_values) > 0 else 0
            
            track_stats[track] = {
                'track_win_rate': win_rate,
                'track_avg_distance': avg_distance
            }

        # Convert to DataFrame
        track_stats_df = pd.DataFrame.from_dict(track_stats, orient='index')

        # Handle any remaining NaN values
        track_stats_df = track_stats_df.fillna(0)

        return track_stats_df
        
    def _calculate_jt_statistics(self):
        """Calculate jockey-trainer combination statistics from historic data"""
        jt_wins = self.historic_data[self.historic_data['label'] == 1].groupby('j_t').size()
        jt_runs = self.historic_data.groupby('j_t').size()
        
        jt_stats = pd.DataFrame({
            'jt_wins': jt_wins,
            'jt_runs': jt_runs
        }).fillna(0)
        
        jt_stats['wilson_score'] = jt_stats.apply(
            lambda row: wilson_score(row['jt_wins'], row['jt_runs']), axis=1
        )
        
        return jt_stats
    
    def _calculate_track_statistics(self):
        """Calculate track statistics from historic data"""
        track_stats = self.historic_data.groupby('track_stripped').agg({
            'label': 'mean',
            'distance': lambda x: x.dropna().mean() if len(x.dropna()) > 0 else 0
        }).rename(columns={
            'label': 'track_win_rate',
            'distance': 'track_avg_distance'
        })
        
        return track_stats
    
    def _calculate_going_statistics(self):
        """Calculate going condition statistics from historic data"""
        # Clean going conditions
        self.historic_data['going_clean'] = self.historic_data['going'].str.split('(').str[0].str.strip()
        self.historic_data['going_clean'] = self.historic_data['going_clean'].apply(extract_clean_going)
        
        going_stats = {}
        for going_type in self.historic_data['going_clean'].dropna().unique():
            going_mask = self.historic_data['going_clean'] == going_type
            perf = self.historic_data.loc[going_mask].groupby('horse_name')['label'].mean()
            going_stats[f'performance_on_{going_type}'] = perf
            
        return going_stats
    
    def engineer_features(self, prediction_data):
        """
        Apply feature engineering to prediction data using historic statistics
        
        Args:
            prediction_data: DataFrame with new race data to predict
            
        Returns:
            DataFrame with engineered features
        """
        print("Engineering features for prediction data...")
        df = prediction_data.copy()
        
        # Normalize basic features
        if 'distance' in df.columns:
            df['distance'] = df['distance'].apply(normalize_distance)
        if 'weight' in df.columns:
            df['weight'] = df['weight'].apply(normalize_weight)
            
        # Ensure numeric columns
        numeric_columns = ['age', 'claims']
        if 'rating_clean' in df.columns:
            numeric_columns.append('rating_clean')
        elif 'rating' in df.columns:
            numeric_columns.append('rating')
            
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean names and create combinations
        df['J_Name'] = df['jockey'].apply(clean_names).str.replace('.','', regex=False)
        df['T_Name'] = df['trainer'].apply(clean_names).str.replace('.','', regex=False)
        df['track_stripped'] = df['track_name'].str.split('|').str[0].str.strip()
        df['j_t'] = df['J_Name'].astype(str) + " | " + df['T_Name'].astype(str)
        
        # Map JT statistics from historic data
        df['jt_wins'] = df['j_t'].map(self.jt_stats['jt_wins']).fillna(0).astype(int)
        df['jt_runs'] = df['j_t'].map(self.jt_stats['jt_runs']).fillna(0).astype(int)
        df['wilson_score'] = df['j_t'].map(self.jt_stats['wilson_score']).fillna(0)
        
        # Clean going conditions
        df['going'] = df['going'].str.split('(').str[0].str.strip()
        df['going'] = df['going'].apply(extract_clean_going)
        
        # Map going performance from historic data
        for going_type, performance_dict in self.going_stats.items():
            df[going_type] = df['horse_name'].map(performance_dict).fillna(0)
        
        # Calculate EMA Form using combined historic + prediction data
        df['EMA_Form'] = 0.5  # Default value
        combined_data = pd.concat([self.historic_data, df], ignore_index=True, sort=False)
        combined_data = combined_data.sort_values('race_date_clean') if 'race_date_clean' in combined_data.columns else combined_data.sort_index()
        
        for horse in df['horse_name'].unique():
            horse_races = combined_data[combined_data['horse_name'] == horse].copy()
            if len(horse_races) >= 2:
                form_features = calculate_form_features(horse_races)
                if 'EMA_Form' in form_features:
                    df.loc[df['horse_name'] == horse, 'EMA_Form'] = form_features['EMA_Form']
        
        # Historical performance features using combined data
        df['recent_win_rate'] = 0.0
        df['career_wins'] = 0
        df['career_races'] = 0
        df['career_win_rate'] = 0.0
        df['days_since_last'] = 0
        
        for horse in df['horse_name'].unique():
            # Get all historic races for this horse
            horse_historic = self.historic_data[self.historic_data['horse_name'] == horse]
            
            if len(horse_historic) > 0:
                horse_historic = horse_historic.sort_values('race_date_clean') if 'race_date_clean' in horse_historic.columns else horse_historic
                
                career_wins = horse_historic['label'].sum()
                career_races = len(horse_historic)
                career_win_rate = career_wins / career_races if career_races > 0 else 0
                
                recent_labels = horse_historic['label'].tail(5)
                recent_win_rate = recent_labels.mean() if len(recent_labels) > 0 else 0
                
                df.loc[df['horse_name'] == horse, 'recent_win_rate'] = recent_win_rate
                df.loc[df['horse_name'] == horse, 'career_wins'] = career_wins
                df.loc[df['horse_name'] == horse, 'career_races'] = career_races
                df.loc[df['horse_name'] == horse, 'career_win_rate'] = career_win_rate
        
        # Track-specific features
        df['track_win_rate'] = df['track_stripped'].map(self.track_stats['track_win_rate']).fillna(df.get('label', pd.Series([0])).mean())
        df['track_avg_distance'] = df['track_stripped'].map(self.track_stats['track_avg_distance']).fillna(0)
        
        # Weight and age features
        if 'weight' in df.columns:
            if df['weight'].notna().sum() > 0:
                df['weight_percentile'] = df['weight'].rank(pct=True)
            else:
                df['weight_percentile'] = 0.5
        
        if 'age' in df.columns:
            df['is_young_horse'] = (df['age'] <= 3).astype(int)
            df['is_prime_age'] = ((df['age'] >= 4) & (df['age'] <= 6)).astype(int)
            df['is_veteran'] = (df['age'] >= 7).astype(int)
        
        print("Feature engineering complete!")
        return df
    
    def prepare_prediction_features(self, df):
        """Prepare feature matrix for prediction (same as training)"""
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
        
        raw_features = ['distance', 'claims', 'age', 'weight', 'rating_clean']
        
        # Check which features exist - ADD MISSING FEATURES WITH DEFAULTS
        all_features = []
        missing_features = []
        
        for feat in high_correlation_features + additional_features + raw_features:
            if feat in df.columns:
                all_features.append(feat)
            else:
                missing_features.append(feat)
                # Add missing feature with default value
                if feat in ['is_young_horse', 'is_prime_age', 'is_veteran']:
                    df[feat] = 0  # Default age category
                elif feat == 'claims':
                    df[feat] = 0  # Default claims
                elif feat == 'age':
                    df[feat] = 5  # Default age
                elif feat == 'rating_clean':
                    df[feat] = 70  # Default rating
                else:
                    df[feat] = 0  # Default for other features
                all_features.append(feat)
        
        if missing_features:
            print(f"Added missing features with defaults: {missing_features}")
        
        self.feature_columns = all_features
        print(f"Using {len(all_features)} features for prediction")
        print(f"Features: {all_features}")
        
        # Create a copy with only the required features
        feature_df = df[all_features].copy()
        
        # Handle missing values
        for col in all_features:
            if feature_df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                if feature_df[col].notna().sum() > 0:
                    feature_df[col] = feature_df[col].fillna(feature_df[col].median())
                else:
                    feature_df[col] = feature_df[col].fillna(0)
        
        # Final validation
        print(f"Final feature_df shape: {feature_df.shape}")
        print(f"Final feature_df columns: {list(feature_df.columns)}")
        
        # Convert to numpy array
        X_array = feature_df.values
        print(f"Final X_array shape: {X_array.shape}")
        
        return X_array
    
    def predict_race(self, prediction_data_path, output_path=None, threshold=0.5):
        """
        Make predictions on new race data
        
        Args:
            prediction_data_path: Path to CSV file with new race data
            output_path: Path to save results (optional)
            threshold: Probability threshold for binary predictions
            
        Returns:
            DataFrame with predictions
        """
        # Load prediction data
        print(f"Loading prediction data from {prediction_data_path}...")
        prediction_df = pd.read_csv(prediction_data_path)
        print(f"Loaded {len(prediction_df)} horses to predict")
        
        # Engineer features
        engineered_df = self.engineer_features(prediction_df)
        
        # Prepare feature matrix
        X_pred = self.prepare_prediction_features(engineered_df)
        
        # Make predictions
        print("Making predictions...")
        probabilities = self.model.predict_proba(X_pred)[:, 1]
        binary_predictions = (probabilities >= threshold).astype(int)
        
        # Create results dataframe
        results = prediction_df.copy()
        results['win_probability'] = probabilities
        results['predicted_winner'] = binary_predictions
        results['prediction_confidence'] = np.abs(probabilities - 0.5) * 2  # Scale to 0-1
        
        # Add rank within each race (if race_id exists)
        if 'race_id' in results.columns:
            results['probability_rank'] = results.groupby('race_id')['win_probability'].rank(ascending=False, method='dense')
        else:
            results['probability_rank'] = results['win_probability'].rank(ascending=False, method='dense')
        
        # Sort by probability (highest first)
        results = results.sort_values('win_probability', ascending=False)
        
        # Print summary
        print(f"\nPrediction Summary:")
        print(f"Total horses: {len(results)}")
        print(f"Predicted winners (threshold {threshold}): {binary_predictions.sum()}")
        print(f"Average win probability: {probabilities.mean():.3f}")
        print(f"Max probability: {probabilities.max():.3f}")
        print(f"Min probability: {probabilities.min():.3f}")
        
        # Show top predictions
        print(f"\nTop 10 Predictions:")
        print("-" * 80)
        top_cols = ['horse_name', 'jockey', 'trainer', 'win_probability', 'predicted_winner', 'probability_rank']
        available_cols = [col for col in top_cols if col in results.columns]
        print(results[available_cols].head(10).to_string(index=False))
        
        # Save results if output path provided
        if output_path:
            results.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
        
        return results

# ============================================================
#                    MAIN PREDICTION FUNCTION
# ============================================================

def main():
    """Main prediction pipeline"""
    
    # File paths
    MODEL_PATH = "tabnet_horse_racing_model.zip"
    HISTORIC_DATA_PATH = "merged_horse_racing_data.csv"
    PREDICTION_DATA_PATH = "daily_racing_data\irish_horses_2025-06-15.csv"
    OUTPUT_PATH = "race_predictions.csv"
    
    # Check if files exist
    required_files = [MODEL_PATH, HISTORIC_DATA_PATH, PREDICTION_DATA_PATH]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"Error: Required file not found: {file_path}")
            return
    
    try:
        # Initialize predictor
        predictor = HorseRacingPredictor(MODEL_PATH, HISTORIC_DATA_PATH)
        
        # Load model and historic data
        predictor.load_model()
        predictor.load_historic_data()
        
        # Make predictions with different thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        for threshold in thresholds:
            print(f"\n{'='*60}")
            print(f"PREDICTIONS WITH THRESHOLD {threshold}")
            print(f"{'='*60}")
            
            output_file = f"race_predictions_threshold_{threshold}.csv"
            results = predictor.predict_race(
                PREDICTION_DATA_PATH, 
                output_file, 
                threshold=threshold
            )
            
            # Show race-by-race breakdown if race_id exists
            if 'race_id' in results.columns:
                print(f"\nRace-by-Race Breakdown (Threshold {threshold}):")
                print("-" * 50)
                race_summary = results.groupby('race_id').agg({
                    'predicted_winner': 'sum',
                    'win_probability': ['mean', 'max'],
                    'horse_name': 'count'
                }).round(3)
                race_summary.columns = ['Predicted_Winners', 'Avg_Probability', 'Max_Probability', 'Total_Horses']
                print(race_summary)
        
        print(f"\n{'='*60}")
        print("PREDICTION COMPLETE")
        print(f"{'='*60}")
        print("Check the generated CSV files for detailed results.")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# ============================================================
#                    MAIN PREDICTION FUNCTION
# ============================================================

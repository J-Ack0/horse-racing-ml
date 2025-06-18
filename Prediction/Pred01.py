import pandas as pd
import numpy as np
import pickle
import re
import os
from datetime import datetime
from pytorch_tabnet.tab_model import TabNetClassifier

class HorseRacingPredictor:
    """
    Production deployment class for horse racing predictions
    """
    
    def __init__(self, model_path, historical_data_path="merged_horses.csv"):
        """
        Initialize the predictor with saved model and historical data
        
        Args:
            model_path: Path to saved model directory
            historical_data_path: Path to historical data CSV for feature generation
        """
        self.model_path = model_path
        self.historical_data_path = historical_data_path
        self.historical_data = None
        self.model_components = None
        
        # Load historical data and model
        self._load_historical_data()
        self._load_model_components()
        
    def _load_historical_data(self):
        """Load and prepare historical data for feature generation"""
        print(f"📂 Loading historical data from {self.historical_data_path}...")
        
        try:
            self.historical_data = pd.read_csv(self.historical_data_path)
            print(f"✅ Loaded {len(self.historical_data)} historical records")
            
            # Ensure we have a label column for historical calculations
            if 'race_position' in self.historical_data.columns:
                self.historical_data['historical_label'] = (self.historical_data['race_position'] == '1st').astype(int)
            elif 'label' in self.historical_data.columns:
                self.historical_data['historical_label'] = self.historical_data['label']
            else:
                # If no position data, create dummy labels (this would need real data in production)
                print("⚠️  No race position data found - using dummy labels")
                self.historical_data['historical_label'] = 0
                
            # Convert date column if available
            if 'race_date' in self.historical_data.columns:
                self.historical_data['race_date'] = pd.to_datetime(
                    self.historical_data['race_date'], errors='coerce'
                )
                
        except FileNotFoundError:
            print(f"❌ Historical data file not found: {self.historical_data_path}")
            print("Creating empty historical data - features will use defaults")
            self.historical_data = pd.DataFrame()
            
    def _load_model_components(self):
        """Load the saved model and all preprocessing components"""
        print(f"🤖 Loading model components from {self.model_path}...")
        
        try:
            # Load TabNet model
            tabnet_path = os.path.join(self.model_path, "tabnet_model.zip")
            model = TabNetClassifier()
            model.load_model(tabnet_path)
            
            # Load encoders
            with open(os.path.join(self.model_path, "encoders.pkl"), 'rb') as f:
                encoders = pickle.load(f)
                
            # Load scaler
            with open(os.path.join(self.model_path, "scaler.pkl"), 'rb') as f:
                scaler = pickle.load(f)
                
            # Load metadata
            with open(os.path.join(self.model_path, "model_metadata.pkl"), 'rb') as f:
                metadata = pickle.load(f)
                
            self.model_components = {
                'model': model,
                'encoders': encoders,
                'scaler': scaler,
                'metadata': metadata
            }
            
            print(f"✅ Model loaded successfully")
            print(f"   📊 AUC Score: {metadata['auc_score']:.4f}")
            print(f"   🎯 Best F1 Threshold: {metadata['best_f1_threshold']:.3f}")
            print(f"   💰 Best Profit Threshold: {metadata['best_profit_threshold']:.3f}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
            
    def _map_columns(self, df):
        """Map new data format to training format"""
        print("🔄 Mapping columns to training format...")
        
        # Create mapping dictionary
        column_mapping = {
            'header': 'track_name',
            'date': 'race_date', 
            'going': 'going',
            'distance': 'distance',
            'class': 'claims',
            'runner_name': 'horse_name',
            'runner_age': 'age',
            'jockey_name': 'jockey',
            'trainer_name': 'trainer',
            'runner_weight': 'weight',
            'runner_rpr': 'rating',
            'time': 'race_time'
        }
        
        # Apply mapping
        mapped_df = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in mapped_df.columns:
                mapped_df[new_col] = mapped_df[old_col]
        
        return mapped_df
    
    def _normalize_distance(self, dist_str):
        """Convert distance format like '7f' to yards"""
        dist_str = str(dist_str).strip()
        
        if dist_str in ['nan', 'NaN', 'N/A', '-', '', 'None']:
            return np.nan
            
        # Handle formats like '7f', '1m 2f', '2m 6f 74y'
        total_yards = 0
        
        # Miles
        miles_match = re.search(r'(\d+(?:\.\d+)?)\s*m', dist_str, re.IGNORECASE)
        if miles_match:
            total_yards += float(miles_match.group(1)) * 1760
            
        # Furlongs
        furlongs_match = re.search(r'(\d+(?:\.\d+)?)\s*f', dist_str, re.IGNORECASE)
        if furlongs_match:
            total_yards += float(furlongs_match.group(1)) * 220
            
        # Yards
        yards_match = re.search(r'(\d+(?:\.\d+)?)\s*y', dist_str, re.IGNORECASE)
        if yards_match:
            total_yards += float(yards_match.group(1))
            
        return total_yards if total_yards > 0 else np.nan
    
    def _normalize_weight(self, w_str):
        """Normalize weight from '10-4' format to pounds"""
        w_str = str(w_str).strip()
        
        if w_str in ['-', '', 'N/A', 'nan', 'NaN', 'None']:
            return np.nan
            
        # British stone-pounds format: "10-4" = 10 stone 4 pounds
        stone_pounds_match = re.match(r'^(\d+)-(\d+)', w_str)
        if stone_pounds_match:
            stone = int(stone_pounds_match.group(1))
            pounds = int(stone_pounds_match.group(2))
            return stone * 14 + pounds
            
        # Fallback: extract leading number
        match = re.match(r'^(\d+(?:\.\d+)?)', w_str)
        if match:
            return float(match.group(1))
            
        return np.nan
    
    def _extract_class_value(self, class_str):
        """Extract numeric value from class like '(Class 3)'"""
        class_str = str(class_str).strip()
        
        # Extract number from formats like "(Class 3)" or "Class 3"
        match = re.search(r'class\s*(\d+)', class_str, re.IGNORECASE)
        if match:
            class_num = int(match.group(1))
            # Convert class to approximate claim value
            # Class 1 = highest value, Class 6 = lowest value
            class_to_claims = {
                1: 100000,
                2: 50000, 
                3: 25000,
                4: 15000,
                5: 8000,
                6: 5000
            }
            return class_to_claims.get(class_num, 15000)  # Default to mid-range
            
        return 15000  # Default value
    
    def _calculate_historical_features(self, df):
        """Calculate historical performance features using historical data"""
        print("📈 Calculating historical features...")
        
        if self.historical_data.empty:
            print("⚠️  No historical data available - using default values")
            # Set default values for all historical features
            df['recent_win_rate'] = 0.1  # 10% default win rate
            df['career_wins'] = 0
            df['career_races'] = 0
            df['career_win_rate'] = 0.1
            df['days_since_last'] = 30  # Default 30 days
            df['jockey_win_rate'] = 0.1
            df['trainer_win_rate'] = 0.1
            df['track_win_rate'] = 0.1
            df['track_avg_distance'] = 1540  # ~7f average
            return df
        
        # Initialize feature columns
        historical_features = [
            'recent_win_rate', 'career_wins', 'career_races', 'career_win_rate',
            'days_since_last', 'jockey_win_rate', 'trainer_win_rate', 
            'track_win_rate', 'track_avg_distance'
        ]
        
        for feature in historical_features:
            df[feature] = 0.0
            
        # Calculate horse historical features
        if 'horse_name' in df.columns and 'horse_name' in self.historical_data.columns:
            for idx in df.index:
                horse_name = df.loc[idx, 'horse_name']
                
                # Get historical races for this horse
                horse_history = self.historical_data[
                    self.historical_data['horse_name'] == horse_name
                ]
                
                if len(horse_history) > 0:
                    # Career statistics
                    career_wins = horse_history['historical_label'].sum()
                    career_races = len(horse_history)
                    career_win_rate = career_wins / career_races if career_races > 0 else 0.1
                    
                    # Recent win rate (last 5 races)
                    recent_history = horse_history.tail(5)
                    recent_win_rate = recent_history['historical_label'].mean() if len(recent_history) > 0 else 0.1
                    
                    df.loc[idx, 'career_wins'] = career_wins
                    df.loc[idx, 'career_races'] = career_races
                    df.loc[idx, 'career_win_rate'] = career_win_rate
                    df.loc[idx, 'recent_win_rate'] = recent_win_rate
                else:
                    # No history found - use defaults
                    df.loc[idx, 'career_wins'] = 0
                    df.loc[idx, 'career_races'] = 0
                    df.loc[idx, 'career_win_rate'] = 0.1
                    df.loc[idx, 'recent_win_rate'] = 0.1
        
        # Calculate jockey win rates
        if 'jockey' in df.columns and 'jockey' in self.historical_data.columns:
            jockey_stats = self.historical_data.groupby('jockey')['historical_label'].agg(['mean', 'count']).reset_index()
            jockey_stats.columns = ['jockey', 'jockey_win_rate', 'jockey_races']
            jockey_stats = jockey_stats[jockey_stats['jockey_races'] >= 5]  # Minimum races
            
            df = df.merge(jockey_stats[['jockey', 'jockey_win_rate']], on='jockey', how='left')
            df['jockey_win_rate'] = df['jockey_win_rate'].fillna(0.1)  # Default 10%
        
        # Calculate trainer win rates
        if 'trainer' in df.columns and 'trainer' in self.historical_data.columns:
            trainer_stats = self.historical_data.groupby('trainer')['historical_label'].agg(['mean', 'count']).reset_index()
            trainer_stats.columns = ['trainer', 'trainer_win_rate', 'trainer_races']
            trainer_stats = trainer_stats[trainer_stats['trainer_races'] >= 5]  # Minimum races
            
            df = df.merge(trainer_stats[['trainer', 'trainer_win_rate']], on='trainer', how='left')
            df['trainer_win_rate'] = df['trainer_win_rate'].fillna(0.1)  # Default 10%
        
        # Calculate track statistics
        if 'track_name' in df.columns and 'track_name' in self.historical_data.columns:
            track_stats = self.historical_data.groupby('track_name').agg({
                'historical_label': 'mean',
                'distance': 'mean'
            }).reset_index()
            track_stats.columns = ['track_name', 'track_win_rate', 'track_avg_distance']
            
            df = df.merge(track_stats, on='track_name', how='left')
            df['track_win_rate'] = df['track_win_rate'].fillna(0.1)
            df['track_avg_distance'] = df['track_avg_distance'].fillna(1540)  # ~7f default
        
        # Set remaining defaults
        df['days_since_last'] = df.get('days_since_last', 30)
        
        print(f"✅ Historical features calculated for {len(df)} horses")
        return df
    
    def _add_additional_features(self, df):
        """Add age-based and other additional features"""
        print("⚡ Adding additional features...")
        
        # Age-based features
        if 'age' in df.columns:
            df['is_young_horse'] = (df['age'] <= 3).astype(int)
            df['is_prime_age'] = ((df['age'] >= 4) & (df['age'] <= 6)).astype(int) 
            df['is_veteran'] = (df['age'] >= 7).astype(int)
        
        # Weight percentile
        if 'weight' in df.columns:
            df['weight_percentile'] = df['weight'].rank(pct=True)
        
        # Race class encoding (if we added it during preprocessing)
        if 'claims' in df.columns:
            df['race_class'] = pd.cut(df['claims'], 
                                     bins=[0, 5000, 15000, 50000, float('inf')], 
                                     labels=['maiden', 'low_class', 'mid_class', 'high_class'])
            df['race_class_encoded'] = df['race_class'].cat.codes
        
        return df
    
    def preprocess_data(self, df):
        """Complete preprocessing pipeline for new data"""
        print("🔧 Starting data preprocessing...")
        
        # Step 1: Map columns
        df = self._map_columns(df)
        
        # Step 2: Normalize distance and weight
        print("📏 Normalizing distance and weight...")
        df['distance'] = df['distance'].apply(self._normalize_distance)
        df['weight'] = df['weight'].apply(self._normalize_weight)
        
        # Step 3: Extract class values
        if 'claims' in df.columns:
            df['claims'] = df['claims'].apply(self._extract_class_value)
        
        # Step 4: Handle missing values
        print("🧹 Handling missing values...")
        
        # Numeric columns
        numeric_cols = ['distance', 'claims', 'age', 'weight', 'rating']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                median_val = df[col].median() if df[col].median() > 0 else 0
                df[col] = df[col].fillna(median_val)
        
        # Categorical columns
        cat_cols = ['track_name', 'race_time', 'going', 'horse_name', 'jockey', 'trainer']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('missing')
                df[col] = df[col].replace({'nan': 'missing', 'NaN': 'missing', 'None': 'missing'})
        
        # Step 5: Calculate historical features
        df = self._calculate_historical_features(df)
        
        # Step 6: Add additional features
        df = self._add_additional_features(df)
        
        print("✅ Preprocessing completed")
        return df
    
    def prepare_features(self, df):
        """Prepare features in the exact format expected by the model"""
        print("🎯 Preparing features for prediction...")
        
        # Get expected feature columns from metadata
        expected_features = self.model_components['metadata']['feature_columns']
        cat_features = self.model_components['metadata']['categorical_features']
        num_features = self.model_components['metadata']['numerical_features']
        
        # Create feature dataframe with only the columns we need
        feature_df = pd.DataFrame()
        
        # Add each expected feature, creating default values if missing
        for feature in expected_features:
            if feature in df.columns:
                feature_df[feature] = df[feature]
            else:
                # Create default values for missing features
                if feature in cat_features:
                    feature_df[feature] = 'missing'
                elif feature in num_features:
                    feature_df[feature] = 0.0
                else:
                    feature_df[feature] = 0.0
                print(f"⚠️  Missing feature '{feature}' - using default value")
        
        # Apply encoders to categorical features
        print("🏷️  Encoding categorical features...")
        encoders = self.model_components['encoders']
        
        for feature in cat_features:
            if feature in feature_df.columns and feature in encoders:
                le = encoders[feature]
                # Handle unseen categories
                feature_df[feature] = feature_df[feature].astype(str)
                unseen_mask = ~feature_df[feature].isin(le.classes_)
                if unseen_mask.any():
                    print(f"   ⚠️ Unseen categories in '{feature}': {feature_df[feature][unseen_mask].unique()}")
                    feature_df.loc[unseen_mask, feature] = 'missing'  # Map to 'missing'
                
                # Apply encoding
                feature_df[feature] = le.transform(feature_df[feature])
        
        # Apply scaler to numerical features
        print("📊 Scaling numerical features...")
        scaler = self.model_components['scaler']
        
        if num_features:
            # Ensure all numerical features are present
            num_features_present = [f for f in num_features if f in feature_df.columns]
            if num_features_present:
                feature_df[num_features_present] = scaler.transform(feature_df[num_features_present])
        
        print(f"✅ Features prepared: {feature_df.shape}")
        return feature_df
    
    def predict(self, race_data):
        """
        Make predictions for new race data
        
        Args:
            race_data: DataFrame with new race data or path to CSV file
            
        Returns:
            DataFrame with predictions and probabilities
        """
        print("🏇 Starting horse racing predictions...")
        
        # Load data if it's a file path
        if isinstance(race_data, str):
            print(f"📂 Loading race data from {race_data}...")
            df = pd.read_csv(race_data)
        else:
            df = race_data.copy()
        
        print(f"📊 Processing {len(df)} horses...")
        
        # Store original data for results
        original_df = df.copy()
        
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Prepare features
        feature_df = self.prepare_features(df)
        
        # Make predictions
        print("🤖 Making predictions...")
        model = self.model_components['model']
        metadata = self.model_components['metadata']
        
        # Get probabilities
        probabilities = model.predict_proba(feature_df.values)[:, 1]
        
        # Apply thresholds
        f1_threshold = metadata['best_f1_threshold']
        profit_threshold = metadata['best_profit_threshold']
        
        predictions_f1 = (probabilities >= f1_threshold).astype(int)
        predictions_profit = (probabilities >= profit_threshold).astype(int)
        
        # Create results dataframe
        results = original_df.copy()
        results['win_probability'] = probabilities
        results['prediction_f1'] = predictions_f1
        results['prediction_profit'] = predictions_profit
        results['confidence_score'] = np.where(
            probabilities >= 0.5, 
            probabilities, 
            1 - probabilities
        )
        
        # Add recommendation
        results['recommendation'] = 'PASS'
        results.loc[results['prediction_profit'] == 1, 'recommendation'] = 'BET'
        results.loc[results['prediction_f1'] == 1, 'recommendation'] = 'CONSIDER'
        results.loc[
            (results['prediction_profit'] == 1) & (results['win_probability'] > 0.3), 
            'recommendation'
        ] = 'STRONG BET'
        
        # Sort by probability descending
        results = results.sort_values('win_probability', ascending=False)
        
        print("✅ Predictions completed!")
        return results
    
    def predict_and_summarize(self, race_data):
        """Make predictions and provide a summary"""
        results = self.predict(race_data)
        
        print("\n" + "="*60)
        print("🏇 HORSE RACING PREDICTION RESULTS")
        print("="*60)
        
        # Summary statistics
        total_horses = len(results)
        strong_bets = len(results[results['recommendation'] == 'STRONG BET'])
        bets = len(results[results['recommendation'] == 'BET'])
        consider = len(results[results['recommendation'] == 'CONSIDER'])
        
        print(f"📊 SUMMARY:")
        print(f"   Total Horses: {total_horses}")
        print(f"   Strong Bets: {strong_bets}")
        print(f"   Regular Bets: {bets}")
        print(f"   Consider: {consider}")
        print(f"   Pass: {total_horses - strong_bets - bets - consider}")
        
        # Top recommendations
        print(f"\n🏆 TOP RECOMMENDATIONS:")
        top_horses = results.head(3)
        for idx, row in top_horses.iterrows():
            print(f"   {row['runner_name']}: {row['win_probability']:.1%} ({row['recommendation']})")
        
        # Betting recommendations
        betting_horses = results[results['recommendation'].isin(['STRONG BET', 'BET'])]
        if len(betting_horses) > 0:
            print(f"\n💰 BETTING RECOMMENDATIONS:")
            for idx, row in betting_horses.iterrows():
                print(f"   {row['recommendation']}: {row['runner_name']} - {row['win_probability']:.1%} chance")
        else:
            print(f"\n💰 BETTING RECOMMENDATIONS: No strong betting opportunities found")
        
        return results


def example_usage():
    """Example of how to use the predictor"""
    
    # Sample new race data (your format)
    sample_data = {
        'url': ['https://www.racingpost.com/racecards/107/york/2025-06-14/895197'],
        'time': ['1:50'],
        'header': ['York'],
        'date': ['14 Jun 2025 ITV4'],
        'going': ['Good To Firm'],
        'distance': ['7f'],
        'class': ['(Class 3)'],
        'runner_name': ['Telemark'],
        'horse_url': ['https://www.racingpost.com/profile/horse/5276317/telemark#race-id=895197'],
        'runner_ts': [42],
        'runner_rpr': [101],
        'price': [''],
        'runner_age': [4],
        'jockey_name': ['Liam Wright'],
        'jockey_url': ['https://www.racingpost.com/profile/jockey/101550/liam-wright'],
        'trainer_name': ['Simon & Ed Crisford'],
        'trainer_url': ['https://www.racingpost.com/profile/trainer/37478/simon-ed-crisford'],
        'runner_weight': ['10-4'],
        'jockey_claim': [5]
    }
    
    # Convert to DataFrame
    race_df = pd.DataFrame(sample_data)
    
    # Initialize predictor (adjust paths as needed)
    predictor = HorseRacingPredictor(
        model_path="models/horse_racing_model_20241215_143022",  # Your saved model path
        historical_data_path="merged_horses.csv"  # Your historical data
    )
    
    # Make predictions
    results = predictor.predict_and_summarize(race_df)
    
    # Save results
    results.to_csv('race_predictions.csv', index=False)
    print(f"\n💾 Results saved to: race_predictions.csv")
    
    return results


if __name__ == "__main__":
    # Run example
    try:
        results = example_usage()
        print("\n🎉 Deployment test completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during deployment test: {e}")
        import traceback
        traceback.print_exc()
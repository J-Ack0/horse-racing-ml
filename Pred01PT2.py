"""
Horse Racing TabNet Model Deployment Script
===========================================

Complete deployment script for making predictions on new horse racing data
using a previously trained TabNet model.

Features:
- Automatic model detection and loading
- Historical performance feature generation
- Race-by-race predictions with ranking
- Comprehensive error handling
- Interactive and command-line interfaces

QUICK FIX for TabNet loading errors:
------------------------------------
If you get "init_params" or similar errors, try:

1. Use minimal loader:
   results = quick_predict('your_file.csv', use_minimal_loader=True)

2. Run diagnostics:
   python deployment_script.py --fix

3. Command line with minimal loader:
   python deployment_script.py your_file.csv --minimal
"""

import pandas as pd
import numpy as np
import pickle
import re
import os
import json
import torch
import warnings
from datetime import datetime
from pytorch_tabnet.tab_model import TabNetClassifier
from typing import Dict, List, Tuple, Optional, Union

warnings.filterwarnings('ignore')


def load_tabnet_minimal(model_dir: str) -> TabNetClassifier:
    """
    Minimal TabNet loader - use this if standard loading fails
    
    Args:
        model_dir: Path to extracted tabnet_model directory
        
    Returns:
        Loaded TabNetClassifier
    """
    print("🔧 Using minimal TabNet loader...")
    
    # Create a basic model
    model = TabNetClassifier(
        device_name='cpu',
        verbose=0
    )
    
    # Try to load using the built-in method first
    try:
        model.load_model(model_dir)
        print("✅ Loaded successfully with built-in method")
        return model
    except Exception as e:
        print(f"⚠️  Built-in loading failed: {str(e)}")
    
    # Manual loading as fallback
    try:
        # Load the network state
        network_path = os.path.join(model_dir, "network.pt")
        if not os.path.exists(network_path):
            raise FileNotFoundError(f"Network file not found: {network_path}")
        
        # Load the state dict
        state_dict = torch.load(network_path, map_location='cpu')
        
        # We need to initialize the model's network first
        # This is a bit hacky but sometimes necessary
        
        # Option 1: Try to get input/output dims from state dict
        try:
            # Look for input layer shape
            input_dim = None
            output_dim = None
            
            for key, tensor in state_dict.items():
                if 'initial_splitter.0.weight' in key:
                    input_dim = tensor.shape[1]
                elif 'final_mapping.weight' in key:
                    output_dim = tensor.shape[0]
            
            if input_dim and output_dim:
                print(f"   Detected dimensions: input={input_dim}, output={output_dim}")
                
                # Create properly sized model
                model = TabNetClassifier(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    device_name='cpu',
                    verbose=0
                )
                
                # Initialize with dummy data
                dummy_X = np.random.randn(10, input_dim)
                dummy_y = np.random.randint(0, output_dim, 10)
                model.fit(dummy_X, dummy_y, max_epochs=1, verbose=0)
                
                # Now load the real weights
                model.network.load_state_dict(state_dict)
                model.network.eval()
                
                print("✅ Loaded successfully with dimension detection")
                return model
                
        except Exception as e:
            print(f"   Dimension detection failed: {str(e)}")
        
        # Option 2: Try with default dimensions
        print("   Trying with default dimensions...")
        dummy_X = np.random.randn(10, 100)  # Assume 100 features
        dummy_y = np.random.randint(0, 2, 10)  # Binary classification
        
        model = TabNetClassifier(device_name='cpu', verbose=0)
        model.fit(dummy_X, dummy_y, max_epochs=1, verbose=0)
        
        # Load weights (may fail if dimensions don't match)
        model.network.load_state_dict(state_dict, strict=False)
        model.network.eval()
        
        print("✅ Loaded with default dimensions (may have warnings)")
        return model
        
    except Exception as e:
        print(f"❌ Manual loading failed: {str(e)}")
        raise RuntimeError(
            "Could not load TabNet model. The model file may be corrupted or "
            "incompatible with the current TabNet version. Try re-training the model."
        )


class HorseRacingPredictor:
    """Production deployment class for horse racing predictions"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 historical_data_path: str = "merged_horses.csv",
                 use_minimal_loader: bool = False):
        """
        Initialize predictor with saved model and historical data
        
        Args:
            model_path: Path to saved model directory (auto-detects if None)
            historical_data_path: Path to historical data CSV
            use_minimal_loader: Use minimal loader if standard loading fails
        """
        # Auto-detect model if not provided
        if model_path is None:
            model_path = self._get_latest_model()
            if model_path is None:
                raise FileNotFoundError(
                    "No trained models found! Please run the training script first."
                )
            print(f"✅ Auto-detected model: {os.path.basename(model_path)}")
        
        self.model_path = model_path
        self.historical_data_path = historical_data_path
        self.use_minimal_loader = use_minimal_loader
        self.historical_data = None
        self.model_components = None
        
        # Load components
        self._load_historical_data()
        self._load_model_components()

    def _load_tabnet_model(model_dir: str) -> TabNetClassifier:
        """
        Minimal TabNet loader - use this if standard loading fails
        
        Args:
            model_dir: Path to extracted tabnet_model directory
            
        Returns:
            Loaded TabNetClassifier
        """
        print("🔧 Using minimal TabNet loader...")
        
        # Create a basic model
        model = TabNetClassifier(
            device_name='cpu',
            verbose=0
        )
        
        # Try to load using the built-in method first
        try:
            model.load_model(model_dir)
            print("✅ Loaded successfully with built-in method")
            return model
        except Exception as e:
            print(f"⚠️  Built-in loading failed: {str(e)}")
        
        # Manual loading as fallback
        try:
            # Load the network state
            network_path = os.path.join(model_dir, "network.pt")
            if not os.path.exists(network_path):
                raise FileNotFoundError(f"Network file not found: {network_path}")
            
            # Load the state dict
            state_dict = torch.load(network_path, map_location='cpu')
            
            # We need to initialize the model's network first
            # This is a bit hacky but sometimes necessary
            
            # Option 1: Try to get input/output dims from state dict
            try:
                # Look for input layer shape
                input_dim = None
                output_dim = None
                
                for key, tensor in state_dict.items():
                    if 'initial_splitter.0.weight' in key:
                        input_dim = tensor.shape[1]
                    elif 'final_mapping.weight' in key:
                        output_dim = tensor.shape[0]
                
                if input_dim and output_dim:
                    print(f"   Detected dimensions: input={input_dim}, output={output_dim}")
                    
                    # Create properly sized model
                    model = TabNetClassifier(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        device_name='cpu',
                        verbose=0
                    )
                    
                    # Initialize with dummy data
                    dummy_X = np.random.randn(10, input_dim)
                    dummy_y = np.random.randint(0, output_dim, 10)
                    model.fit(dummy_X, dummy_y, max_epochs=1, verbose=0)
                    
                    # Now load the real weights
                    model.network.load_state_dict(state_dict)
                    model.network.eval()
                    
                    print("✅ Loaded successfully with dimension detection")
                    return model
                    
            except Exception as e:
                print(f"   Dimension detection failed: {str(e)}")
            
            # Option 2: Try with default dimensions
            print("   Trying with default dimensions...")
            dummy_X = np.random.randn(10, 100)  # Assume 100 features
            dummy_y = np.random.randint(0, 2, 10)  # Binary classification
            
            model = TabNetClassifier(device_name='cpu', verbose=0)
            model.fit(dummy_X, dummy_y, max_epochs=1, verbose=0)
            
            # Load weights (may fail if dimensions don't match)
            model.network.load_state_dict(state_dict, strict=False)
            model.network.eval()
            
            print("✅ Loaded with default dimensions (may have warnings)")
            return model
            
        except Exception as e:
            print(f"❌ Manual loading failed: {str(e)}")
            raise RuntimeError(
                "Could not load TabNet model. The model file may be corrupted or "
                "incompatible with the current TabNet version. Try re-training the model."
            )

    def _load_model_components(self):
        """
        Load saved model, encoders, scaler, and metadata.
        """
        print("🤖 Loading model components...")
        try:
            # Load preprocessing components
            encoders_path = os.path.join(self.model_path, "encoders.pkl")
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f)

            # Ensure 'missing' category exists in all encoders
            for name, encoder in encoders.items():
                if hasattr(encoder, 'classes_') and 'missing' not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, 'missing')

            scaler_path = os.path.join(self.model_path, "scaler.pkl")
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            metadata_path = os.path.join(self.model_path, "model_metadata.pkl")
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            # Load TabNet model
            model = self._load_tabnet_model(metadata)

            # Store all components
            self.model_components = {
                'model': model,
                'encoders': encoders,
                'scaler': scaler,
                'metadata': metadata
            }

            # Print summary
            auc = metadata.get('auc_score')
            threshold = metadata.get('best_profit_threshold', 0.5)
            print("✅ Model loaded successfully")
            print(f"   📊 AUC Score: {auc:.4f}" if auc is not None else "   📊 AUC Score: N/A")
            print(f"   🎯 Best Threshold: {threshold:.3f}")

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
        
    def _get_latest_model(self, model_dir: str = "models") -> Optional[str]:
        """Find the most recent trained model"""
        if not os.path.exists(model_dir):
            return None
        
        model_paths = []
        for item in os.listdir(model_dir):
            item_path = os.path.join(model_dir, item)
            if os.path.isdir(item_path):
                # Check for required files
                required = ['tabnet_model.zip', 'encoders.pkl', 
                           'scaler.pkl', 'model_metadata.pkl']
                if all(os.path.exists(os.path.join(item_path, f)) for f in required):
                    model_paths.append(item_path)
        
        if not model_paths:
            return None
        
        # Sort by timestamp in directory name
        def extract_timestamp(path):
            dirname = os.path.basename(path)
            match = re.search(r'(\d{8}_\d{6})', dirname)
            if match:
                try:
                    return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')
                except:
                    pass
            return datetime.fromtimestamp(os.path.getmtime(path))
        
        model_paths.sort(key=extract_timestamp, reverse=True)
        return model_paths[0]
    
    def _load_historical_data(self):
        """Load historical race data for feature generation"""
        print(f"📂 Loading historical data...")
        
        try:
            self.historical_data = pd.read_csv(self.historical_data_path)
            
            # Create win indicator
            if 'race_position' in self.historical_data.columns:
                self.historical_data['is_winner'] = (
                    self.historical_data['race_position'] == '1st'
                ).astype(int)
            elif 'label' in self.historical_data.columns:
                self.historical_data['is_winner'] = self.historical_data['label']
            else:
                print("⚠️  No race position data found in historical data")
                self.historical_data['is_winner'] = 0
            
            # Convert date if available
            if 'race_date' in self.historical_data.columns:
                self.historical_data['race_date'] = pd.to_datetime(
                    self.historical_data['race_date'], errors='coerce'
                )
            
            # Clean numeric columns
            numeric_cols = ['distance', 'age', 'weight', 'rating', 'claims']
            for col in numeric_cols:
                if col in self.historical_data.columns:
                    self.historical_data[col] = pd.to_numeric(
                        self.historical_data[col], errors='coerce'
                    )
            
            print(f"✅ Loaded {len(self.historical_data):,} historical records")
            
        except FileNotFoundError:
            print(f"⚠️  Historical data not found - using empty dataset")
            self.historical_data = pd.DataFrame()
    
    import os
import pickle
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

class TabNetModelHandler:
    def __init__(self, model_path: str, use_minimal_loader: bool = False):
        """
        Handles loading of TabNet model and its preprocessing artifacts.

        Args:
            model_path (str): Directory where model artifacts are stored.
            use_minimal_loader (bool): Whether to use a minimal loader for the model archive.
        """
        self.model_path = model_path
        self.use_minimal_loader = use_minimal_loader
        self.model_components = {}
        self._load_model_components()

    def _load_model_components(self):
        """
        Load saved model, encoders, scaler, and metadata.
        """
        print("🤖 Loading model components...")
        try:
            # Load preprocessing components
            encoders_path = os.path.join(self.model_path, "encoders.pkl")
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f)

            # Ensure 'missing' category exists in all encoders
            for name, encoder in encoders.items():
                if hasattr(encoder, 'classes_') and 'missing' not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, 'missing')

            scaler_path = os.path.join(self.model_path, "scaler.pkl")
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)

            metadata_path = os.path.join(self.model_path, "model_metadata.pkl")
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            # Load TabNet model
            model = self._load_tabnet_model(metadata)

            # Store all components
            self.model_components = {
                'model': model,
                'encoders': encoders,
                'scaler': scaler,
                'metadata': metadata
            }

            # Print summary
            auc = metadata.get('auc_score')
            threshold = metadata.get('best_profit_threshold', 0.5)
            print("✅ Model loaded successfully")
            print(f"   📊 AUC Score: {auc:.4f}" if auc is not None else "   📊 AUC Score: N/A")
            print(f"   🎯 Best Threshold: {threshold:.3f}")

        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise

    def _load_tabnet_model(self, metadata: dict) -> TabNetClassifier:
        """
        Load the TabNet model from disk using full loader.

        Args:
            metadata (dict): Metadata containing model parameters or paths.

        Returns:
            TabNetClassifier: Loaded TabNet model.
        """
        # If model was saved via .save_model, use load_model
        model_zip = os.path.join(self.model_path, "tabnet_model.zip")
        if os.path.exists(model_zip):
            model = TabNetClassifier(**metadata.get('model_params', {}))
            model.load_model(model_zip)
            return model

        # Fallback: load state_dict
        state_path = os.path.join(self.model_path, "tabnet_state_dict.pth")
        if os.path.exists(state_path):
            params = metadata.get('model_params', {})
            model = TabNetClassifier(**params)
            state_dict = torch.load(state_path, map_location=torch.device('cpu'))
            model.network.load_state_dict(state_dict)
            return model

        raise FileNotFoundError(
            "No TabNet model archive or state dict found in model_path."
        )

# Example usage:
# handler = TabNetModelHandler('/path/to/saved_model')
# components = handler.model_components
# model = components['model']
# encoders = components['encoders']
# scaler = components['scaler']
# metadata = components['metadata']

    
    def _load_tabnet_model(self):
        """Load TabNet model with proper error handling"""
        # First, extract the model if it's zipped
        zip_path = os.path.join(self.model_path, "tabnet_model.zip")
        model_dir = os.path.join(self.model_path, "tabnet_model")
        
        if os.path.exists(zip_path) and not os.path.exists(model_dir):
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.model_path)
        
        # Try multiple loading methods
        try:
            # Method 1: Direct load_model (preferred)
            model = TabNetClassifier()
            model.load_model(model_dir)
            return model
        except Exception as e1:
            print(f"   Method 1 failed: {str(e1)}")
            
            try:
                # Method 2: Load with filtered parameters
                params_path = os.path.join(model_dir, "model_params.json")
                if os.path.exists(params_path):
                    with open(params_path, 'r') as f:
                        saved_params = json.load(f)
                    
                    # Filter out internal parameters
                    valid_params = [
                        'n_d', 'n_a', 'n_steps', 'gamma', 'cat_idxs', 'cat_dims',
                        'cat_emb_dim', 'n_independent', 'n_shared', 'epsilon',
                        'momentum', 'lambda_sparse', 'seed', 'clip_value', 'verbose',
                        'optimizer_fn', 'optimizer_params', 'scheduler_fn', 'scheduler_params',
                        'mask_type', 'input_dim', 'output_dim', 'device_name'
                    ]
                    
                    filtered_params = {k: v for k, v in saved_params.items() 
                                     if k in valid_params}
                    
                    # Set defaults for missing params
                    filtered_params.setdefault('device_name', 'cpu')
                    filtered_params.setdefault('verbose', 0)
                    
                    # Create model with filtered parameters
                    model = TabNetClassifier(**filtered_params)
                    
                    # Load the saved weights
                    model.load_model(model_dir)
                    return model
                else:
                    raise FileNotFoundError("model_params.json not found")
                    
            except Exception as e2:
                print(f"   Method 2 failed: {str(e2)}")
                
                # Method 3: Create basic model and load state dict
                try:
                    # Create a basic TabNet model
                    model = TabNetClassifier(
                        device_name='cpu',
                        verbose=0
                    )
                    
                    # Load network weights directly
                    network_path = os.path.join(model_dir, "network.pt")
                    if os.path.exists(network_path):
                        # Load state dict
                        state_dict = torch.load(network_path, map_location='cpu')
                        
                        # Try to infer model architecture from state dict
                        # This is a workaround when params are not properly saved
                        if hasattr(model, 'network'):
                            model.network.load_state_dict(state_dict, strict=False)
                        else:
                            # Initialize network first
                            # You might need to call fit with dummy data to initialize
                            import numpy as np
                            dummy_X = np.zeros((10, 10))  # Adjust size as needed
                            dummy_y = np.zeros(10)
                            model.fit(dummy_X, dummy_y, max_epochs=1, verbose=0)
                            model.network.load_state_dict(state_dict, strict=False)
                        
                        model.network.eval()
                        print("   ✅ Loaded using fallback method")
                        return model
                    else:
                        raise FileNotFoundError("network.pt not found")
                        
                except Exception as e3:
                    print(f"   Method 3 failed: {str(e3)}")
                    raise RuntimeError(
                        f"Failed to load TabNet model from {model_dir}. "
                        "Please ensure the model was saved correctly."
                    )
    
    def _map_deployment_to_training_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map deployment data columns to training format"""
        print("🔄 Mapping columns to training format...")
        
        # Column mapping dictionary
        mapping = {
            'header': 'track_name',
            'date': 'race_date',
            'time': 'race_time',
            'going': 'going',
            'distance': 'distance',
            'class': 'claims',
            'runner_name': 'horse_name',
            'runner_age': 'age',
            'jockey_name': 'jockey',
            'trainer_name': 'trainer',
            'runner_weight': 'weight',
            'runner_rpr': 'rating'
        }
        
        # Create new dataframe with mapped columns
        mapped_df = pd.DataFrame()
        
        for deploy_col, train_col in mapping.items():
            if deploy_col in df.columns:
                mapped_df[train_col] = df[deploy_col]
            else:
                # Set default values for missing columns
                if train_col in ['track_name', 'going', 'horse_name', 'jockey', 'trainer']:
                    mapped_df[train_col] = 'missing'
                elif train_col == 'race_time':
                    mapped_df[train_col] = '17:10'  # Default time
                else:
                    mapped_df[train_col] = 0
        
        # Keep original columns for reference
        for col in df.columns:
            if col not in mapping and col not in mapped_df.columns:
                mapped_df[col] = df[col]
        
        return mapped_df
    
    def _normalize_distance(self, dist_str: str) -> float:
        """Convert distance strings to yards"""
        if pd.isna(dist_str) or str(dist_str).strip() in ['', '-', 'N/A']:
            return 1540  # Default to 7 furlongs
        
        dist_str = str(dist_str).strip().lower()
        total_yards = 0
        
        # Extract miles, furlongs, and yards
        miles = re.search(r'(\d+(?:\.\d+)?)\s*m(?:ile)?', dist_str)
        furlongs = re.search(r'(\d+(?:\.\d+)?)\s*f(?:urlong)?', dist_str)
        yards = re.search(r'(\d+(?:\.\d+)?)\s*y(?:ard)?', dist_str)
        
        if miles:
            total_yards += float(miles.group(1)) * 1760
        if furlongs:
            total_yards += float(furlongs.group(1)) * 220
        if yards:
            total_yards += float(yards.group(1))
        
        return total_yards if total_yards > 0 else 1540
    
    def _normalize_weight(self, weight_str: str) -> float:
        """Convert weight from stone-pounds to total pounds"""
        if pd.isna(weight_str) or str(weight_str).strip() in ['', '-', 'N/A']:
            return 140  # Default weight
        
        weight_str = str(weight_str).strip()
        
        # Handle stone-pounds format (e.g., "10-4")
        match = re.match(r'(\d+)-(\d+)', weight_str)
        if match:
            stones = int(match.group(1))
            pounds = int(match.group(2))
            return stones * 14 + pounds
        
        # Try to extract any number
        match = re.search(r'(\d+(?:\.\d+)?)', weight_str)
        if match:
            return float(match.group(1))
        
        return 140  # Default weight
    
    def _extract_class_value(self, class_str: str) -> float:
        """Extract numeric claim value from class string"""
        if pd.isna(class_str):
            return 15000  # Default mid-range value
        
        class_str = str(class_str).strip()
        
        # Extract class number
        match = re.search(r'class\s*(\d+)', class_str, re.IGNORECASE)
        if match:
            class_num = int(match.group(1))
            # Map class to approximate claim values
            class_to_claims = {
                1: 100000, 2: 50000, 3: 25000,
                4: 15000, 5: 8000, 6: 5000
            }
            return class_to_claims.get(class_num, 15000)
        
        # Look for claim amount directly
        match = re.search(r'(\d+)k', class_str, re.IGNORECASE)
        if match:
            return float(match.group(1)) * 1000
        
        return 15000  # Default
    
    def _calculate_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate historical performance features"""
        print("📈 Calculating historical features...")
        
        # Initialize feature columns
        feature_cols = [
            'recent_win_rate', 'career_wins', 'career_races', 
            'career_win_rate', 'days_since_last', 'jockey_win_rate',
            'trainer_win_rate', 'track_win_rate', 'track_avg_distance'
        ]
        
        for col in feature_cols:
            df[col] = 0.0
        
        if self.historical_data.empty:
            print("⚠️  No historical data - using default values")
            df['recent_win_rate'] = 0.15
            df['career_win_rate'] = 0.15
            df['jockey_win_rate'] = 0.15
            df['trainer_win_rate'] = 0.15
            df['track_win_rate'] = 0.15
            df['track_avg_distance'] = 1540
            df['days_since_last'] = 30
            return df
        
        # Calculate features for each horse
        for idx in df.index:
            # Horse statistics
            if 'horse_name' in df.columns:
                horse_name = df.at[idx, 'horse_name']
                horse_hist = self.historical_data[
                    self.historical_data['horse_name'] == horse_name
                ]
                
                if len(horse_hist) > 0:
                    df.at[idx, 'career_races'] = len(horse_hist)
                    df.at[idx, 'career_wins'] = horse_hist['is_winner'].sum()
                    df.at[idx, 'career_win_rate'] = (
                        horse_hist['is_winner'].mean() 
                        if len(horse_hist) > 0 else 0.15
                    )
                    
                    # Recent form (last 5 races)
                    recent_races = horse_hist.tail(5)
                    df.at[idx, 'recent_win_rate'] = (
                        recent_races['is_winner'].mean() 
                        if len(recent_races) > 0 else 0.15
                    )
                    
                    # Days since last race
                    if 'race_date' in horse_hist.columns:
                        last_date = pd.to_datetime(horse_hist['race_date']).max()
                        if pd.notna(last_date):
                            current_date = pd.to_datetime(df.at[idx, 'race_date'])
                            if pd.notna(current_date):
                                df.at[idx, 'days_since_last'] = (
                                    current_date - last_date
                                ).days
            
            # Jockey statistics
            if 'jockey' in df.columns:
                jockey_name = df.at[idx, 'jockey']
                jockey_hist = self.historical_data[
                    self.historical_data['jockey'] == jockey_name
                ]
                if len(jockey_hist) >= 5:  # Minimum races for reliability
                    df.at[idx, 'jockey_win_rate'] = jockey_hist['is_winner'].mean()
                else:
                    df.at[idx, 'jockey_win_rate'] = 0.15
            
            # Trainer statistics
            if 'trainer' in df.columns:
                trainer_name = df.at[idx, 'trainer']
                trainer_hist = self.historical_data[
                    self.historical_data['trainer'] == trainer_name
                ]
                if len(trainer_hist) >= 5:
                    df.at[idx, 'trainer_win_rate'] = trainer_hist['is_winner'].mean()
                else:
                    df.at[idx, 'trainer_win_rate'] = 0.15
            
            # Track statistics
            if 'track_name' in df.columns:
                track_name = df.at[idx, 'track_name']
                track_hist = self.historical_data[
                    self.historical_data['track_name'] == track_name
                ]
                if len(track_hist) >= 10:
                    df.at[idx, 'track_win_rate'] = track_hist['is_winner'].mean()
                    if 'distance' in track_hist.columns:
                        df.at[idx, 'track_avg_distance'] = track_hist['distance'].mean()
                else:
                    df.at[idx, 'track_win_rate'] = 0.15
                    df.at[idx, 'track_avg_distance'] = 1540
        
        # Ensure no NaN values
        for col in feature_cols:
            df[col] = df[col].fillna(0.15 if 'rate' in col else 0)
        
        print(f"✅ Historical features calculated")
        return df
    
    def _add_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add additional engineered features"""
        # Age categories
        if 'age' in df.columns:
            df['is_young_horse'] = (df['age'] <= 3).astype(int)
            df['is_prime_age'] = ((df['age'] >= 4) & (df['age'] <= 6)).astype(int)
            df['is_veteran'] = (df['age'] >= 7).astype(int)
        
        # Weight percentile within race
        if 'weight' in df.columns:
            df['weight_percentile'] = df.groupby('url')['weight'].rank(pct=True)
        
        # Rating percentile within race
        if 'rating' in df.columns:
            df['rating_percentile'] = df.groupby('url')['rating'].rank(pct=True)
        
        # Distance categories
        if 'distance' in df.columns:
            df['is_sprint'] = (df['distance'] <= 1320).astype(int)  # 6f or less
            df['is_middle'] = ((df['distance'] > 1320) & (df['distance'] <= 2200)).astype(int)
            df['is_long'] = (df['distance'] > 2200).astype(int)
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        print("🔧 Preprocessing data...")
        
        # Map columns
        df = self._map_deployment_to_training_columns(df)
        
        # Normalize values
        df['distance'] = df['distance'].apply(self._normalize_distance)
        df['weight'] = df['weight'].apply(self._normalize_weight)
        
        if 'claims' in df.columns:
            df['claims'] = df['claims'].apply(self._extract_class_value)
        
        # Convert date
        if 'race_date' in df.columns:
            df['race_date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Handle missing values
        numeric_cols = ['distance', 'claims', 'age', 'weight', 'rating']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if col == 'distance':
                    df[col] = df[col].fillna(1540)  # 7 furlongs
                elif col == 'weight':
                    df[col] = df[col].fillna(140)  # 10 stone
                elif col == 'claims':
                    df[col] = df[col].fillna(15000)
                else:
                    df[col] = df[col].fillna(df[col].median())
        
        # Handle categorical columns
        cat_cols = ['track_name', 'race_time', 'going', 'horse_name', 'jockey', 'trainer']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna('missing').astype(str)
                df[col] = df[col].replace(['nan', 'NaN', 'None', ''], 'missing')
        
        # Calculate historical features
        df = self._calculate_historical_features(df)
        
        # Add engineered features
        df = self._add_engineered_features(df)
        
        print("✅ Preprocessing completed")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for model input"""
        print("🎯 Preparing features for model...")
        
        metadata = self.model_components['metadata']
        expected_features = metadata['feature_columns']
        cat_features = metadata['categorical_features']
        num_features = metadata['numerical_features']
        
        # Create feature dataframe
        X = pd.DataFrame()
        
        # Add expected features
        for feature in expected_features:
            if feature in df.columns:
                X[feature] = df[feature]
            else:
                # Use appropriate defaults
                if feature in cat_features:
                    X[feature] = 'missing'
                else:
                    X[feature] = 0.0
                print(f"⚠️  Missing feature '{feature}' - using default")
        
        # Encode categorical features
        encoders = self.model_components['encoders']
        for feature in cat_features:
            if feature in X.columns and feature in encoders:
                encoder = encoders[feature]
                
                # Handle unseen categories
                X[feature] = X[feature].astype(str)
                unseen_mask = ~X[feature].isin(encoder.classes_)
                if unseen_mask.any():
                    X.loc[unseen_mask, feature] = 'missing'
                
                # Transform
                X[feature] = encoder.transform(X[feature])
        
        # Scale numerical features
        if num_features:
            scaler = self.model_components['scaler']
            X[num_features] = scaler.transform(X[num_features])
        
        # Ensure column order matches training
        X = X[expected_features]
        
        print(f"✅ Features prepared: {X.shape}")
        return X
    
    def predict_races(self, race_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Make predictions for races, grouping horses by race
        
        Args:
            race_data: DataFrame or path to CSV file
            
        Returns:
            DataFrame with predictions grouped by race
        """
        print("🏇 Starting race predictions...")
        
        # Load data
        if isinstance(race_data, str):
            print(f"📂 Loading data from {race_data}")
            df = pd.read_csv(race_data)
        else:
            df = race_data.copy()
        
        print(f"📊 Found {len(df)} horses across races")
        
        # Store original data
        original_df = df.copy()
        
        # Preprocess
        df = self.preprocess_data(df)
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Make predictions
        print("🤖 Generating predictions...")
        model = self.model_components['model']
        metadata = self.model_components['metadata']
        
        # Get win probabilities
        probabilities = model.predict_proba(X.values)[:, 1]
        
        # Apply thresholds
        threshold = metadata.get('best_profit_threshold', 0.5)
        
        # Create results
        results = original_df.copy()
        results['win_probability'] = probabilities
        results['predicted_winner'] = (probabilities >= threshold).astype(int)
        
        # Group by race and add rankings
        if 'url' in results.columns:
            # Add race-level rankings
            results['race_rank'] = results.groupby('url')['win_probability'].rank(
                ascending=False, method='min'
            )
            results['horses_in_race'] = results.groupby('url')['url'].transform('count')
            
            # Add confidence metrics
            results['confidence'] = np.abs(results['win_probability'] - 0.5) * 2
            
            # Betting recommendations
            results['recommendation'] = 'PASS'
            
            # Top horse in each race
            top_horses = results.loc[results.groupby('url')['win_probability'].idxmax()]
            results.loc[top_horses.index, 'recommendation'] = 'TOP PICK'
            
            # High probability horses
            high_prob = results['win_probability'] > 0.3
            confident = results['confidence'] > 0.4
            results.loc[high_prob & confident, 'recommendation'] = 'CONSIDER'
            
            # Very high probability
            very_high = results['win_probability'] > 0.4
            results.loc[very_high, 'recommendation'] = 'STRONG BET'
            
            # Sort by race and probability
            results = results.sort_values(['url', 'win_probability'], ascending=[True, False])
        else:
            # No race grouping available
            results = results.sort_values('win_probability', ascending=False)
        
        print("✅ Predictions completed!")
        return results
    
    def generate_race_summary(self, results: pd.DataFrame) -> None:
        """Generate a summary of predictions by race"""
        print("\n" + "="*60)
        print("🏇 RACE-BY-RACE PREDICTIONS SUMMARY")
        print("="*60)
        
        if 'url' in results.columns:
            # Group by race
            for race_url, race_df in results.groupby('url'):
                race_df = race_df.sort_values('win_probability', ascending=False)
                
                print(f"\n📍 Race: {race_df.iloc[0]['header']} - {race_df.iloc[0]['time']}")
                print(f"   Distance: {race_df.iloc[0]['distance']}")
                print(f"   Class: {race_df.iloc[0]['class']}")
                print(f"   Going: {race_df.iloc[0]['going']}")
                print(f"   Horses: {len(race_df)}")
                print("\n   TOP 3 PICKS:")
                
                for i, (_, horse) in enumerate(race_df.head(3).iterrows(), 1):
                    print(f"   {i}. {horse['runner_name']:<20} "
                          f"Win Prob: {horse['win_probability']:.1%} "
                          f"({horse['recommendation']})")
                
                # Betting edge
                top_prob = race_df.iloc[0]['win_probability']
                if 'price' in race_df.columns and pd.notna(race_df.iloc[0]['price']):
                    price = race_df.iloc[0]['price']
                    print(f"\n   💰 Top pick odds: {price}")
        else:
            # Single race or no URL column
            print("\n🏆 TOP PREDICTIONS:")
            for i, (_, horse) in enumerate(results.head(5).iterrows(), 1):
                print(f"{i}. {horse['runner_name']:<20} "
                      f"Win Prob: {horse['win_probability']:.1%} "
                      f"({horse['recommendation']})")
        
        # Summary statistics
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total horses analyzed: {len(results)}")
        print(f"   Strong bets: {len(results[results['recommendation'] == 'STRONG BET'])}")
        print(f"   Top picks: {len(results[results['recommendation'] == 'TOP PICK'])}")
        print(f"   Consider: {len(results[results['recommendation'] == 'CONSIDER'])}")
        
        return None


def quick_predict(csv_path: str, save_results: bool = True, use_minimal_loader: bool = False) -> pd.DataFrame:
    """
    Quick prediction function - one line to get predictions!
    
    Args:
        csv_path: Path to race data CSV
        save_results: Whether to save results to file
        use_minimal_loader: Use minimal TabNet loader if having issues
        
    Returns:
        DataFrame with predictions
    """
    try:
        # Initialize predictor
        predictor = HorseRacingPredictor(use_minimal_loader=use_minimal_loader)
        
        # Make predictions
        results = predictor.predict_races(csv_path)
        
        # Generate summary
        predictor.generate_race_summary(results)
        
        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"predictions_{timestamp}.csv"
            results.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
        if "init_params" in str(e) or "TabModel" in str(e):
            print("\n💡 TIP: Try using the minimal loader:")
            print("   quick_predict('your_file.csv', use_minimal_loader=True)")
        
        print("\n💡 Other things to check:")
        print("   1. Trained model exists (run training script)")
        print("   2. Historical data file: merged_horses.csv")
        print("   3. Valid race data CSV file")
        print("   4. Run diagnose_model() to check model files")
        return None


def validate_csv(csv_path: str) -> Tuple[bool, List[str]]:
    """Validate race data CSV has required columns"""
    try:
        df = pd.read_csv(csv_path, nrows=5)
        
        required = [
            'header', 'date', 'going', 'distance', 'class',
            'runner_name', 'runner_age', 'jockey_name', 
            'trainer_name', 'runner_weight', 'runner_rpr'
        ]
        
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            print(f"❌ Missing required columns: {missing}")
            return False, missing
        
        print(f"✅ CSV validation passed ({len(df)} rows)")
        return True, []
        
    except Exception as e:
        print(f"❌ Error reading CSV: {str(e)}")
        return False, []


def create_sample_data() -> str:
    """Create sample race data for testing"""
    sample = pd.DataFrame({
        'url': ['race1'] * 5 + ['race2'] * 4,
        'time': ['14:30'] * 5 + ['15:00'] * 4,
        'header': ['Ascot'] * 5 + ['Newmarket'] * 4,
        'date': ['14 Jun 2025'] * 9,
        'going': ['Good to Firm'] * 9,
        'distance': ['1m'] * 5 + ['7f'] * 4,
        'class': ['(Class 2)'] * 5 + ['(Class 3)'] * 4,
        'runner_name': [
            'Thunder Strike', 'Golden Dream', 'Fast Lightning', 
            'Storm Chaser', 'Royal Fortune',
            'Speed Demon', 'Lucky Star', 'Wild Fire', 'Blue Moon'
        ],
        'runner_age': [4, 3, 5, 4, 3, 4, 3, 5, 4],
        'jockey_name': [
            'R. Moore', 'W. Buick', 'J. Doyle', 'T. Marquand', 'O. Murphy',
            'F. Dettori', 'C. Soumillon', 'K. Shoemark', 'D. Tudhope'
        ],
        'trainer_name': [
            'A. O\'Brien', 'J. Gosden', 'C. Appleby', 'R. Varian', 'A. Balding',
            'J. Gosden', 'W. Haggas', 'M. Johnston', 'R. Fahey'
        ],
        'runner_weight': [
            '9-2', '8-12', '9-5', '9-0', '8-10',
            '9-3', '8-11', '9-1', '8-13'
        ],
        'runner_rpr': [115, 112, 118, 110, 108, 120, 105, 116, 109],
        'runner_ts': [45, 42, 48, 40, 38, 50, 35, 46, 40],
        'price': ['5/2', '3/1', '2/1', '8/1', '10/1', '11/4', '6/1', '4/1', '9/2']
    })
    
    filename = 'sample_race_data.csv'
    sample.to_csv(filename, index=False)
    print(f"✅ Created sample data: {filename}")
    return filename


def diagnose_model(model_path: Optional[str] = None) -> None:
    """
    Diagnose issues with a saved model
    
    Args:
        model_path: Path to model directory (auto-detects if None)
    """
    print("🔍 MODEL DIAGNOSTICS")
    print("="*50)
    
    # Find model path
    if model_path is None:
        try:
            predictor = HorseRacingPredictor()
            model_path = predictor._get_latest_model()
        except:
            model_path = None
            
        if model_path is None:
            print("❌ No models found")
            return
    
    print(f"📂 Checking model: {model_path}")
    
    # Check files
    required_files = {
        'tabnet_model.zip': 'TabNet model archive',
        'encoders.pkl': 'Label encoders',
        'scaler.pkl': 'Feature scaler',
        'model_metadata.pkl': 'Model metadata'
    }
    
    print("\n📋 File check:")
    for file, desc in required_files.items():
        path = os.path.join(model_path, file)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024  # KB
            print(f"   ✅ {file:<20} ({size:.1f} KB) - {desc}")
        else:
            print(f"   ❌ {file:<20} - {desc}")
    
    # Check model parameters
    zip_path = os.path.join(model_path, "tabnet_model.zip")
    model_dir = os.path.join(model_path, "tabnet_model")
    
    if os.path.exists(zip_path):
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zf:
            print(f"\n📦 Zip contents:")
            for info in zf.filelist:
                print(f"   - {info.filename} ({info.file_size} bytes)")
            
            # Extract temporarily to check params
            if not os.path.exists(model_dir):
                zf.extractall(model_path)
    
    # Check model params
    params_path = os.path.join(model_dir, "model_params.json")
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        print(f"\n🔧 Model parameters:")
        print(f"   Total parameters: {len(params)}")
        print(f"   Parameters: {list(params.keys())}")
        
        # Check for problematic params
        valid_params = [
            'n_d', 'n_a', 'n_steps', 'gamma', 'cat_idxs', 'cat_dims',
            'cat_emb_dim', 'n_independent', 'n_shared', 'epsilon',
            'momentum', 'lambda_sparse', 'seed', 'clip_value', 'verbose',
            'optimizer_fn', 'optimizer_params', 'scheduler_fn', 'scheduler_params',
            'mask_type', 'input_dim', 'output_dim', 'device_name'
        ]
        
        invalid_params = [p for p in params.keys() if p not in valid_params]
        if invalid_params:
            print(f"\n   ⚠️  Invalid parameters found: {invalid_params}")
            print("   These will be filtered out during loading")
    
    # Check metadata
    metadata_path = os.path.join(model_path, "model_metadata.pkl")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"\n📊 Model metadata:")
        print(f"   AUC Score: {metadata.get('auc_score', 'N/A')}")
        print(f"   Features: {metadata.get('total_features', 'N/A')}")
        print(f"   Training samples: {metadata.get('training_samples', 'N/A')}")
        print(f"   Best threshold: {metadata.get('best_profit_threshold', 'N/A')}")
    
    print("\n✅ Diagnostics complete")


def main():
    """Interactive main menu"""
    print("🏇 HORSE RACING PREDICTION SYSTEM")
    print("="*50)
    print("TabNet Model Deployment Tool")
    print()
    
    while True:
        print("\n📋 OPTIONS:")
        print("1. Quick predict from CSV file")
        print("2. Create sample data for testing")
        print("3. Validate CSV file")
        print("4. Diagnose model issues")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            csv_path = input("Enter CSV file path: ").strip()
            if csv_path and os.path.exists(csv_path):
                use_minimal = input("Use minimal loader? (y/n, default=n): ").strip().lower() == 'y'
                quick_predict(csv_path, use_minimal_loader=use_minimal)
            else:
                print("❌ File not found")
                
        elif choice == '2':
            sample_file = create_sample_data()
            print(f"💡 Now run: quick_predict('{sample_file}')")
            
        elif choice == '3':
            csv_path = input("Enter CSV file path: ").strip()
            if csv_path and os.path.exists(csv_path):
                validate_csv(csv_path)
            else:
                print("❌ File not found")
                
        elif choice == '4':
            diagnose_model()
                
        elif choice == '5':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    import sys
    
    # Quick fix for common TabNet loading issues
    if "--fix" in sys.argv:
        print("🔧 Running TabNet fix...")
        diagnose_model()
        print("\n💡 Try running with minimal loader:")
        print("   python deployment_script.py your_file.csv --minimal")
        sys.exit(0)
    
    # Check for minimal loader flag
    use_minimal = "--minimal" in sys.argv
    if use_minimal:
        print("📌 Using minimal TabNet loader")
        sys.argv.remove("--minimal")
    
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        # Command line usage
        csv_file = sys.argv[1]
        print(f"🚀 Command line mode: {csv_file}")
        quick_predict(csv_file, use_minimal_loader=use_minimal)
    else:
        # Interactive mode
        main()
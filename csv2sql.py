import pandas as pd
import re
import sqlite3
from typing import Union

class HorseRacingDataProcessor:
    def __init__(self, csv_file_path: str):
        """Initialize the processor with a CSV file path."""
        self.csv_file_path = csv_file_path
        self.df = None
        self.sql_statements = []
        
    def load_data(self):
        """Load the CSV data into a pandas DataFrame."""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print(f"Loaded {len(self.df)} rows from {self.csv_file_path}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False
        return True
    
    def clean_race_position(self, position: str) -> int:
        """
        Clean race position data.
        - Convert '1st', '2nd', etc. to integers
        - Convert non-finishers (PU, NR, RO, UR, F, BD) to -1
        """
        if pd.isna(position) or position == '':
            return -1
            
        position = str(position).strip()
        
        # Handle non-finishers (2-character codes)
        non_finishers = ['PU', 'NR', 'RO', 'UR', 'F', 'BD']
        if position in non_finishers:
            return -1
            
        # Extract numeric part from positions like '1st', '2nd', '3rd', etc.
        match = re.match(r'(\d+)', position)
        if match:
            return int(match.group(1))
        
        # If we can't parse it, mark as non-finisher
        return -1
    
    def clean_lost_by_length(self, lost_length: str, race_position: int) -> float:
        """
        Clean lost by length data.
        - If position is 1 (1st place), set to 0
        - Convert fractions and lengths to decimal
        - Handle special cases like 'Nose', 'Head', 'N/A'
        """
        if race_position == 1:
            return 0.0
            
        if pd.isna(lost_length) or lost_length in ['N/A', '', '-']:
            return 0.0
            
        lost_length = str(lost_length).strip().lower()
        
        # Handle special distance terms
        if 'nose' in lost_length:
            return 0.05  # Very small margin
        elif 'head' in lost_length:
            return 0.1   # Small margin
        elif 'neck' in lost_length:
            return 0.2
        elif 'short head' in lost_length:
            return 0.08
        
        # Extract numeric values and fractions
        # Pattern matches things like "3 1/4 l", "1/2 l", "5 l", "69 l"
        pattern = r'(\d+)?\s*(\d+/\d+)?\s*l?'
        match = re.match(pattern, lost_length)
        
        if match:
            whole_part = match.group(1)
            fraction_part = match.group(2)
            
            total = 0.0
            
            if whole_part:
                total += float(whole_part)
            
            if fraction_part:
                numerator, denominator = map(int, fraction_part.split('/'))
                total += numerator / denominator
                
            return total
        
        # If we can't parse it, return 0
        return 0.0
    
    def clean_rating(self, rating: Union[str, int]) -> int:
        """
        Clean rating data.
        - Convert '-' to 0
        - Keep numeric ratings as is
        """
        if pd.isna(rating) or rating == '-' or rating == '':
            return 0
        
        try:
            return int(rating)
        except (ValueError, TypeError):
            return 0
    
    def clean_weight(self, weight: str) -> str:
        """
        Clean weight data - remove extra characters but keep the weight format.
        """
        if pd.isna(weight) or weight == '' or weight == '-':
            return '0-0'
        
        weight = str(weight).strip()
        # Remove trailing letters like 'v', 't', etc. but keep the weight format
        weight = re.sub(r'[a-zA-Z]+$', '', weight)
        return weight.strip()
    
    def clean_string_field(self, value: str) -> str:
        """Clean string fields - handle NaN and empty values."""
        if pd.isna(value) or value == '' or value == '-':
            return 'Unknown'
        return str(value).strip()
    
    def clean_track_name(self, track_name: str) -> str:
        """
        Clean track name by removing 'Racing Results' and '|' character.
        Example: 'Sligo Racing Results | 10th June 2025 17:58' -> 'Sligo 10th June 2025 17:58'
        """
        if pd.isna(track_name) or track_name == '' or track_name == '-':
            return 'Unknown'
        
        track_name = str(track_name).strip()
        
        # Remove 'Racing Results' (case insensitive)
        track_name = re.sub(r'Racing Results', '', track_name, flags=re.IGNORECASE)
        
        # Remove '|' character
        track_name = re.sub(r'\|', '', track_name)
        
        # Clean up extra whitespace
        track_name = re.sub(r'\s+', ' ', track_name).strip()
        
        return track_name if track_name else 'Unknown'
    
    def clean_data(self):
        """Apply all cleaning transformations to the dataset."""
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        print("Cleaning data...")
        
        # Clean race positions first
        self.df['race_position_clean'] = self.df['race_position'].apply(self.clean_race_position)
        
        # Clean lost by length (depends on race position)
        self.df['lost_by_length_clean'] = self.df.apply(
            lambda row: self.clean_lost_by_length(row['lost_by_length'], row['race_position_clean']), 
            axis=1
        )
        
        # Clean ratings
        self.df['rating_clean'] = self.df['rating'].apply(self.clean_rating)
        
        # Clean weights
        self.df['weight_clean'] = self.df['weight'].apply(self.clean_weight)
        
        # Clean track name specifically
        self.df['track_name_clean'] = self.df['track_name'].apply(self.clean_track_name)
        
        # Clean other string fields
        string_fields = ['race_date', 'race_time', 'going', 'distance', 
                        'horse_name', 'jockey', 'trainer']
        
        for field in string_fields:
            self.df[f'{field}_clean'] = self.df[field].apply(self.clean_string_field)
        
        # Handle claims and age
        self.df['claims_clean'] = self.df['claims'].fillna(0).astype(int)
        self.df['age_clean'] = self.df['age'].fillna(0).astype(int)
        
        # Handle boolean field
        self.df['first_place_found_clean'] = self.df['first_place_found'].apply(
            lambda x: 1 if str(x).lower() == 'true' else 0
        )
        
        print("Data cleaning completed.")
    
    def generate_sql_statements(self, table_name: str = 'horse_racing_data'):
        """Generate SQL INSERT statements for the cleaned data."""
        if self.df is None:
            print("No data loaded.")
            return
        
        print(f"Generating SQL INSERT statements for table '{table_name}'...")
        
        # Create table statement
        create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {table_name} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_name VARCHAR(200),
    race_date VARCHAR(50),
    race_time VARCHAR(20),
    going VARCHAR(100),
    distance VARCHAR(50),
    horse_name VARCHAR(100),
    jockey VARCHAR(100),
    claims INTEGER,
    trainer VARCHAR(100),
    race_position INTEGER,
    lost_by_length DECIMAL(5,2),
    age INTEGER,
    weight VARCHAR(20),
    rating INTEGER,
    first_place_found BOOLEAN
);"""
        
        self.sql_statements.append(create_table_sql)
        
        # Generate INSERT statements
        for _, row in self.df.iterrows():
            insert_sql = f"""
INSERT INTO {table_name} (
    track_name, race_date, race_time, going, distance, horse_name, 
    jockey, claims, trainer, race_position, lost_by_length, 
    age, weight, rating, first_place_found
) VALUES (
    '{row['track_name_clean'].replace("'", "''")}',
    '{row['race_date_clean'].replace("'", "''")}',
    '{row['race_time_clean'].replace("'", "''")}',
    '{row['going_clean'].replace("'", "''")}',
    '{row['distance_clean'].replace("'", "''")}',
    '{row['horse_name_clean'].replace("'", "''")}',
    '{row['jockey_clean'].replace("'", "''")}',
    {row['claims_clean']},
    '{row['trainer_clean'].replace("'", "''")}',
    {row['race_position_clean']},
    {row['lost_by_length_clean']},
    {row['age_clean']},
    '{row['weight_clean'].replace("'", "''")}',
    {row['rating_clean']},
    {row['first_place_found_clean']}
);"""
            
            self.sql_statements.append(insert_sql)
        
        print(f"Generated {len(self.sql_statements)} SQL statements.")
    
    def save_sql_to_file(self, output_file: str = 'horse_racing_insert_statements.sql'):
        """Save the SQL statements to a file."""
        if not self.sql_statements:
            print("No SQL statements to save. Run generate_sql_statements() first.")
            return
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for statement in self.sql_statements:
                    f.write(statement.strip() + '\n\n')
            
            print(f"SQL statements saved to {output_file}")
            return True
        except Exception as e:
            print(f"Error saving SQL file: {e}")
            return False
    
    def print_data_summary(self):
        """Print a summary of the cleaned data."""
        if self.df is None:
            print("No data loaded.")
            return
        
        print("\n" + "="*50)
        print("DATA CLEANING SUMMARY")
        print("="*50)
        
        print(f"Total records: {len(self.df)}")
        
        print(f"\nRace Position Analysis:")
        print(f"Winners (1st place): {(self.df['race_position_clean'] == 1).sum()}")
        print(f"Non-finishers (-1): {(self.df['race_position_clean'] == -1).sum()}")
        print(f"Regular positions: {((self.df['race_position_clean'] > 1) & (self.df['race_position_clean'] < 50)).sum()}")
        
        print(f"\nRating Analysis:")
        print(f"Missing ratings (set to 0): {(self.df['rating_clean'] == 0).sum()}")
        print(f"Valid ratings: {(self.df['rating_clean'] > 0).sum()}")
        
        print(f"\nLost by Length Analysis:")
        print(f"Winners (0 length): {(self.df['lost_by_length_clean'] == 0).sum()}")
        print(f"Average lost by length: {self.df['lost_by_length_clean'].mean():.2f}")

def main():
    """Main function to process the horse racing data."""
    # Initialize processor
    processor = HorseRacingDataProcessor('csv/test.csv')
    
    # Load and process data
    if processor.load_data():
        processor.clean_data()
        processor.print_data_summary()
        processor.generate_sql_statements()
        processor.save_sql_to_file()
        print("\nProcessing completed successfully!")
    else:
        print("Failed to process data.")

if __name__ == "__main__":
    main()
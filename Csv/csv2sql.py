import pandas as pd
import re
import sqlite3
import os
from pathlib import Path
from typing import Union, List

class HorseRacingDataProcessor:
    def __init__(self, folder_path: str):
        """Initialize the processor with a folder path containing CSV files."""
        self.folder_path = Path(folder_path)
        self.csv_files = []
        self.df = None
        self.sql_statements = []
        
    def discover_csv_files(self) -> List[Path]:
        """Discover all CSV files in the specified folder."""
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder does not exist: {self.folder_path}")
        
        if not self.folder_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.folder_path}")
        
        csv_files = list(self.folder_path.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in folder: {self.folder_path}")
        
        print(f"Found {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"  - {csv_file.name}")
        
        return csv_files
        
    def load_data(self):
        """Load and merge all CSV files in the folder into a single DataFrame."""
        try:
            self.csv_files = self.discover_csv_files()
            
            dataframes = []
            total_rows = 0
            
            for csv_file in self.csv_files:
                print(f"\nLoading {csv_file.name}...")
                try:
                    df = pd.read_csv(csv_file)
                    rows_loaded = len(df)
                    total_rows += rows_loaded
                    
                    # Add a source file column to track which file each row came from
                    df['source_file'] = csv_file.name
                    
                    dataframes.append(df)
                    print(f"  ✓ Loaded {rows_loaded} rows from {csv_file.name}")
                    
                except Exception as e:
                    print(f"  ✗ Error loading {csv_file.name}: {e}")
                    continue
            
            if not dataframes:
                print("No CSV files were successfully loaded.")
                return False
            
            # Merge all dataframes
            print(f"\nMerging {len(dataframes)} dataframes...")
            self.df = pd.concat(dataframes, ignore_index=True)
            
            print(f"Successfully merged data:")
            print(f"  - Total files processed: {len(dataframes)}")
            print(f"  - Total rows: {len(self.df)}")
            print(f"  - Columns: {list(self.df.columns)}")
            
            # Show breakdown by source file
            print(f"\nRows per file:")
            file_counts = self.df['source_file'].value_counts()
            for file_name, count in file_counts.items():
                print(f"  - {file_name}: {count} rows")
                
        except Exception as e:
            print(f"Error during data loading: {e}")
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
    
    def analyze_duplicates(self, exclude_source_file: bool = True, show_examples: bool = True):
        """
        Analyze duplicate rows without removing them.
        
        Args:
            exclude_source_file (bool): If True, ignore source_file column when checking for duplicates
            show_examples (bool): If True, show examples of duplicate rows
        """
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        print(f"\n" + "="*50)
        print("DUPLICATE ANALYSIS")
        print("="*50)
        
        # Determine which columns to use for duplicate detection
        if exclude_source_file and 'source_file' in self.df.columns:
            duplicate_subset = [col for col in self.df.columns if col != 'source_file']
            print(f"Analyzing duplicates across {len(duplicate_subset)} columns (excluding source_file)")
        else:
            duplicate_subset = None
            print(f"Analyzing duplicates across all {len(self.df.columns)} columns")
        
        # Find duplicates
        if duplicate_subset:
            duplicates_mask = self.df.duplicated(subset=duplicate_subset, keep=False)
        else:
            duplicates_mask = self.df.duplicated(keep=False)
        
        duplicate_count = duplicates_mask.sum()
        unique_duplicate_groups = len(self.df[duplicates_mask].drop_duplicates(subset=duplicate_subset)) if duplicate_subset else len(self.df[duplicates_mask].drop_duplicates())
        
        print(f"Total duplicate rows: {duplicate_count}")
        print(f"Unique duplicate groups: {unique_duplicate_groups}")
        print(f"Percentage of data that's duplicated: {(duplicate_count/len(self.df)*100):.1f}%")
        
        if duplicate_count > 0:
            # Show breakdown by source file
            if 'source_file' in self.df.columns:
                print(f"\nDuplicates by source file:")
                duplicate_files = self.df[duplicates_mask]['source_file'].value_counts()
                for file_name, count in duplicate_files.items():
                    print(f"  - {file_name}: {count} rows")
            
            # Show examples if requested
            if show_examples and duplicate_count > 0:
                print(f"\nExample duplicate groups (showing first 3):")
                
                if duplicate_subset:
                    duplicate_groups = self.df[duplicates_mask].groupby(duplicate_subset)
                else:
                    duplicate_groups = self.df[duplicates_mask].groupby(list(self.df.columns))
                
                for i, (name, group) in enumerate(duplicate_groups):
                    if i >= 3:  # Show only first 3 groups
                        break
                    print(f"\n  Group {i+1} ({len(group)} duplicates):")
                    if 'horse_name' in self.df.columns and 'race_date' in self.df.columns:
                        print(f"    Horse: {group.iloc[0]['horse_name']}, Date: {group.iloc[0]['race_date']}")
                    if 'source_file' in self.df.columns:
                        files = group['source_file'].tolist()
                        print(f"    Found in files: {', '.join(files)}")
        
        return duplicate_count, unique_duplicate_groups

    def remove_duplicates(self, keep: str = 'first', exclude_source_file: bool = True):
        """
        Remove duplicate rows from the merged dataset.
        
        Args:
            keep (str): Which duplicate to keep ('first', 'last', False for remove all)
            exclude_source_file (bool): If True, ignore source_file column when checking for duplicates
        """
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        print(f"\nChecking for duplicate rows...")
        initial_count = len(self.df)
        
        # Determine which columns to use for duplicate detection
        if exclude_source_file and 'source_file' in self.df.columns:
            # Check duplicates based on all columns except source_file
            duplicate_subset = [col for col in self.df.columns if col != 'source_file']
            print(f"Checking duplicates across {len(duplicate_subset)} columns (excluding source_file)")
        else:
            # Check duplicates based on all columns
            duplicate_subset = None
            print(f"Checking duplicates across all {len(self.df.columns)} columns")
        
        # Find duplicates before removing them
        if duplicate_subset:
            duplicates_mask = self.df.duplicated(subset=duplicate_subset, keep=False)
        else:
            duplicates_mask = self.df.duplicated(keep=False)
        
        duplicate_count = duplicates_mask.sum()
        
        if duplicate_count > 0:
            print(f"Found {duplicate_count} duplicate rows")
            
            # Show breakdown by source file for duplicates
            if 'source_file' in self.df.columns:
                print("Duplicates by source file:")
                duplicate_files = self.df[duplicates_mask]['source_file'].value_counts()
                for file_name, count in duplicate_files.items():
                    print(f"  - {file_name}: {count} duplicate rows")
            
            # Remove duplicates
            self.df = self.df.drop_duplicates(subset=duplicate_subset, keep=keep)
            
            final_count = len(self.df)
            removed_count = initial_count - final_count
            
            print(f"Removed {removed_count} duplicate rows")
            print(f"Dataset size: {initial_count:,} → {final_count:,} rows")
            
        else:
            print("No duplicate rows found")
    
    def clean_data(self):
        """Apply all cleaning transformations to the dataset."""
        if self.df is None:
            print("No data loaded. Call load_data() first.")
            return
        
        print("\nCleaning merged data...")
        
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
        
        print(f"\nGenerating SQL INSERT statements for table '{table_name}'...")
        
        # Create table statement with source_file column
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
    first_place_found BOOLEAN,
    source_file VARCHAR(255)
);"""
        
        self.sql_statements.append(create_table_sql)
        
        # Generate INSERT statements
        for _, row in self.df.iterrows():
            insert_sql = f"""
INSERT INTO {table_name} (
    track_name, race_date, race_time, going, distance, horse_name, 
    jockey, claims, trainer, race_position, lost_by_length, 
    age, weight, rating, first_place_found, source_file
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
    {row['first_place_found_clean']},
    '{row['source_file'].replace("'", "''")}'
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
    
    def save_merged_csv(self, output_file: str = 'merged_horse_racing_data.csv'):
        """Save the cleaned merged data to a CSV file."""
        if self.df is None:
            print("No data to save. Load and clean data first.")
            return False
        
        try:
            # Save the cleaned data with all the clean columns
            self.df.to_csv(output_file, index=False)
            print(f"Merged data saved to {output_file}")
            return True
        except Exception as e:
            print(f"Error saving merged CSV: {e}")
            return False
    
    def print_data_summary(self):
        """Print a summary of the cleaned data."""
        if self.df is None:
            print("No data loaded.")
            return
        
        print("\n" + "="*60)
        print("DATA CLEANING SUMMARY")
        print("="*60)
        
        print(f"Total records: {len(self.df)}")
        print(f"Total source files: {self.df['source_file'].nunique()}")
        
        print(f"\nFiles breakdown:")
        file_counts = self.df['source_file'].value_counts()
        for file_name, count in file_counts.items():
            print(f"  - {file_name}: {count:,} rows")
        
        print(f"\nRace Position Analysis:")
        print(f"  - Winners (1st place): {(self.df['race_position_clean'] == 1).sum():,}")
        print(f"  - Non-finishers (-1): {(self.df['race_position_clean'] == -1).sum():,}")
        print(f"  - Regular positions: {((self.df['race_position_clean'] > 1) & (self.df['race_position_clean'] < 50)).sum():,}")
        
        print(f"\nRating Analysis:")
        print(f"  - Missing ratings (set to 0): {(self.df['rating_clean'] == 0).sum():,}")
        print(f"  - Valid ratings: {(self.df['rating_clean'] > 0).sum():,}")
        
        print(f"\nLost by Length Analysis:")
        print(f"  - Winners (0 length): {(self.df['lost_by_length_clean'] == 0).sum():,}")
        print(f"  - Average lost by length: {self.df['lost_by_length_clean'].mean():.2f}")
        
        print(f"\nUnique Values:")
        print(f"  - Tracks: {self.df['track_name_clean'].nunique()}")
        print(f"  - Horses: {self.df['horse_name_clean'].nunique()}")
        print(f"  - Jockeys: {self.df['jockey_clean'].nunique()}")
        print(f"  - Trainers: {self.df['trainer_clean'].nunique()}")
        
        # Check for any remaining duplicates after cleaning
        if 'source_file' in self.df.columns:
            duplicate_subset = [col for col in self.df.columns if col != 'source_file' and not col.endswith('_clean')]
            if duplicate_subset:
                remaining_duplicates = self.df.duplicated(subset=duplicate_subset).sum()
                if remaining_duplicates > 0:
                    print(f"\nWarning: {remaining_duplicates} potential duplicates still remain in original data columns")

def main():
    """Main function to process the horse racing data from a folder."""
    # Initialize processor with folder path
    folder_path = input("Enter the folder path containing CSV files (or press Enter for 'csv'): ").strip()
    if not folder_path:
        folder_path = 'csv'
    
    try:
        processor = HorseRacingDataProcessor(folder_path)
        
        # Load and process data
        if processor.load_data():
            # First analyze duplicates
            duplicate_count, _ = processor.analyze_duplicates()
            
            if duplicate_count > 0:
                print(f"\nDuplicate removal options:")
                print("1. Remove duplicates (keep first occurrence)")
                print("2. Remove duplicates (keep last occurrence)")
                print("3. Keep all rows (including duplicates)")
                
                dup_choice = input("Enter your choice (1-3, or press Enter for option 1): ").strip()
                if not dup_choice:
                    dup_choice = '1'
                
                if dup_choice == '1':
                    processor.remove_duplicates(keep='first')
                elif dup_choice == '2':
                    processor.remove_duplicates(keep='last')
                else:
                    print("Keeping all rows including duplicates")
            
            processor.clean_data()
            processor.print_data_summary()
            
            # Ask user what outputs they want
            print(f"\nChoose output options:")
            print("1. Generate SQL statements")
            print("2. Save merged CSV")
            print("3. Both")
            
            choice = input("Enter your choice (1-3, or press Enter for both): ").strip()
            if not choice:
                choice = '3'
            
            if choice in ['1', '3']:
                processor.generate_sql_statements()
                processor.save_sql_to_file()
            
            if choice in ['2', '3']:
                processor.save_merged_csv()
            
            print("\nProcessing completed successfully!")
        else:
            print("Failed to process data.")
            
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
# Usage Example and Requirements

## Requirements
# pip install pandas

## Simple Usage Example

from csv2sql import HorseRacingDataProcessor

# Quick usage
def quick_process(csv_file_path):
    """Process a CSV file and generate SQL statements in one go."""
    processor = HorseRacingDataProcessor(csv_file_path)
    
    if processor.load_data():
        processor.clean_data()
        processor.print_data_summary()
        # Generate MySQL-compatible SQL
        processor.generate_sql_statements(table_name='racing_results')
        processor.save_sql_to_file('horse_racing_inserts.sql')
        return True
    return False

# Advanced usage with custom table name and output file
def advanced_process():
    """Example with more control over the process."""
    processor = HorseRacingDataProcessor('csv/test.csv')
    
    # Load data
    if not processor.load_data():
        print("Failed to load data")
        return
    
    # Clean data
    processor.clean_data()
    
    # Generate SQL for a custom table (MySQL format)
    processor.generate_sql_statements(table_name='racing_results')
    
    # Save to custom file
    processor.save_sql_to_file('racing_results_insert.sql')
    
    # Print summary
    processor.print_data_summary()
    
    # Access the cleaned DataFrame for further analysis
    cleaned_df = processor.df
    print(f"\nCleaned DataFrame shape: {cleaned_df.shape}")
    
    # Example: Show all winners
    winners = cleaned_df[cleaned_df['race_position_clean'] == 1]
    print(f"Number of winners: {len(winners)}")

# Database-specific examples
def generate_for_different_databases():
    """Generate SQL for different database types."""
    processor = HorseRacingDataProcessor('csv/test.csv')
    
    if processor.load_data():
        processor.clean_data()
        
        # Generate MySQL/MariaDB SQL
        processor.sql_statements = []  # Clear previous statements
        processor.generate_sql_statements(database_type='mysql')
        processor.save_sql_to_file('mysql_racing_data.sql')
        
        # Generate PostgreSQL SQL
        processor.sql_statements = []  # Clear previous statements
        processor.generate_sql_statements(database_type='postgresql')
        processor.save_sql_to_file('postgresql_racing_data.sql')
        
        # Generate SQLite SQL
        processor.sql_statements = []  # Clear previous statements
        processor.generate_sql_statements(database_type='sqlite')
        processor.save_sql_to_file('sqlite_racing_data.sql')
        
        print("Generated SQL files for MySQL, PostgreSQL, and SQLite")

# Direct function call for simple cases
def process_horse_racing_csv(input_file, output_sql_file=None, table_name='horse_racing_data', database_type='mysql'):
    """
    Simple function to process a CSV file and generate SQL statements.
    
    Args:
        input_file (str): Path to the input CSV file
        output_sql_file (str, optional): Path for the output SQL file
        table_name (str): Name for the database table
        database_type (str): 'mysql', 'sqlite', or 'postgresql'
        
    Returns:
        bool: True if successful, False otherwise
    """
    if output_sql_file is None:
        output_sql_file = input_file.replace('.csv', '_inserts.sql')
    
    processor = HorseRacingDataProcessor(input_file)
    
    if processor.load_data():
        processor.clean_data()
        processor.generate_sql_statements(table_name, database_type)
        return processor.save_sql_to_file(output_sql_file)
    
    return False

if __name__ == "__main__":
    # Example 1: Quick processing
    print("Example 1: Quick processing")
    quick_process('csv/test.csv')
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Advanced processing
    print("Example 2: Advanced processing")
    advanced_process()
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Direct function call for MySQL
    print("Example 3: Direct function call for MySQL")
    success = process_horse_racing_csv('csv/test.csv', 'my_racing_data.sql', 'race_results', 'mysql')
    print(f"Processing successful: {success}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 4: Generate for different databases
    print("Example 4: Generate for different databases")
    generate_for_different_databases()

# Data Transformation Summary:
"""
PREPROCESSING:
- Removes duplicate rows automatically during data loading
- Reports how many duplicates were found and removed

TRACK NAME TRANSFORMATIONS:
- "Sligo Racing Results | 10th June 2025 17:58" → "Sligo 10th June 2025 17:58"
- Removes "Racing Results" text
- Removes "|" character
- Cleans up extra whitespace

RACE POSITION TRANSFORMATIONS:
- "1st", "2nd", "3rd", etc. → 1, 2, 3, etc.
- "PU" (Pulled Up) → -1
- "NR" (Non Runner) → -1  
- "RO" (Ran Out) → -1
- "UR" (Unseated Rider) → -1
- "F" (Fell) → -1
- "BD" (Brought Down) → -1

LOST BY LENGTH TRANSFORMATIONS:
- Winners (1st place) → 0.0
- "1 l" → 1.0
- "1/2 l" → 0.5
- "3 1/4 l" → 3.25
- "Nose" → 0.05
- "Head" → 0.1
- "Neck" → 0.2
- "N/A", empty, or "-" → 0.0

RATING TRANSFORMATIONS:
- "-" → 0
- Empty/null → 0
- Valid numbers → kept as integers

WEIGHT TRANSFORMATIONS:
- Removes trailing letters (e.g., "11-0v" → "11-0")
- Empty/null → "0-0"

STRING FIELD TRANSFORMATIONS:
- Empty/null/"-" → "Unknown"
- Trims whitespace
- Escapes single quotes for SQL

BOOLEAN TRANSFORMATIONS:
- "True" → 1
- Everything else → 0

DATABASE COMPATIBILITY:
- MySQL/MariaDB: AUTO_INCREMENT, INT, TINYINT(1)
- PostgreSQL: SERIAL, INTEGER, BOOLEAN  
- SQLite: AUTOINCREMENT, INTEGER, BOOLEAN
"""
#!/usr/bin/env python3
"""
Daily Racing Data Master Script
Automatically collects today's race URLs and scrapes horse data
Perfect for daily automation
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

class DailyRacingMaster:
    def __init__(self, output_dir="daily_racing_data", debug=False):
        """
        Initialize the daily racing master
        
        Args:
            output_dir (str): Directory to store all outputs
            debug (bool): Enable debug mode
        """
        self.output_dir = Path(output_dir)
        self.debug = debug
        self.today = datetime.now().strftime('%Y-%m-%d')
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # File paths for today
        self.irish_csv = self.output_dir / f"irish_races_{self.today}.csv"
        self.uk_csv = self.output_dir / f"uk_races_{self.today}.csv"
        self.combined_csv = self.output_dir / f"all_races_{self.today}.csv"
        self.irish_results = self.output_dir / f"irish_horses_{self.today}.csv"
        self.uk_results = self.output_dir / f"uk_horses_{self.today}.csv"
        self.combined_results = self.output_dir / f"all_horses_{self.today}.csv"
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"daily_racing_{self.today}.log"
        
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run_url_collector(self, date=None):
        """
        Run the URL collector for specified date (defaults to today)
        
        Args:
            date (str): Date in YYYY-MM-DD format (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if date is None:
            date = self.today
            
        self.logger.info(f"Starting URL collection for {date}")
        
        # Build command for racing_url_collector.py
        cmd = [
            sys.executable, "racing_url_collector.py",
            "--start-date", date,
            "--end-date", date,
            "--headless",
            "--irish-output", str(self.irish_csv),
            "--uk-output", str(self.uk_csv)
        ]
        
        if self.debug:
            cmd.append("--debug")
            
        try:
            # Run the URL collector
            self.logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            if result.returncode == 0:
                self.logger.info("URL collection completed successfully")
                self.logger.info(f"STDOUT: {result.stdout}")
                return True
            else:
                self.logger.error(f"URL collection failed with return code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("URL collection timed out after 10 minutes")
            return False
        except Exception as e:
            self.logger.error(f"Error running URL collector: {e}")
            return False
    
    def combine_csv_files(self):
        """Combine Irish and UK CSV files into one"""
        try:
            dfs = []
            
            if self.irish_csv.exists():
                irish_df = pd.read_csv(self.irish_csv)
                irish_df['country'] = 'Ireland'
                dfs.append(irish_df)
                self.logger.info(f"Loaded {len(irish_df)} Irish races")
            else:
                self.logger.warning(f"Irish CSV not found: {self.irish_csv}")
            
            if self.uk_csv.exists():
                uk_df = pd.read_csv(self.uk_csv)
                uk_df['country'] = 'UK'
                dfs.append(uk_df)
                self.logger.info(f"Loaded {len(uk_df)} UK races")
            else:
                self.logger.warning(f"UK CSV not found: {self.uk_csv}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df.to_csv(self.combined_csv, index=False)
                self.logger.info(f"Combined CSV saved: {self.combined_csv} ({len(combined_df)} total races)")
                return True
            else:
                self.logger.error("No CSV files to combine")
                return False
                
        except Exception as e:
            self.logger.error(f"Error combining CSV files: {e}")
            return False
    
    def run_horse_scraper(self, csv_file, output_file):
        """
        Run the horse scraper on a specific CSV file
        
        Args:
            csv_file (Path): Input CSV file path
            output_file (Path): Output CSV file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not csv_file.exists():
            self.logger.warning(f"CSV file not found: {csv_file}")
            return False
            
        self.logger.info(f"Starting horse scraping for {csv_file}")
        
        # Build command for Whole9.py
        cmd = [
            sys.executable, "Whole9.py",
            "--csv", str(csv_file),
            "--output", str(output_file),
            "--headless",
            "--delay", "3"  # Be nice to the server
        ]
        
        if self.debug:
            cmd.append("--verbose")
            
        try:
            # Run the horse scraper
            self.logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                self.logger.info(f"Horse scraping completed successfully for {csv_file}")
                self.logger.info(f"STDOUT: {result.stdout}")
                return True
            else:
                self.logger.error(f"Horse scraping failed for {csv_file} with return code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Horse scraping timed out after 1 hour for {csv_file}")
            return False
        except Exception as e:
            self.logger.error(f"Error running horse scraper for {csv_file}: {e}")
            return False
    
    def combine_horse_results(self):
        """Combine Irish and UK horse results into one file"""
        try:
            dfs = []
            
            if self.irish_results.exists():
                irish_df = pd.read_csv(self.irish_results)
                irish_df['country'] = 'Ireland'
                dfs.append(irish_df)
                self.logger.info(f"Loaded {len(irish_df)} Irish horse records")
            else:
                self.logger.warning(f"Irish results not found: {self.irish_results}")
            
            if self.uk_results.exists():
                uk_df = pd.read_csv(self.uk_results)
                uk_df['country'] = 'UK'
                dfs.append(uk_df)
                self.logger.info(f"Loaded {len(uk_df)} UK horse records")
            else:
                self.logger.warning(f"UK results not found: {self.uk_results}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df.to_csv(self.combined_results, index=False)
                self.logger.info(f"Combined horse results saved: {self.combined_results} ({len(combined_df)} total horses)")
                return True
            else:
                self.logger.error("No horse result files to combine")
                return False
                
        except Exception as e:
            self.logger.error(f"Error combining horse results: {e}")
            return False
    
    def get_summary(self):
        """Generate summary of today's data collection"""
        summary = {
            'date': self.today,
            'irish_races': 0,
            'uk_races': 0,
            'total_races': 0,
            'irish_horses': 0,
            'uk_horses': 0,
            'total_horses': 0,
            'files_created': []
        }
        
        try:
            # Count races
            if self.irish_csv.exists():
                irish_df = pd.read_csv(self.irish_csv)
                summary['irish_races'] = len(irish_df)
                summary['files_created'].append(str(self.irish_csv))
            
            if self.uk_csv.exists():
                uk_df = pd.read_csv(self.uk_csv)
                summary['uk_races'] = len(uk_df)
                summary['files_created'].append(str(self.uk_csv))
            
            if self.combined_csv.exists():
                summary['files_created'].append(str(self.combined_csv))
            
            summary['total_races'] = summary['irish_races'] + summary['uk_races']
            
            # Count horses
            if self.irish_results.exists():
                irish_horses_df = pd.read_csv(self.irish_results)
                summary['irish_horses'] = len(irish_horses_df)
                summary['files_created'].append(str(self.irish_results))
            
            if self.uk_results.exists():
                uk_horses_df = pd.read_csv(self.uk_results)
                summary['uk_horses'] = len(uk_horses_df)
                summary['files_created'].append(str(self.uk_results))
            
            if self.combined_results.exists():
                summary['files_created'].append(str(self.combined_results))
            
            summary['total_horses'] = summary['irish_horses'] + summary['uk_horses']
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
        
        return summary
    
    def run_daily_collection(self, date=None):
        """
        Run the complete daily collection process
        
        Args:
            date (str): Date to collect data for (defaults to today)
            
        Returns:
            dict: Summary of the collection process
        """
        if date:
            self.today = date
            # Update file paths for the specified date
            self.irish_csv = self.output_dir / f"irish_races_{self.today}.csv"
            self.uk_csv = self.output_dir / f"uk_races_{self.today}.csv"
            self.combined_csv = self.output_dir / f"all_races_{self.today}.csv"
            self.irish_results = self.output_dir / f"irish_horses_{self.today}.csv"
            self.uk_results = self.output_dir / f"uk_horses_{self.today}.csv"
            self.combined_results = self.output_dir / f"all_horses_{self.today}.csv"
        
        self.logger.info(f"=== Starting Daily Racing Collection for {self.today} ===")
        
        success_steps = []
        
        # Step 1: Collect URLs
        self.logger.info("Step 1: Collecting race URLs...")
        if self.run_url_collector(self.today):
            success_steps.append("URL Collection")
            
            # Step 2: Combine URL files
            self.logger.info("Step 2: Combining URL files...")
            if self.combine_csv_files():
                success_steps.append("URL Combination")
            
            # Step 3: Scrape Irish horses (if file exists)
            if self.irish_csv.exists():
                self.logger.info("Step 3: Scraping Irish horse data...")
                if self.run_horse_scraper(self.irish_csv, self.irish_results):
                    success_steps.append("Irish Horse Scraping")
            
            # Step 4: Scrape UK horses (if file exists)
            if self.uk_csv.exists():
                self.logger.info("Step 4: Scraping UK horse data...")
                if self.run_horse_scraper(self.uk_csv, self.uk_results):
                    success_steps.append("UK Horse Scraping")
            
            # Step 5: Combine horse results
            if self.irish_results.exists() or self.uk_results.exists():
                self.logger.info("Step 5: Combining horse results...")
                if self.combine_horse_results():
                    success_steps.append("Horse Results Combination")
        
        # Generate final summary
        summary = self.get_summary()
        summary['successful_steps'] = success_steps
        summary['total_steps'] = 5
        summary['success_rate'] = f"{len(success_steps)}/5 steps completed"
        
        self.logger.info(f"=== Daily Collection Complete ===")
        self.logger.info(f"Success Rate: {summary['success_rate']}")
        self.logger.info(f"Irish Races: {summary['irish_races']}, UK Races: {summary['uk_races']}")
        self.logger.info(f"Irish Horses: {summary['irish_horses']}, UK Horses: {summary['uk_horses']}")
        self.logger.info(f"Files Created: {len(summary['files_created'])}")
        
        return summary


def main():
    """Main function with CLI"""
    parser = argparse.ArgumentParser(
        description="Daily Racing Data Master - Collect and scrape today's racing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run for today
  %(prog)s --date 2025-06-10        # Run for specific date
  %(prog)s --output-dir ./data       # Custom output directory
  %(prog)s --debug                   # Enable debug mode
        """
    )
    
    parser.add_argument(
        "--date", "-d",
        help="Date to collect data for (YYYY-MM-DD format, defaults to today)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="daily_racing_data",
        help="Output directory for all files (default: daily_racing_data)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Validate date if provided
    if args.date:
        try:
            datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print("Error: Date must be in YYYY-MM-DD format")
            return 1
    
    try:
        # Create master instance
        master = DailyRacingMaster(output_dir=args.output_dir, debug=args.debug)
        
        # Run daily collection
        summary = master.run_daily_collection(args.date)
        
        # Print final summary
        print("\n" + "="*60)
        print(f"DAILY RACING COLLECTION SUMMARY - {summary['date']}")
        print("="*60)
        print(f"Success Rate: {summary['success_rate']}")
        print(f"Successful Steps: {', '.join(summary['successful_steps'])}")
        print(f"\nRaces Collected:")
        print(f"  Irish: {summary['irish_races']}")
        print(f"  UK: {summary['uk_races']}")
        print(f"  Total: {summary['total_races']}")
        print(f"\nHorses Scraped:")
        print(f"  Irish: {summary['irish_horses']}")
        print(f"  UK: {summary['uk_horses']}")
        print(f"  Total: {summary['total_horses']}")
        print(f"\nFiles Created: {len(summary['files_created'])}")
        for file_path in summary['files_created']:
            print(f"  - {file_path}")
        
        # Exit code based on success
        if len(summary['successful_steps']) >= 3:  # At least URL collection + some scraping
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


"""
==============================================
              DAILY USAGE EXAMPLES
==============================================

# Basic daily run (for today)
python daily_racing_master.py

# Run for a specific date
python daily_racing_master.py --date 2025-06-10

# Run with custom output directory
python daily_racing_master.py --output-dir /path/to/data

# Run with debug output
python daily_racing_master.py --debug

# For automation (add to crontab)
# Run every day at 6 AM
0 6 * * * /usr/bin/python3 /path/to/daily_racing_master.py

# For automation with logging
0 6 * * * /usr/bin/python3 /path/to/daily_racing_master.py >> /var/log/daily_racing.log 2>&1

==============================================
                 OUTPUT FILES
==============================================

For each day, the following files are created:
- irish_races_YYYY-MM-DD.csv      (Irish race URLs)
- uk_races_YYYY-MM-DD.csv         (UK race URLs)  
- all_races_YYYY-MM-DD.csv        (Combined race URLs)
- irish_horses_YYYY-MM-DD.csv     (Irish horse data)
- uk_horses_YYYY-MM-DD.csv        (UK horse data)
- all_horses_YYYY-MM-DD.csv       (Combined horse data)
- daily_racing_YYYY-MM-DD.log     (Process log)

==============================================
                ERROR HANDLING
==============================================

The script handles various failure scenarios:
- Network timeouts
- Missing race data
- Scraping failures
- File I/O errors

Each step is logged and the process continues with
available data. Check the log file for details.
"""
#!/usr/bin/env python3
"""
Daily Racing Data Master Script - WITH DATE RANGE SUPPORT
Automatically collects race URLs and scrapes horse data for date ranges
Perfect for daily automation or historical data collection
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
NULL_DEVICE = 'NUL'


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
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"daily_racing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.DEBUG if self.debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_date_range(self, start_date, end_date):
        """
        Generate list of dates between start and end
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            list: List of date strings in YYYY-MM-DD format
        """
        dates = []
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        
        return dates
        
    def run_url_collector(self, start_date, end_date, irish_csv, uk_csv):
        """
        Run the URL collector for specified date range
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            irish_csv (Path): Output path for Irish races CSV
            uk_csv (Path): Output path for UK races CSV
            
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info(f"Starting URL collection for {start_date} to {end_date}")
        
        # Build command for racing_url_collector.py
        cmd = [
            sys.executable, "racing_url_collector.py",
            "--start-date", start_date,
            "--end-date", end_date,
            "--headless",
            "--irish-output", str(irish_csv),
            "--uk-output", str(uk_csv)
        ]
        
        if self.debug:
            cmd.append("--debug")
            
        try:
            # Run the URL collector
            self.logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
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
            "--delay", "3"
        ]
        
        if self.debug:
            cmd.append("--verbose")
            
        try:
            # Run the horse scraper
            self.logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
            
            if result.returncode == 0:
                self.logger.info(f"Horse scraping completed successfully for {csv_file}")
                self.logger.info(f"STDOUT: {result.stdout}")
                return True
            else:
                self.logger.error(f"Horse scraping failed for {csv_file} with return code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Horse scraping timed out after 2 hours for {csv_file}")
            return False
        except Exception as e:
            self.logger.error(f"Error running horse scraper for {csv_file}: {e}")
            return False
    
    def combine_csv_files(self, irish_csv, uk_csv, combined_csv):
        """Combine Irish and UK CSV files into one"""
        try:
            dfs = []
            
            if irish_csv.exists():
                irish_df = pd.read_csv(irish_csv)
                irish_df['country'] = 'Ireland'
                dfs.append(irish_df)
                self.logger.info(f"Loaded {len(irish_df)} Irish races")
            else:
                self.logger.warning(f"Irish CSV not found: {irish_csv}")
            
            if uk_csv.exists():
                uk_df = pd.read_csv(uk_csv)
                uk_df['country'] = 'UK'
                dfs.append(uk_df)
                self.logger.info(f"Loaded {len(uk_df)} UK races")
            else:
                self.logger.warning(f"UK CSV not found: {uk_csv}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df.to_csv(combined_csv, index=False)
                self.logger.info(f"Combined CSV saved: {combined_csv} ({len(combined_df)} total races)")
                return True
            else:
                self.logger.error("No CSV files to combine")
                return False
                
        except Exception as e:
            self.logger.error(f"Error combining CSV files: {e}")
            return False
    
    def combine_horse_results(self, irish_results, uk_results, combined_results):
        """Combine Irish and UK horse results into one file"""
        try:
            dfs = []
            
            if irish_results.exists():
                irish_df = pd.read_csv(irish_results)
                irish_df['country'] = 'Ireland'
                dfs.append(irish_df)
                self.logger.info(f"Loaded {len(irish_df)} Irish horse records")
            else:
                self.logger.warning(f"Irish results not found: {irish_results}")
            
            if uk_results.exists():
                uk_df = pd.read_csv(uk_results)
                uk_df['country'] = 'UK'
                dfs.append(uk_df)
                self.logger.info(f"Loaded {len(uk_df)} UK horse records")
            else:
                self.logger.warning(f"UK results not found: {uk_results}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df.to_csv(combined_results, index=False)
                self.logger.info(f"Combined horse results saved: {combined_results} ({len(combined_df)} total horses)")
                return True
            else:
                self.logger.error("No horse result files to combine")
                return False
                
        except Exception as e:
            self.logger.error(f"Error combining horse results: {e}")
            return False
    
    def run_date_range_collection(self, start_date, end_date):
        """
        Run collection for a date range - collects all URLs first, then scrapes all
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: Summary of the collection process
        """
        self.logger.info(f"=== Starting Date Range Collection: {start_date} to {end_date} ===")
        
        # Create file paths for the entire range
        date_range_str = f"{start_date}_to_{end_date}"
        irish_csv = self.output_dir / f"irish_races_{date_range_str}.csv"
        uk_csv = self.output_dir / f"uk_races_{date_range_str}.csv"
        combined_csv = self.output_dir / f"all_races_{date_range_str}.csv"
        irish_results = self.output_dir / f"irish_horses_{date_range_str}.csv"
        uk_results = self.output_dir / f"uk_horses_{date_range_str}.csv"
        combined_results = self.output_dir / f"all_horses_{date_range_str}.csv"
        
        success_steps = []
        
        # Step 1: Collect all URLs for the date range
        self.logger.info("Step 1: Collecting race URLs for entire date range...")
        if self.run_url_collector(start_date, end_date, irish_csv, uk_csv):
            success_steps.append("URL Collection")
            
            # Step 2: Combine URL files
            self.logger.info("Step 2: Combining URL files...")
            if self.combine_csv_files(irish_csv, uk_csv, combined_csv):
                success_steps.append("URL Combination")
            
            # Step 3: Scrape Irish horses (if file exists)
            if irish_csv.exists():
                self.logger.info("Step 3: Scraping Irish horse data...")
                if self.run_horse_scraper(irish_csv, irish_results):
                    success_steps.append("Irish Horse Scraping")
            
            # Step 4: Scrape UK horses (if file exists)
            if uk_csv.exists():
                self.logger.info("Step 4: Scraping UK horse data...")
                if self.run_horse_scraper(uk_csv, uk_results):
                    success_steps.append("UK Horse Scraping")
            
            # Step 5: Combine horse results
            if irish_results.exists() or uk_results.exists():
                self.logger.info("Step 5: Combining horse results...")
                if self.combine_horse_results(irish_results, uk_results, combined_results):
                    success_steps.append("Horse Results Combination")
        
        # Generate summary
        summary = self.get_summary(
            irish_csv, uk_csv, combined_csv,
            irish_results, uk_results, combined_results,
            success_steps, start_date, end_date
        )
        
        self.logger.info(f"=== Date Range Collection Complete ===")
        self.logger.info(f"Success Rate: {summary['success_rate']}")
        self.logger.info(f"Date Range: {summary['date_range']}")
        self.logger.info(f"Total Races: {summary['total_races']}")
        self.logger.info(f"Total Horses: {summary['total_horses']}")
        
        return summary
    
    def get_summary(self, irish_csv, uk_csv, combined_csv, 
                    irish_results, uk_results, combined_results,
                    success_steps, start_date, end_date):
        """Generate summary of data collection"""
        summary = {
            'date_range': f"{start_date} to {end_date}",
            'start_date': start_date,
            'end_date': end_date,
            'irish_races': 0,
            'uk_races': 0,
            'total_races': 0,
            'irish_horses': 0,
            'uk_horses': 0,
            'total_horses': 0,
            'files_created': [],
            'successful_steps': success_steps,
            'total_steps': 5,
            'success_rate': f"{len(success_steps)}/5 steps completed"
        }
        
        try:
            # Count races
            if irish_csv.exists():
                irish_df = pd.read_csv(irish_csv)
                summary['irish_races'] = len(irish_df)
                summary['files_created'].append(str(irish_csv))
            
            if uk_csv.exists():
                uk_df = pd.read_csv(uk_csv)
                summary['uk_races'] = len(uk_df)
                summary['files_created'].append(str(uk_csv))
            
            if combined_csv.exists():
                summary['files_created'].append(str(combined_csv))
            
            summary['total_races'] = summary['irish_races'] + summary['uk_races']
            
            # Count horses
            if irish_results.exists():
                irish_horses_df = pd.read_csv(irish_results)
                summary['irish_horses'] = len(irish_horses_df)
                summary['files_created'].append(str(irish_results))
            
            if uk_results.exists():
                uk_horses_df = pd.read_csv(uk_results)
                summary['uk_horses'] = len(uk_horses_df)
                summary['files_created'].append(str(uk_results))
            
            if combined_results.exists():
                summary['files_created'].append(str(combined_results))
            
            summary['total_horses'] = summary['irish_horses'] + summary['uk_horses']
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
        
        return summary


def main():
    """Main function with CLI"""
    parser = argparse.ArgumentParser(
        description="Daily Racing Data Master - Collect and scrape racing data for date ranges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for today only
  %(prog)s
  
  # Run for a single specific date
  %(prog)s --date 2025-06-10
  
  # Run for a date range
  %(prog)s --start-date 2025-06-10 --end-date 2025-06-15
  
  # Run from specific date to today
  %(prog)s --start-date 2025-06-01 --end-date today
  
  # Custom output directory with debug
  %(prog)s --start-date 2025-06-10 --end-date 2025-06-15 --output-dir ./data --debug
        """
    )
    
    parser.add_argument(
        "--date", "-d",
        help="Single date to collect data for (YYYY-MM-DD format)"
    )
    
    parser.add_argument(
        "--start-date", "-s",
        help="Start date for range collection (YYYY-MM-DD format)"
    )
    
    parser.add_argument(
        "--end-date", "-e",
        help="End date for range collection (YYYY-MM-DD format or 'today')"
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
    
    # Determine date range
    if args.date:
        # Single date mode
        try:
            datetime.strptime(args.date, '%Y-%m-%d')
            start_date = args.date
            end_date = args.date
        except ValueError:
            print("Error: Date must be in YYYY-MM-DD format")
            return 1
    elif args.start_date and args.end_date:
        # Date range mode
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
            start_date = args.start_date
            
            # Handle 'today' keyword
            if args.end_date.lower() == 'today':
                end_date = datetime.now().strftime('%Y-%m-%d')
            else:
                datetime.strptime(args.end_date, '%Y-%m-%d')
                end_date = args.end_date
        except ValueError:
            print("Error: Dates must be in YYYY-MM-DD format")
            return 1
    elif args.start_date:
        # Start date only - assume end date is today
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
            start_date = args.start_date
            end_date = datetime.now().strftime('%Y-%m-%d')
        except ValueError:
            print("Error: Start date must be in YYYY-MM-DD format")
            return 1
    else:
        # Default to today
        start_date = datetime.now().strftime('%Y-%m-%d')
        end_date = start_date
    
    # Validate date range
    if datetime.strptime(start_date, '%Y-%m-%d') > datetime.strptime(end_date, '%Y-%m-%d'):
        print("Error: Start date must be before or equal to end date")
        return 1
    
    try:
        # Create master instance
        master = DailyRacingMaster(output_dir=args.output_dir, debug=args.debug)
        
        # Run date range collection
        summary = master.run_date_range_collection(start_date, end_date)
        
        # Print final summary
        print("\n" + "="*60)
        print(f"RACING COLLECTION SUMMARY")
        print("="*60)
        print(f"Date Range: {summary['date_range']}")
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
        if len(summary['successful_steps']) >= 3:
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
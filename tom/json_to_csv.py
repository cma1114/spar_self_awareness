#!/usr/bin/env python3
"""
Convert JSON game data files to CSV format for analysis.

Usage:
    python json_to_csv.py [directory] [output_file.csv]
    
Arguments:
    directory    - Directory containing JSON files (default: current directory)
    output_file  - Output CSV file (default: combined_game_data.csv)
    
The script will:
- Process all .json files in the specified directory
- Add a 'source_file' column as the first column
- Combine all data into a single CSV file
"""

import json
import csv
import sys
import os
import glob

def json_to_csv(directory=".", csv_file="combined_game_data.csv"):
    """Convert all JSON game data files in directory to a single CSV."""
    
    # Find all JSON files in directory
    json_pattern = os.path.join(directory, "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in directory: {directory}")
        return
    
    print(f"Found {len(json_files)} JSON file(s) in {directory}")
    
    # Collect all data from all files
    all_data = []
    all_keys = set(['source_file'])  # Start with source_file column
    
    for json_file in sorted(json_files):
        filename = os.path.basename(json_file)
        print(f"Processing: {filename}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if not data:
                print(f"  Warning: {filename} is empty, skipping")
                continue
            
            # Handle both formats: 
            # - Old format: array of records directly
            # - New format: {"game_setup": "...", "turn_records": [...]}
            if isinstance(data, dict) and 'turn_records' in data:
                records = data['turn_records']
            elif isinstance(data, list):
                records = data
            else:
                print(f"  Warning: {filename} has unexpected format, skipping")
                continue
            
            # Add source_file to each record and collect keys
            for record in records:
                record['source_file'] = filename
                all_keys.update(record.keys())
                all_data.append(record)
            
            print(f"  Added {len(records)} records")
            
        except json.JSONDecodeError as e:
            print(f"  Error: Could not parse {filename} as JSON: {e}")
            continue
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue
    
    if not all_data:
        print("No data to write")
        return
    
    # Sort fieldnames with source_file first, then alphabetically
    fieldnames = ['source_file'] + sorted([k for k in all_keys if k != 'source_file'])
    
    # Write to CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"\nSuccessfully created {csv_file}")
    print(f"Total records: {len(all_data)}")
    print(f"Total columns: {len(fieldnames)}")
    print(f"Files processed: {len(json_files)}")

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    csv_file = sys.argv[2] if len(sys.argv) > 2 else "combined_game_data.csv"
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found")
        sys.exit(1)
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory")
        sys.exit(1)
    
    json_to_csv(directory, csv_file)

"""This script combines all the scraped CSV files from the "fox_scraped" directory into a single CSV file for easier analysis. 
It reads each individual CSV, concatenates them into one DataFrame, and saves the combined DataFrame as "fox_scraped_all.csv".
The original scraper produced some failed files that needed to be retried, so this script allows us to easily manage and combine the successful scrapes without manual copying and pasting.
"""

import pandas as pd
from pathlib import Path

# Define output path
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "fox_scraped_all.csv"

# Write down all scraped CSV files to concatenate
FILENAMES = [
    "fox_scraped_20260401_224625.csv"
]

# Construct full paths
SCRAPED_DIR = Path(__file__).parent.parent / "data" / "raw"/ "fox_scraped"
filepaths = [SCRAPED_DIR / filename for filename in FILENAMES]


def main(filepaths: list[Path], output_path: Path = OUTPUT_PATH, filenames: list[str] = FILENAMES) -> None:

    # Read and concatentate all CSVs
    full_df = pd.concat([pd.read_csv(filepath) for filepath in filepaths], ignore_index = True)

    # Save combined CSV
    full_df.to_csv(output_path, index = False)
    print(f"Combined {len(filenames)} files into {output_path} with {len(full_df)} total rows.")
    
if __name__ == "__main__":
    main(filepaths = filepaths)
import pandas as pd
from pathlib import Path

FOX = Path(__file__).parent.parent / "data" / "raw" / "fox_scraped_all.csv"
NBC = Path(__file__).parent.parent / "data" / "raw" / "nbc_scraped_all.csv"

# Combine into one df
def main(fox_path: Path = FOX, nbc_path: Path = NBC) -> None:
    fox_df = pd.read_csv(fox_path)
    nbc_df = pd.read_csv(nbc_path)

    combined_df = pd.concat([fox_df, nbc_df], ignore_index = True)
    combined_df.to_csv(Path(__file__).parent.parent / "data" / "raw" / "combined_scraped_all.csv", index = False)
    print(f"Combined {len(fox_df)} rows from FOX and {len(nbc_df)} rows from NBC into {len(combined_df)} total rows.")
    
    
if __name__ == "__main__":
    main()
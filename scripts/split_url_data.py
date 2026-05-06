"""This script reads a CSV file containing URLs of news articles, identifies which URLs belong to Fox News and which belong to NBC News, 
and then splits the data into two separate CSV files: one for Fox News URLs and another for NBC News URLs. 
This allows us to easily manage and analyze the two datasets separately in subsequent steps of our project.
"""

import pandas as pd
from pathlib import Path

URL_ONLY_DATA_PATH = Path(__file__).parent / "data" / "external" / "url_only_data.csv"
FOX_OUTPUT_PATH = Path(__file__).parent / "data" / "interim" / "fox_news_urls.csv"
NBC_OUTPUT_PATH = Path(__file__).parent / "data" / "interim" / "nbc_news_urls.csv"

def split_url_data(url_only_data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    url_df = pd.read_csv(url_only_data_path)
    
    fox_news_mask = url_df["url"].str.contains("foxnews.com")
    
    fox_df = url_df[fox_news_mask]
    nbc_df = url_df[~fox_news_mask]
    
    return fox_df, nbc_df

if __name__ == "__main__":
    fox_df, nbc_df = split_url_data(str(URL_ONLY_DATA_PATH))
    
    fox_df.to_csv(FOX_OUTPUT_PATH, index=False)
    nbc_df.to_csv(NBC_OUTPUT_PATH, index=False)
#news_download_test.py

import asyncio
import nest_asyncio
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent

# Add parent directory to sys.path if not already present
parent_str = str(parent_dir)
if parent_str not in sys.path:
    sys.path.insert(0, parent_str)

from agent_scripts.buck import Buck

# Enable nested event loops
nest_asyncio.apply()

async def main():
    buck_get_data = Buck()._create_default_data_provider()
    data = await buck_get_data.get_news_data("BHEL.NS")
    
    # Convert news data to DataFrame
    news_df = pd.DataFrame(data['news'])
    
    # Convert pub_date strings to datetime
    news_df['pub_date'] = pd.to_datetime(news_df['pub_date'])
    
    # Sort by publication date
    news_df = news_df.sort_values('pub_date', ascending=False)
    
    print("News data shape:", news_df.shape)
    print("\nFirst few rows:")
    print(news_df[['title', 'source', 'pub_date']].head())
    
    #return news_df
# Run the async function
#news_df = asyncio.run(main())

if __name__ == "__main__":
    asyncio.run(main())

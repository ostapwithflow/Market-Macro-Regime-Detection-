import pandas as pd
from fredapi import Fred
import logging
from typing import List
from config import FRED_API_KEY, FRED_TICKERS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FredClient:
    """
    Handles data extraction from FRED for the Macro ETL Pipeline.
    """
    def __init__(self, api_key: str = FRED_API_KEY):
        self.fred = Fred(api_key=api_key)
        
    def fetch_series(self, series_id: str, start_date: str = None) -> pd.Series:
        """Fetches a single series from FRED."""
        logger.info(f"Fetching {series_id} from FRED...")
        try:
            return self.fred.get_series(series_id, observation_start=start_date)
        except Exception as e:
            logger.error(f"Failed to fetch {series_id}: {e}")
            return pd.Series(dtype='float64')

    def fetch_all_macro_data(self, start_date: str = "2000-01-01") -> pd.DataFrame:
        """
        Fetches all core macro tickers required by the Engine 
        and combines them into a daily DataFrame.
        """
        import numpy as np
        series_dict = {}
        for ticker in FRED_TICKERS.keys():
            series = self.fetch_series(ticker, start_date)
            if not series.empty:
                series.name = ticker
                series_dict[ticker] = series
                
        if not series_dict:
            logger.error("All FRED series failed to fetch.")
            return pd.DataFrame()
            
        # Find a template index from the first successful series
        first_series = next(iter(series_dict.values()))
        template_index = first_series.index
        
        # Populate failed series with NaNs using the template index
        for ticker in FRED_TICKERS.keys():
            if ticker not in series_dict:
                logger.warning(f"Ticker {ticker} failed to fetch. Creating a dummy column of NaNs.")
                series_dict[ticker] = pd.Series(np.nan, index=template_index, name=ticker)
                
        df = pd.concat([series_dict[t] for t in FRED_TICKERS.keys()], axis=1)
        
        # Forward-fill to handle missing daily data (weekends, holidays, and weekly releases)
        # Limit ffill to avoid filling excessively old stale data if a series discontinues
        df = df.ffill(limit=30)
        
        # Drop rows where critical data might still be NaN at the very beginning
        df = df.dropna(subset=['DGS10', 'DGS3MO', 'WALCL'])
        
        return df

if __name__ == "__main__":
    client = FredClient()
    df = client.fetch_all_macro_data(start_date="2020-01-01")
    print(df.tail())

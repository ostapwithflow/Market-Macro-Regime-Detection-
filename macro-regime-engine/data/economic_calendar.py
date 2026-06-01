import pandas as pd
import requests
import logging
from typing import Set
from config import FRED_API_KEY

logger = logging.getLogger(__name__)

def fetch_actual_fomc_dates() -> Set[pd.Timestamp]:
    """
    Fetches actual historical FOMC statement dates from vtasca's GitHub scraper.
    """
    url = "https://raw.githubusercontent.com/vtasca/fed-statement-scraping/master/communications.csv"
    logger.info("Fetching FOMC dates from GitHub communications.csv...")
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'])
        # Filter dates from 2010 to 2026
        dates = df[(df['Date'] >= '2010-01-01') & (df['Date'] <= '2026-12-31')]['Date']
        fomc_dates = set(dates)
        logger.info(f"Loaded {len(fomc_dates)} actual FOMC dates.")
        return fomc_dates
    except Exception as e:
        logger.error(f"Failed to fetch FOMC dates: {e}. Falling back to empty set.")
        return set()

def fetch_actual_fred_vintages(series_id: str, api_key: str = FRED_API_KEY) -> Set[pd.Timestamp]:
    """
    Fetches actual publication/vintage dates for a FRED series.
    These dates represent the exact day the economic release was published.
    """
    url = f"https://api.stlouisfed.org/fred/series/vintagedates?series_id={series_id}&api_key={api_key}&file_type=json"
    logger.info(f"Fetching vintage dates for {series_id} from FRED...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        dates_str = data.get('vintage_dates', [])
        dates = pd.to_datetime(dates_str)
        # Filter dates from 2010 to 2026
        vints = set(dates[(dates >= '2010-01-01') & (dates <= '2026-12-31')])
        logger.info(f"Loaded {len(vints)} vintage dates for {series_id}.")
        return vints
    except Exception as e:
        logger.error(f"Failed to fetch vintage dates for {series_id}: {e}. Falling back to empty set.")
        return set()

class EconomicCalendar:
    """
    Manages economic calendar release dates for FOMC/CPI/NFP event flagging.
    """
    def __init__(self, api_key: str = FRED_API_KEY):
        self.api_key = api_key
        self.fomc_dates = set()
        self.cpi_dates = set()
        self.nfp_dates = set()
        self.all_event_dates = set()
        
    def load_all_events(self):
        """Loads and combines all CPI, NFP, and FOMC event dates."""
        self.fomc_dates = fetch_actual_fomc_dates()
        self.cpi_dates = fetch_actual_fred_vintages("CPIAUCSL", self.api_key)
        self.nfp_dates = fetch_actual_fred_vintages("PAYEMS", self.api_key)
        self.all_event_dates = self.fomc_dates.union(self.cpi_dates).union(self.nfp_dates)
        logger.info(f"Total economic calendar events loaded: {len(self.all_event_dates)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cal = EconomicCalendar()
    cal.load_all_events()
    print("Sample FOMC dates (sorted):", sorted(list(cal.fomc_dates))[:5])
    print("Sample CPI dates (sorted):", sorted(list(cal.cpi_dates))[:5])
    print("Sample NFP dates (sorted):", sorted(list(cal.nfp_dates))[:5])

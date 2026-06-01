import os

# FRED API Configuration
# Set FRED_API_KEY as an environment variable or create a .env file
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# BLS API Configuration (for Event Dummies like CPI/NFP)
BLS_API_KEY = os.getenv("BLS_API_KEY", "")

# Core Macro Tickers for the Engine
FRED_TICKERS = {
    "WALCL": "Fed Total Assets (Millions)",
    "RRPONTSYD": "Reverse Repo (Billions)",
    "WTREGEN": "Treasury General Account (Billions)",  # TGA
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS3MO": "3-Month Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread",
    "T5YIE": "5-Year Breakeven Inflation Rate"
}

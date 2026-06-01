import pandas as pd
import numpy as np

def calculate_net_liquidity(df: pd.DataFrame) -> pd.Series:
    """
    Calculates Net Liquidity = Fed Balance Sheet - RRP - TGA.
    Ensure units match!
    WALCL is in Millions of Dollars.
    RRPONTSYD is in Billions of Dollars.
    WTREGEN is in Billions of Dollars.
    So Net Liquidity (in Billions) = (WALCL / 1000) - RRPONTSYD - WTREGEN
    """
    # Extract and normalize units to Billions
    walcl = df.get('WALCL', pd.Series(0, index=df.index)) / 1000.0
    rrp = df.get('RRPONTSYD', pd.Series(0, index=df.index))
    tga = df.get('WTREGEN', pd.Series(0, index=df.index)) / 1000.0
    
    # RRP and TGA might be NaN in early 2000s, fill with 0 before subtraction
    rrp = rrp.fillna(0)
    tga = tga.fillna(0)
    
    net_liquidity = walcl - rrp - tga
    net_liquidity.name = "Net_Liquidity_B"
    return net_liquidity

def calculate_yield_spread(df: pd.DataFrame, long_leg: str = 'DGS10', short_leg: str = 'DGS3MO') -> pd.Series:
    """
    Calculates the yield curve spread (e.g., 10Y - 3M).
    """
    spread = df[long_leg] - df[short_leg]
    spread.name = f"Spread_{long_leg}_{short_leg}"
    return spread

def ewma_z_score(series: pd.Series, halflife: int = 45) -> pd.Series:
    """
    Calculates the EWMA Z-score of a series to make it stationary.
    mu_t = EWMA(x_t)
    sigma_t = EWMA_STD(x_t)
    Z = (x_t - mu_t) / sigma_t
    """
    ewm_mean = series.ewm(halflife=halflife, min_periods=halflife).mean()
    ewm_std = series.ewm(halflife=halflife, min_periods=halflife).std()
    
    z_score = (series - ewm_mean) / ewm_std
    return z_score

def build_emission_vector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the final X_t emission vector for the HMM.
    X_t = [L_t, S_t]
    """
    # Rule 9: Assert positive/non-negative prices for key raw macro variables
    assert (df['DGS10'] >= 0).all(), "DGS10 yield contains negative values"
    assert (df['DGS3MO'] >= 0).all(), "DGS3MO yield contains negative values"
    assert (df['WALCL'] > 0).all(), "WALCL contains zero or negative values"
    
    net_liquidity = calculate_net_liquidity(df)
    
    # Rule 9: Assert Net Liquidity bounds (in Trillions of Dollars)
    # Divided by 1000.0 because WALCL is Millions, RRP/TGA are Billions -> net_liq is Billions.
    # We check the latest value. Lower bound is adjusted to 1.0 to support 2010 data.
    latest_net_liq_trillions = net_liquidity.iloc[-1] / 1000.0
    assert 1.0 < latest_net_liq_trillions < 12.0, f"Net Liquidity out of bounds: {latest_net_liq_trillions:.2f}T"
    
    spread = calculate_yield_spread(df, 'DGS10', 'DGS3MO')
    
    # Calculate EWMA Z-Scores (L_t and S_t)
    # Using 60-day halflife as per MACRO_MATH_LOGIC.md recommendations
    L_t = ewma_z_score(net_liquidity, halflife=60) 
    S_t = ewma_z_score(spread, halflife=60)
    
    L_t.name = 'L_t'
    S_t.name = 'S_t'
    
    X_t = pd.concat([L_t, S_t], axis=1)
    
    # Drop NaNs created by the min_periods in EWMA
    X_t = X_t.dropna()
    
    # Rule 9: Assert no NaNs in final features
    assert not X_t.isnull().any().any(), "NaN values found in HMM features"
    
    return X_t

if __name__ == "__main__":
    from fred_client import FredClient
    client = FredClient()
    raw_df = client.fetch_all_macro_data("2010-01-01")
    X_t = build_emission_vector(raw_df)
    print("--- Emission Vector X_t Head ---")
    print(X_t.head())
    print("\n--- Emission Vector X_t Tail ---")
    print(X_t.tail())

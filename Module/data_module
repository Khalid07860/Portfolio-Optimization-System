"""
Data Module: Fetches and manages stock data with proper validation
"""
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches and manages historical stock data"""
    
    def __init__(self):
        self.data = None
        self.tickers = None
        
    def fetch_data(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical stock data
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            
        Returns:
            DataFrame with adjusted close prices
        """
        logger.info(f"Fetching data for {len(tickers)} tickers")
        
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                raise ValueError("No data fetched")
            
            # Extract Close prices
            if len(tickers) == 1:
                prices = data[['Close']].copy()
                prices.columns = tickers
            else:
                prices = data['Close'].copy()
            
            # Clean data
            prices = self._clean_data(prices)
            
            self.data = prices
            self.tickers = list(prices.columns)
            
            logger.info(f"Successfully fetched {len(prices)} rows")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data - handle missing values"""
        # Remove columns with >20% missing
        threshold = len(data) * 0.80
        data = data.dropna(axis=1, thresh=threshold)
        
        # Forward fill then backward fill
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Drop remaining NaN rows
        data = data.dropna()
        
        # Remove extreme outliers (>50% daily change)
        returns = data.pct_change()
        mask = (returns.abs() < 0.5).all(axis=1)
        data = data[mask]
        
        logger.info(f"Data cleaned. Shape: {data.shape}")
        return data
    
    def calculate_returns(self, prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate simple returns"""
        if prices is None:
            prices = self.data
        
        if prices is None:
            raise ValueError("No price data available")
        
        returns = prices.pct_change().dropna()
        logger.info(f"Calculated returns. Shape: {returns.shape}")
        
        return returns
    
    def get_latest_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get latest prices"""
        try:
            data = yf.download(tickers, period="1d", progress=False)
            
            if len(tickers) == 1:
                latest_prices = {tickers[0]: float(data['Close'].iloc[-1])}
            else:
                latest_prices = data['Close'].iloc[-1].to_dict()
            
            logger.info(f"Retrieved latest prices for {len(latest_prices)} tickers")
            return latest_prices
            
        except Exception as e:
            logger.error(f"Error getting latest prices: {str(e)}")
            raise
    
    def validate_tickers(self, tickers: List[str]) -> List[str]:
        """Validate tickers"""
        valid_tickers = []
        
        for ticker in tickers:
            try:
                test_data = yf.download(ticker, period="5d", progress=False)
                
                if not test_data.empty and len(test_data) >= 3:
                    valid_tickers.append(ticker)
                else:
                    logger.warning(f"{ticker} has insufficient data")
                    
            except Exception as e:
                logger.warning(f"{ticker} is invalid: {str(e)}")
        
        logger.info(f"Validated {len(valid_tickers)}/{len(tickers)} tickers")
        return valid_tickers
    
    def get_summary_statistics(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate summary statistics"""
        stats = pd.DataFrame({
            'Mean_Daily': returns.mean(),
            'Std_Daily': returns.std(),
            'Annual_Return': returns.mean() * config.TRADING_DAYS_PER_YEAR,
            'Annual_Volatility': returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR),
            'Sharpe': ((returns.mean() * config.TRADING_DAYS_PER_YEAR) - config.DEFAULT_RISK_FREE_RATE) / 
                      (returns.std() * np.sqrt(config.TRADING_DAYS_PER_YEAR)),
            'Min': returns.min(),
            'Max': returns.max(),
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis()
        })
        
        return stats

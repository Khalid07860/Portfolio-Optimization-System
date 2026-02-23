"""
Feature Engineering: Creates ML features from price data
"""
import logging
import pandas as pd
import numpy as np
from typing import Tuple
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features for ML models"""
    
    def __init__(self):
        self.feature_columns = []
        
    def create_features(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical and statistical features
        
        Args:
            prices: Price data
            returns: Returns data
            
        Returns:
            DataFrame with features for each ticker
        """
        logger.info("Creating features...")
        
        features_list = []
        
        for ticker in prices.columns:
            ticker_features = self._create_ticker_features(
                prices[ticker],
                returns[ticker],
                ticker
            )
            features_list.append(ticker_features)
        
        features = pd.concat(features_list, axis=1)
        features = features.dropna()
        
        self.feature_columns = features.columns.tolist()
        logger.info(f"Created {len(self.feature_columns)} features")
        
        return features
    
    def _create_ticker_features(
        self, 
        price_series: pd.Series,
        return_series: pd.Series,
        ticker: str
    ) -> pd.DataFrame:
        """Create features for a single ticker"""
        features = pd.DataFrame(index=price_series.index)
        
        # 1. Moving Averages and ratios
        for period in config.LOOKBACK_PERIODS:
            ma = price_series.rolling(window=period).mean()
            features[f'{ticker}_MA_{period}'] = ma
            features[f'{ticker}_MA_ratio_{period}'] = price_series / ma
        
        # 2. Momentum (rate of change)
        features[f'{ticker}_momentum_5'] = price_series.pct_change(5)
        features[f'{ticker}_momentum_10'] = price_series.pct_change(10)
        features[f'{ticker}_momentum_20'] = price_series.pct_change(20)
        
        # 3. Volatility
        features[f'{ticker}_volatility_20'] = return_series.rolling(window=20).std()
        
        # 4. Returns statistics
        features[f'{ticker}_return_mean_20'] = return_series.rolling(window=20).mean()
        features[f'{ticker}_return_std_20'] = return_series.rolling(window=20).std()
        
        # 5. RSI
        features[f'{ticker}_RSI_14'] = self._calculate_rsi(price_series, 14)
        
        # 6. Bollinger Bands position
        ma_20 = price_series.rolling(window=20).mean()
        std_20 = price_series.rolling(window=20).std()
        bb_upper = ma_20 + (std_20 * 2)
        bb_lower = ma_20 - (std_20 * 2)
        features[f'{ticker}_BB_position'] = (price_series - bb_lower) / (bb_upper - bb_lower)
        
        # 7. Lag features
        for lag in [1, 2, 3, 5]:
            features[f'{ticker}_return_lag_{lag}'] = return_series.shift(lag)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def create_target_variables(
        self, 
        returns: pd.DataFrame,
        forecast_horizon: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create target variables for ML
        
        Args:
            returns: Historical returns
            forecast_horizon: Periods ahead to forecast
            
        Returns:
            (future_returns, future_volatility)
        """
        logger.info(f"Creating targets with horizon={forecast_horizon}")
        
        # Future returns (shift negative = look ahead)
        future_returns = returns.shift(-forecast_horizon)
        
        # Future volatility (rolling std of next N periods)
        future_volatility = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for col in returns.columns:
            # Calculate forward-looking volatility
            rolling_vol = returns[col].rolling(window=forecast_horizon).std().shift(-forecast_horizon)
            future_volatility[col] = rolling_vol
        
        future_returns = future_returns.dropna()
        future_volatility = future_volatility.dropna()
        
        logger.info(f"Targets created: returns {future_returns.shape}, vol {future_volatility.shape}")
        
        return future_returns, future_volatility

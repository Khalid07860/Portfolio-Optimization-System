"""
ML Model: Predicts returns and volatility with PROPER validation
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class MLPredictor:
    """ML predictor with validation"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.historical_means = {}
        self.use_ml = {}
        
    def train_models(
        self,
        features: pd.DataFrame,
        future_returns: pd.DataFrame,
        future_volatility: pd.DataFrame
    ) -> Dict:
        """
        Train ML models with fallback to historical means
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training models for {len(future_returns.columns)} assets...")
        
        metrics = {}
        
        for ticker in future_returns.columns:
            logger.info(f"Training {ticker}...")
            
            # Get features for this ticker
            ticker_features = [col for col in features.columns if col.startswith(ticker)]
            
            if not ticker_features:
                logger.warning(f"No features for {ticker}")
                continue
            
            X = features[ticker_features]
            
            # Train return model
            y_return = future_returns[ticker]
            return_metrics, use_ml_return = self._train_single_model(
                X, y_return, f"{ticker}_return"
            )
            
            # Train volatility model
            y_volatility = future_volatility[ticker]
            vol_metrics, use_ml_vol = self._train_single_model(
                X, y_volatility, f"{ticker}_volatility"
            )
            
            # Store whether to use ML or fallback
            self.use_ml[f"{ticker}_return"] = use_ml_return
            self.use_ml[f"{ticker}_volatility"] = use_ml_vol
            
            # Store historical means as fallback
            self.historical_means[f"{ticker}_return"] = y_return.mean()
            self.historical_means[f"{ticker}_volatility"] = y_volatility.mean()
            
            metrics[ticker] = {
                'return': return_metrics,
                'volatility': vol_metrics,
                'use_ml_return': use_ml_return,
                'use_ml_volatility': use_ml_vol
            }
        
        logger.info(f"Training completed")
        return metrics
    
    def _train_single_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str
    ) -> Tuple[Dict, bool]:
        """
        Train single model with validation
        
        Returns:
            (metrics_dict, use_ml_flag)
        """
        # Align X and y
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        # Remove NaNs
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            logger.warning(f"{model_name}: Insufficient data ({len(X)} samples)")
            return {}, False
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, shuffle=False
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train
        model = RandomForestRegressor(
            n_estimators=config.N_ESTIMATORS,
            max_depth=config.MAX_DEPTH,
            min_samples_split=config.MIN_SAMPLES_SPLIT,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        metrics = {
            'test_mse': test_mse,
            'test_r2': test_r2,
            'n_samples': len(X)
        }
        
        # Decide if ML is good enough (R² > 0.1 for financial data is reasonable)
        use_ml = test_r2 > 0.05
        
        if use_ml:
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            logger.info(f"{model_name}: R²={test_r2:.3f} - Using ML")
        else:
            logger.info(f"{model_name}: R²={test_r2:.3f} - Using historical mean")
        
        return metrics, use_ml
    
    def predict(
        self,
        features: pd.DataFrame,
        tickers: list
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Predict returns and volatility with fallback
        
        Returns:
            (expected_returns, expected_volatility) - ANNUALIZED
        """
        logger.info(f"Making predictions for {len(tickers)} tickers...")
        
        expected_returns = {}
        expected_volatility = {}
        
        for ticker in tickers:
            ticker_features = [col for col in features.columns if col.startswith(ticker)]
            
            if not ticker_features:
                logger.warning(f"No features for {ticker}")
                expected_returns[ticker] = 0.10  # 10% fallback
                expected_volatility[ticker] = 0.20  # 20% fallback
                continue
            
            X = features[ticker_features].iloc[-1:].values
            
            # Predict return
            return_model_name = f"{ticker}_return"
            if return_model_name in self.models and self.use_ml.get(return_model_name, False):
                X_scaled = self.scalers[return_model_name].transform(X)
                pred_return = self.models[return_model_name].predict(X_scaled)[0]
            else:
                # Use historical mean
                pred_return = self.historical_means.get(return_model_name, 0.0004)  # ~10% annually
            
            # Predict volatility
            vol_model_name = f"{ticker}_volatility"
            if vol_model_name in self.models and self.use_ml.get(vol_model_name, False):
                X_scaled = self.scalers[vol_model_name].transform(X)
                pred_vol = self.models[vol_model_name].predict(X_scaled)[0]
            else:
                # Use historical mean
                pred_vol = self.historical_means.get(vol_model_name, 0.015)  # ~24% annually
            
            # Ensure positive volatility
            pred_vol = max(pred_vol, 0.001)
            
            # Clip extreme predictions
            pred_return = np.clip(pred_return, -0.002, 0.004)  # Daily: -50% to +100% annually
            pred_vol = np.clip(pred_vol, 0.005, 0.05)  # Daily: 8% to 80% annually
            
            expected_returns[ticker] = pred_return
            expected_volatility[ticker] = pred_vol
        
        # Convert to series
        expected_returns_series = pd.Series(expected_returns)
        expected_volatility_series = pd.Series(expected_volatility)
        
        # ANNUALIZE
        expected_returns_annual = expected_returns_series * config.TRADING_DAYS_PER_YEAR
        expected_volatility_annual = expected_volatility_series * np.sqrt(config.TRADING_DAYS_PER_YEAR)
        
        logger.info("Predictions (annualized):")
        logger.info(f"Returns: {expected_returns_annual.to_dict()}")
        logger.info(f"Volatility: {expected_volatility_annual.to_dict()}")
        
        # VALIDATION: Ensure reasonable values
        if (expected_returns_annual < -0.50).any() or (expected_returns_annual > 2.0).any():
            logger.warning("Extreme return predictions detected, using historical means")
            # Fallback to pure historical
            for ticker in tickers:
                expected_returns_annual[ticker] = self.historical_means.get(f"{ticker}_return", 0.0004) * 252
                expected_volatility_annual[ticker] = self.historical_means.get(f"{ticker}_volatility", 0.015) * np.sqrt(252)
        
        return expected_returns_annual, expected_volatility_annual

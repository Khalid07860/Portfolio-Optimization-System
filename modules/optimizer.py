"""
Portfolio Optimizer: Mean-Variance Optimization with correct math
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from scipy.optimize import minimize
import config

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Mean-Variance Portfolio Optimization"""
    
    def __init__(self, risk_free_rate: float = config.DEFAULT_RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        self.expected_returns = None
        self.cov_matrix = None
        self.tickers = None
        
    def set_parameters(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ):
        """Set optimization parameters with validation"""
        
        # Validate expected returns
        if expected_returns.isna().any():
            logger.warning("NaN in expected returns")
            expected_returns = expected_returns.fillna(expected_returns.mean())
        
        # Clip extreme values
        expected_returns = expected_returns.clip(-0.50, 2.0)
        
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.tickers = list(expected_returns.index)
        
        logger.info(f"Optimizer set for {len(self.tickers)} assets")
        logger.info(f"Expected returns: min={expected_returns.min():.2%}, max={expected_returns.max():.2%}")
        
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio metrics
        
        Formula:
        - Return: R_p = w^T * μ
        - Volatility: σ_p = sqrt(w^T * Σ * w)
        - Sharpe: (R_p - R_f) / σ_p
        
        Returns:
            (return, volatility, sharpe_ratio)
        """
        # Portfolio return
        portfolio_return = np.dot(weights, self.expected_returns.values)
        
        # Portfolio variance
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix.values, weights))
        
        # Portfolio volatility (std dev)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe(self, weights: np.ndarray) -> float:
        """Objective: minimize negative Sharpe"""
        return -self.portfolio_performance(weights)[2]
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Objective: minimize volatility"""
        return self.portfolio_performance(weights)[1]
    
    def optimize_max_sharpe(
        self,
        max_weight: float = config.DEFAULT_MAX_WEIGHT,
        min_weight: float = config.DEFAULT_MIN_WEIGHT,
        allow_short: bool = config.ALLOW_SHORT_SELLING
    ) -> Dict:
        """
        Optimize for Maximum Sharpe Ratio
        
        Constraints:
        - Σw_i = 1 (weights sum to 1)
        - 0 ≤ w_i ≤ max_weight (no short, bounded weights)
        """
        logger.info("Optimizing for Maximum Sharpe Ratio...")
        
        n_assets = len(self.tickers)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum to 1
        ]
        
        # Bounds
        if allow_short:
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        else:
            bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        # Initial guess: equal weights
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize using SLSQP
        result = minimize(
            self.negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            logger.warning(f"Optimization: {result.message}")
        
        # Get optimal weights
        optimal_weights = result.x
        
        # Calculate metrics
        port_return, port_vol, port_sharpe = self.portfolio_performance(optimal_weights)
        
        # VALIDATION: Check if results are reasonable
        if port_return < -0.50 or port_return > 2.0:
            logger.error(f"Invalid return: {port_return:.2%}")
            # Fallback to equal weights
            optimal_weights = np.array([1/n_assets] * n_assets)
            port_return, port_vol, port_sharpe = self.portfolio_performance(optimal_weights)
            logger.info("Using equal weights fallback")
        
        result_dict = {
            'weights': dict(zip(self.tickers, optimal_weights)),
            'expected_return': float(port_return),
            'volatility': float(port_vol),
            'sharpe_ratio': float(port_sharpe),
            'success': result.success
        }
        
        logger.info(f"Max Sharpe: Return={port_return:.2%}, Vol={port_vol:.2%}, Sharpe={port_sharpe:.3f}")
        
        return result_dict
    
    def optimize_min_volatility(
        self,
        max_weight: float = config.DEFAULT_MAX_WEIGHT,
        min_weight: float = config.DEFAULT_MIN_WEIGHT,
        allow_short: bool = config.ALLOW_SHORT_SELLING
    ) -> Dict:
        """Optimize for Minimum Volatility"""
        logger.info("Optimizing for Minimum Volatility...")
        
        n_assets = len(self.tickers)
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        if allow_short:
            bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        else:
            bounds = tuple((0, max_weight) for _ in range(n_assets))
        
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            self.portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x
        port_return, port_vol, port_sharpe = self.portfolio_performance(optimal_weights)
        
        result_dict = {
            'weights': dict(zip(self.tickers, optimal_weights)),
            'expected_return': float(port_return),
            'volatility': float(port_vol),
            'sharpe_ratio': float(port_sharpe),
            'success': result.success
        }
        
        logger.info(f"Min Vol: Return={port_return:.2%}, Vol={port_vol:.2%}, Sharpe={port_sharpe:.3f}")
        
        return result_dict
    
    def generate_efficient_frontier(
        self,
        n_portfolios: int = config.N_PORTFOLIOS,
        max_weight: float = config.DEFAULT_MAX_WEIGHT
    ) -> pd.DataFrame:
        """
        Generate Efficient Frontier using Monte Carlo
        """
        logger.info(f"Generating {n_portfolios} random portfolios...")
        
        n_assets = len(self.tickers)
        results = []
        
        np.random.seed(42)
        
        for _ in range(n_portfolios):
            # Random weights
            weights = np.random.random(n_assets)
            weights = weights / np.sum(weights)
            
            # Check max weight constraint
            if np.any(weights > max_weight):
                continue
            
            # Calculate performance
            ret, vol, sharpe = self.portfolio_performance(weights)
            
            # Only add reasonable portfolios
            if -0.50 < ret < 2.0 and vol < 1.0:
                results.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe
                })
        
        frontier = pd.DataFrame(results)
        logger.info(f"Generated {len(frontier)} valid portfolios")
        
        return frontier
    
    def calculate_capital_allocation(
        self,
        weights: Dict[str, float],
        investment_amount: float,
        latest_prices: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate exact share allocation"""
        logger.info(f"Calculating allocation for ₹{investment_amount:,.0f}")
        
        allocation = []
        
        for ticker, weight in weights.items():
            if weight > 0.001:
                target_amount = investment_amount * weight
                
                if ticker in latest_prices:
                    price = latest_prices[ticker]
                    shares = int(target_amount / price)
                    actual_amount = shares * price
                else:
                    shares = 0
                    actual_amount = 0
                    price = 0
                
                allocation.append({
                    'Ticker': ticker,
                    'Weight': weight,
                    'Target_Amount': target_amount,
                    'Latest_Price': price,
                    'Shares': shares,
                    'Actual_Amount': actual_amount
                })
        
        allocation_df = pd.DataFrame(allocation)
        
        if not allocation_df.empty:
            allocation_df = allocation_df.sort_values('Weight', ascending=False)
            
            total_actual = allocation_df['Actual_Amount'].sum()
            if total_actual > 0:
                allocation_df['Actual_Weight'] = allocation_df['Actual_Amount'] / total_actual
            
            cash = investment_amount - total_actual
            logger.info(f"Cash remaining: ₹{cash:,.0f}")
        
        return allocation_df
    
    def stress_test(
        self,
        weights: Dict[str, float],
        scenario: str = "market_crash"
    ) -> Dict:
        """Stress test portfolio"""
        logger.info(f"Stress testing: {scenario}")
        
        weights_array = np.array([weights[ticker] for ticker in self.tickers])
        
        original_return, original_vol, original_sharpe = self.portfolio_performance(weights_array)
        
        if scenario == "market_crash":
            stressed_returns = self.expected_returns * (1 + config.STRESS_SCENARIOS['market_crash'])
            stressed_return = np.dot(weights_array, stressed_returns.values)
            stressed_vol = original_vol
            
        elif scenario == "volatility_spike":
            stressed_cov = self.cov_matrix * config.STRESS_SCENARIOS['volatility_spike']
            stressed_variance = np.dot(weights_array.T, np.dot(stressed_cov.values, weights_array))
            stressed_vol = np.sqrt(stressed_variance)
            stressed_return = original_return
        else:
            stressed_return = original_return
            stressed_vol = original_vol
        
        stressed_sharpe = (stressed_return - self.risk_free_rate) / stressed_vol
        
        results = {
            'scenario': scenario,
            'original_return': original_return,
            'stressed_return': stressed_return,
            'return_change': stressed_return - original_return,
            'original_volatility': original_vol,
            'stressed_volatility': stressed_vol,
            'volatility_change': stressed_vol - original_vol,
            'original_sharpe': original_sharpe,
            'stressed_sharpe': stressed_sharpe,
            'sharpe_change': stressed_sharpe - original_sharpe
        }
        
        logger.info(f"Stress test: Return change = {results['return_change']:.2%}")
        
        return results

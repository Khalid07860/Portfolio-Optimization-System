"""
FIXED Dashboard - Selects EXACTLY the number requested + ML validation
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_module import DataFetcher
from modules.feature_engineering import FeatureEngineer
from modules.ml_model import MLPredictor
from modules.optimizer import PortfolioOptimizer
import config

st.set_page_config(page_title="AI Portfolio Optimizer", page_icon="📊", layout="wide")

# CSS
st.markdown("""
<style>
    .main { background-color: #f5f7fa; }
    .stMetric { 
        background-color: white; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    }
    h1 { color: #1f2937; text-align: center; padding: 20px; }
</style>
""", unsafe_allow_html=True)

def auto_select_stocks(n_stocks_requested):
    """
    FIXED: Select EXACTLY n_stocks_requested
    Validates more stocks to ensure we get enough
    """
    st.info(f"🤖 Analyzing stocks to select EXACTLY {n_stocks_requested} best performers...")
    
    all_stocks = config.get_all_stocks()
    st.write(f"📊 Total stocks in universe: {len(all_stocks)}")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    fetcher = DataFetcher()
    
    # Validate MORE stocks to ensure we have enough
    # Check 2x the requested amount to account for invalid tickers
    stocks_to_check = min(len(all_stocks), n_stocks_requested * 3)
    
    st.write(f"🔍 Validating {stocks_to_check} stocks...")
    progress_val = st.progress(0)
    status_val = st.empty()
    
    valid_stocks = []
    for idx, ticker in enumerate(all_stocks[:stocks_to_check]):
        status_val.text(f"Validating {ticker}... ({idx+1}/{stocks_to_check})")
        progress_val.progress((idx + 1) / stocks_to_check)
        
        try:
            test_data = fetcher.fetch_data([ticker], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if len(test_data) >= 200:  # Need enough data
                valid_stocks.append(ticker)
                
                # Stop if we have 2x what we need (for scoring)
                if len(valid_stocks) >= n_stocks_requested * 2:
                    break
        except:
            continue
    
    progress_val.empty()
    status_val.empty()
    
    st.success(f"✅ Found {len(valid_stocks)} valid stocks")
    
    if len(valid_stocks) < n_stocks_requested:
        st.error(f"❌ Only {len(valid_stocks)} valid stocks available. Please select fewer stocks.")
        return valid_stocks, []
    
    # Score all valid stocks
    st.write(f"📈 Scoring {len(valid_stocks)} stocks...")
    stock_scores = []
    progress_score = st.progress(0)
    
    for idx, ticker in enumerate(valid_stocks):
        try:
            progress_score.progress((idx + 1) / len(valid_stocks))
            
            prices = fetcher.fetch_data([ticker], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            returns = fetcher.calculate_returns()
            
            # Calculate metrics
            annual_return = returns.mean().iloc[0] * 252
            annual_vol = returns.std().iloc[0] * np.sqrt(252)
            sharpe = (annual_return - 0.06) / annual_vol if annual_vol > 0 else 0
            
            # Get sector
            sector = None
            for s, stocks in config.INDIAN_STOCKS.items():
                if ticker in stocks:
                    sector = s
                    break
            
            stock_scores.append({
                'ticker': ticker,
                'sharpe': sharpe,
                'return': annual_return,
                'volatility': annual_vol,
                'sector': sector
            })
        except:
            continue
    
    progress_score.empty()
    
    # Sort by Sharpe ratio
    stock_scores = sorted(stock_scores, key=lambda x: x['sharpe'], reverse=True)
    
    # Select EXACTLY n_stocks_requested with sector diversification
    selected = []
    sectors_used = {}
    
    # First pass: one stock per sector
    for stock in stock_scores:
        if len(selected) >= n_stocks_requested:
            break
        
        sector = stock['sector']
        if sector and sector not in sectors_used:
            selected.append(stock['ticker'])
            sectors_used[sector] = 1
    
    # Second pass: fill remaining slots with best Sharpe ratios
    for stock in stock_scores:
        if len(selected) >= n_stocks_requested:
            break
        
        if stock['ticker'] not in selected:
            selected.append(stock['ticker'])
    
    # ENSURE we have EXACTLY the requested number
    selected = selected[:n_stocks_requested]
    
    # Get info for selected stocks
    selected_info = [s for s in stock_scores if s['ticker'] in selected]
    
    st.success(f"🎯 Selected EXACTLY {len(selected)} stocks as requested!")
    
    return selected, selected_info

# Session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'ml_validation' not in st.session_state:
    st.session_state.ml_validation = None

# Header
st.title("📊 AI Portfolio Optimizer - Indian Market")
st.markdown(f"**ML-powered optimization | {config.TOTAL_STOCKS}+ NSE stocks available**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    mode = st.radio("Selection Mode", ["🤖 Auto-Select", "✍️ Manual"])
    
    if mode == "🤖 Auto-Select":
        n_stocks = st.slider("Number of Stocks", 3, 15, 10)
        st.info(f"AI will select EXACTLY {n_stocks} stocks from {config.TOTAL_STOCKS}+ available")
    else:
        # Show ALL sectors with stock counts
        st.markdown("**Available by Sector:**")
        for sector, stocks in config.INDIAN_STOCKS.items():
            st.caption(f"{sector}: {len(stocks)} stocks")
        
        sectors = st.multiselect(
            "Select Sectors",
            list(config.INDIAN_STOCKS.keys()),
            default=['Technology', 'Banking', 'FMCG']
        )
        
        available = []
        for sector in sectors:
            available.extend(config.INDIAN_STOCKS[sector])
        
        st.info(f"{len(available)} stocks available from selected sectors")
        
        manual_tickers = st.multiselect(
            "Select Stocks",
            available,
            default=available[:5] if available else []
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=datetime.now() - timedelta(days=730))
    with col2:
        end_date = st.date_input("End", value=datetime.now())
    
    investment = st.number_input("Investment (₹)", 10000, 10000000, 100000, 10000)
    
    with st.expander("Advanced"):
        max_weight = st.slider("Max Weight %", 10, 50, 35) / 100
        n_simulations = st.slider("Simulations", 1000, 10000, 3000)
        show_ml_details = st.checkbox("Show ML Validation Details", value=True)
    
    optimize_btn = st.button("🚀 OPTIMIZE", type="primary", use_container_width=True)

# Main
if optimize_btn:
    with st.spinner("Optimizing..."):
        try:
            # Get tickers
            if mode == "🤖 Auto-Select":
                tickers, stock_info = auto_select_stocks(n_stocks)
                
                if len(tickers) != n_stocks:
                    st.error(f"⚠️ Could only select {len(tickers)} stocks instead of {n_stocks}")
                    if len(tickers) < 2:
                        st.stop()
                
                # Show selected
                st.success(f"✅ Selected: {', '.join([t.replace('.NS', '') for t in tickers])}")
                
                cols = st.columns(min(len(tickers), 6))
                for idx, ticker in enumerate(tickers):
                    with cols[idx % 6]:
                        name = ticker.replace('.NS', '')
                        sector = stock_info[idx]['sector'] if idx < len(stock_info) else 'N/A'
                        sharpe = stock_info[idx]['sharpe'] if idx < len(stock_info) else 0
                        st.metric(name, sector, f"Sharpe: {sharpe:.2f}")
            else:
                tickers = manual_tickers
            
            if len(tickers) < 2:
                st.error("Need at least 2 stocks")
                st.stop()
            
            # Progress
            progress = st.progress(0)
            status = st.empty()
            
            # Initialize
            fetcher = DataFetcher()
            engineer = FeatureEngineer()
            predictor = MLPredictor()
            optimizer = PortfolioOptimizer()
            
            # Step 1
            status.text("📊 Fetching historical data...")
            progress.progress(15)
            
            prices = fetcher.fetch_data(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            returns = fetcher.calculate_returns()
            
            st.info(f"📊 Fetched {len(prices)} days of data for {len(tickers)} stocks")
            
            # Step 2
            status.text("🔧 Engineering features...")
            progress.progress(30)
            
            features = engineer.create_features(prices, returns)
            future_returns, future_vol = engineer.create_target_variables(returns)
            
            # Step 3
            status.text("🤖 Training ML models...")
            progress.progress(50)
            
            training_metrics = predictor.train_models(features, future_returns, future_vol)
            
            # Store ML validation
            ml_validation = {}
            for ticker, metrics in training_metrics.items():
                ml_validation[ticker] = {
                    'return_r2': metrics['return'].get('test_r2', 0),
                    'vol_r2': metrics['volatility'].get('test_r2', 0),
                    'use_ml_return': metrics.get('use_ml_return', False),
                    'use_ml_vol': metrics.get('use_ml_volatility', False)
                }
            
            st.session_state.ml_validation = ml_validation
            
            # Step 4
            status.text("🔮 Predicting returns & volatility...")
            progress.progress(65)
            
            expected_returns, expected_volatility = predictor.predict(features, tickers)
            
            # Step 5
            status.text("⚡ Optimizing portfolio...")
            progress.progress(80)
            
            cov_matrix = returns.cov() * 252
            optimizer.set_parameters(expected_returns, cov_matrix)
            
            max_sharpe = optimizer.optimize_max_sharpe(max_weight=max_weight)
            min_vol = optimizer.optimize_min_volatility(max_weight=max_weight)
            frontier = optimizer.generate_efficient_frontier(n_portfolios=n_simulations)
            
            # Step 6
            status.text("💰 Calculating allocation...")
            progress.progress(95)
            
            latest_prices = fetcher.get_latest_prices(tickers)
            allocation = optimizer.calculate_capital_allocation(
                max_sharpe['weights'],
                investment,
                latest_prices
            )
            
            summary = fetcher.get_summary_statistics(returns)
            
            progress.progress(100)
            status.empty()
            progress.empty()
            
            # Store
            st.session_state.results = {
                'max_sharpe': max_sharpe,
                'min_vol': min_vol,
                'frontier': frontier,
                'allocation': allocation,
                'summary': summary,
                'expected_returns': expected_returns,
                'expected_volatility': expected_volatility,
                'training_metrics': training_metrics,
                'tickers': tickers,
                'investment': investment
            }
            
            st.balloons()
            st.success("✅ Optimization complete!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Results
if st.session_state.results:
    r = st.session_state.results
    
    st.markdown("---")
    st.subheader("📈 Optimization Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    ret = r['max_sharpe']['expected_return'] * 100
    vol = r['max_sharpe']['volatility'] * 100
    sharpe = r['max_sharpe']['sharpe_ratio']
    proj = r['investment'] * (1 + r['max_sharpe']['expected_return'])
    gain = proj - r['investment']
    
    with col1:
        st.metric("Expected Return", f"{ret:.2f}%", help="Annualized expected return")
    with col2:
        st.metric("Volatility (Risk)", f"{vol:.2f}%", help="Annualized standard deviation")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.3f}", help="Risk-adjusted return measure")
    with col4:
        st.metric("Projected Value (1Y)", f"₹{proj:,.0f}", delta=f"+₹{gain:,.0f}")
    
    # ML Validation Display
    if show_ml_details and st.session_state.ml_validation:
        st.markdown("---")
        st.subheader("🤖 ML Model Validation")
        
        ml_data = []
        for ticker, val in st.session_state.ml_validation.items():
            ml_data.append({
                'Stock': ticker.replace('.NS', ''),
                'Return R²': f"{val['return_r2']:.3f}",
                'Vol R²': f"{val['vol_r2']:.3f}",
                'Using ML for Return': '✅' if val['use_ml_return'] else '❌ (Historical)',
                'Using ML for Vol': '✅' if val['use_ml_vol'] else '❌ (Historical)'
            })
        
        ml_df = pd.DataFrame(ml_data)
        st.dataframe(ml_df, hide_index=True, use_container_width=True)
        
        st.info("""
        **ML Validation Explained:**
        - **R² > 0.05**: ML model is used (captures some pattern)
        - **R² < 0.05**: Historical mean is used (ML not reliable)
        - This ensures predictions are always reasonable!
        """)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Allocation", 
        "💰 Investment", 
        "📈 Frontier",
        "📋 Details"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Weights")
            
            weights = {k.replace('.NS', ''): v for k, v in r['max_sharpe']['weights'].items() if v > 0.001}
            
            fig = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.3,
                textinfo='label+percent'
            )])
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Distribution")
            
            df = pd.DataFrame({
                'Stock': list(weights.keys()),
                'Weight %': [v*100 for v in weights.values()]
            }).sort_values('Weight %', ascending=False)
            
            st.dataframe(df, hide_index=True, use_container_width=True)
    
    with tab2:
        st.subheader("Your Investment Plan")
        
        if not r['allocation'].empty:
            alloc = r['allocation'].copy()
            alloc['Ticker'] = alloc['Ticker'].str.replace('.NS', '')
            
            display = alloc[['Ticker', 'Shares', 'Latest_Price', 'Actual_Amount', 'Weight']].copy()
            display.columns = ['Stock', 'Shares', 'Price ₹', 'Amount ₹', 'Weight %']
            display['Weight %'] = (display['Weight %'] * 100).round(2)
            display['Price ₹'] = display['Price ₹'].round(2)
            display['Amount ₹'] = display['Amount ₹'].round(0)
            
            st.dataframe(display, hide_index=True, use_container_width=True)
            
            total = alloc['Actual_Amount'].sum()
            cash = r['investment'] - total
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Invested", f"₹{total:,.0f}")
            with col2:
                st.metric("Cash Remaining", f"₹{cash:,.0f}")
            with col3:
                st.metric("Total Shares", int(alloc['Shares'].sum()))
    
    with tab3:
        st.subheader("Efficient Frontier")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=r['frontier']['volatility']*100,
            y=r['frontier']['return']*100,
            mode='markers',
            marker=dict(
                size=5,
                color=r['frontier']['sharpe_ratio'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe")
            ),
            name='Portfolios',
            hovertemplate='Return: %{y:.2f}%<br>Risk: %{x:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[vol],
            y=[ret],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star', line=dict(width=2, color='white')),
            name='Optimal Portfolio'
        ))
        
        fig.update_layout(
            xaxis_title="Risk (Volatility %)",
            yaxis_title="Expected Return %",
            height=500,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📊 ML Predictions vs Historical")
        
        pred_df = pd.DataFrame({
            'Stock': [t.replace('.NS', '') for t in r['tickers']],
            'ML Expected Return %': [r['expected_returns'][t] * 100 for t in r['tickers']],
            'Historical Return %': [r['summary'].loc[t, 'Annual_Return'] * 100 for t in r['tickers']],
            'ML Expected Vol %': [r['expected_volatility'][t] * 100 for t in r['tickers']],
            'Historical Vol %': [r['summary'].loc[t, 'Annual_Volatility'] * 100 for t in r['tickers']]
        })
        
        st.dataframe(pred_df.round(2), hide_index=True, use_container_width=True)
        
        st.info("""
        **Comparison:** ML predictions vs pure historical averages.  
        ML tries to capture patterns, but falls back to historical if not reliable.
        """)
        
        st.subheader("Summary Statistics")
        summary_display = r['summary'][['Annual_Return', 'Annual_Volatility', 'Sharpe']].copy()
        summary_display.columns = ['Expected Return', 'Volatility', 'Sharpe Ratio']
        summary_display.index = [i.replace('.NS', '') for i in summary_display.index]
        st.dataframe((summary_display * 100).round(2), use_container_width=True)

else:
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"🤖 **AI-Powered**\n\nML selects from {config.TOTAL_STOCKS}+ stocks")
    
    with col2:
        st.info("🇮🇳 **Indian Market**\n\nNSE stocks across 16 sectors")
    
    with col3:
        st.info("📊 **Validated**\n\nML with fallback to historical")
    
    st.markdown("---")
    st.markdown("**👈 Configure your portfolio and click OPTIMIZE**")
    
    # Show available stocks
    with st.expander("📋 View All Available Stocks"):
        for sector, stocks in config.INDIAN_STOCKS.items():
            st.markdown(f"**{sector}** ({len(stocks)} stocks)")
            st.caption(", ".join([s.replace('.NS', '') for s in stocks]))

st.markdown("---")
st.caption("⚠️ Educational purposes only | 🇮🇳 NSE | ML + Mean-Variance Optimization")

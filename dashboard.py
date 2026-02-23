"""
Complete AI Portfolio Optimizer Dashboard
With ML predictions, optimization, and Indian stocks
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

# Page config
st.set_page_config(
    page_title="AI Portfolio Optimizer",
    page_icon="📊",
    layout="wide"
)

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

def get_all_indian_stocks():
    """Get all Indian stocks"""
    all_stocks = []
    for stocks in config.INDIAN_STOCKS.values():
        all_stocks.extend(stocks)
    return all_stocks

def auto_select_stocks(n_stocks):
    """Auto-select best stocks using historical Sharpe ratio"""
    st.info(f"🤖 Analyzing {n_stocks} best Indian stocks...")
    
    all_stocks = get_all_indian_stocks()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    fetcher = DataFetcher()
    
    # Validate first
    st.write("Validating tickers...")
    valid_stocks = fetcher.validate_tickers(all_stocks[:25])  # Check first 25
    
    if len(valid_stocks) < n_stocks:
        st.warning(f"Only {len(valid_stocks)} valid stocks found")
        n_stocks = min(n_stocks, len(valid_stocks))
    
    # Score stocks
    stock_scores = []
    progress = st.progress(0)
    
    for idx, ticker in enumerate(valid_stocks):
        try:
            progress.progress((idx + 1) / len(valid_stocks))
            
            prices = fetcher.fetch_data([ticker], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            returns = fetcher.calculate_returns()
            
            # Calculate annualized metrics
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
                'sector': sector
            })
        except:
            continue
    
    progress.empty()
    
    # Sort by Sharpe
    stock_scores = sorted(stock_scores, key=lambda x: x['sharpe'], reverse=True)
    
    # Select diversified
    selected = []
    sectors_used = set()
    
    for stock in stock_scores:
        if len(selected) >= n_stocks:
            break
        if stock['sector'] not in sectors_used or len(selected) < 3:
            selected.append(stock['ticker'])
            if stock['sector']:
                sectors_used.add(stock['sector'])
    
    return selected, stock_scores[:len(selected)]

# Session state
if 'results' not in st.session_state:
    st.session_state.results = None

# Header
st.title("📊 AI Portfolio Optimizer - Indian Market")
st.markdown("**ML-powered portfolio optimization for NSE stocks**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    mode = st.radio(
        "Selection Mode",
        ["🤖 Auto-Select", "✍️ Manual"]
    )
    
    if mode == "🤖 Auto-Select":
        n_stocks = st.slider("Number of Stocks", 3, 10, 6)
        st.info(f"AI will select top {n_stocks} stocks")
    else:
        sectors = st.multiselect(
            "Sectors",
            list(config.INDIAN_STOCKS.keys()),
            default=['Technology', 'Banking']
        )
        
        available = []
        for sector in sectors:
            available.extend(config.INDIAN_STOCKS[sector])
        
        manual_tickers = st.multiselect(
            "Stocks",
            available,
            default=available[:4] if available else []
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start",
            value=datetime.now() - timedelta(days=730)
        )
    with col2:
        end_date = st.date_input(
            "End",
            value=datetime.now()
        )
    
    investment = st.number_input(
        "Investment (₹)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    with st.expander("Advanced"):
        max_weight = st.slider("Max Weight %", 10, 50, 35) / 100
        n_simulations = st.slider("Simulations", 1000, 10000, 3000)
    
    optimize_btn = st.button("🚀 OPTIMIZE", type="primary", use_container_width=True)

# Main
if optimize_btn:
    with st.spinner("Optimizing..."):
        try:
            # Get tickers
            if mode == "🤖 Auto-Select":
                tickers, stock_info = auto_select_stocks(n_stocks)
                
                st.success(f"✅ Selected: {', '.join([t.replace('.NS', '') for t in tickers])}")
                
                # Show selected stocks
                cols = st.columns(len(tickers))
                for idx, ticker in enumerate(tickers):
                    with cols[idx]:
                        name = ticker.replace('.NS', '')
                        sector = stock_info[idx]['sector'] if idx < len(stock_info) else 'N/A'
                        st.metric(name, sector)
            else:
                tickers = manual_tickers
            
            if len(tickers) < 2:
                st.error("Select at least 2 stocks")
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
            status.text("📊 Fetching data...")
            progress.progress(15)
            
            prices = fetcher.fetch_data(
                tickers,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            returns = fetcher.calculate_returns()
            
            # Step 2
            status.text("🔧 Engineering features...")
            progress.progress(30)
            
            features = engineer.create_features(prices, returns)
            future_returns, future_vol = engineer.create_target_variables(returns)
            
            # Step 3
            status.text("🤖 Training ML models...")
            progress.progress(50)
            
            training_metrics = predictor.train_models(features, future_returns, future_vol)
            
            # Step 4
            status.text("🔮 Making predictions...")
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
    st.subheader("📈 Results")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    ret = r['max_sharpe']['expected_return'] * 100
    vol = r['max_sharpe']['volatility'] * 100
    sharpe = r['max_sharpe']['sharpe_ratio']
    proj = r['investment'] * (1 + r['max_sharpe']['expected_return'])
    gain = proj - r['investment']
    
    with col1:
        st.metric("Expected Return", f"{ret:.2f}%")
    with col2:
        st.metric("Volatility", f"{vol:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
    with col4:
        st.metric("Projected (1Y)", f"₹{proj:,.0f}", delta=f"+₹{gain:,.0f}")
    
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
                hole=0.3
            )])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Distribution")
            
            df = pd.DataFrame({
                'Stock': list(weights.keys()),
                'Weight %': [v*100 for v in weights.values()]
            }).sort_values('Weight %', ascending=False)
            
            st.dataframe(df, hide_index=True)
    
    with tab2:
        st.subheader("Investment Plan")
        
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
                st.metric("Invested", f"₹{total:,.0f}")
            with col2:
                st.metric("Cash", f"₹{cash:,.0f}")
            with col3:
                st.metric("Shares", int(alloc['Shares'].sum()))
    
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
                showscale=True
            ),
            name='Portfolios'
        ))
        
        fig.add_trace(go.Scatter(
            x=[vol],
            y=[ret],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Optimal'
        ))
        
        fig.update_layout(
            xaxis_title="Risk %",
            yaxis_title="Return %",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("ML Predictions")
        
        pred_df = pd.DataFrame({
            'Stock': [t.replace('.NS', '') for t in r['tickers']],
            'Expected Return %': r['expected_returns'].values * 100,
            'Expected Vol %': r['expected_returns'].index.map(
                lambda x: r['expected_returns'][x]
            ).values * 100 if 'expected_volatility' in r else [0] * len(r['tickers'])
        })
        
        st.dataframe(pred_df, hide_index=True)
        
        st.subheader("Summary Statistics")
        st.dataframe(r['summary'].round(4))

else:
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🤖 **AI-Powered**\n\nML selects best stocks")
    
    with col2:
        st.info("🇮🇳 **Indian Market**\n\n40+ NSE stocks")
    
    with col3:
        st.info("📊 **Optimized**\n\nMax Sharpe & Min Vol")
    
    st.markdown("---")
    st.markdown("**👈 Configure and click OPTIMIZE**")

st.markdown("---")
st.caption("⚠️ Educational purposes | 🇮🇳 NSE | Made with ML & Mean-Variance Optimization")

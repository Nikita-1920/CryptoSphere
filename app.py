import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

from data.database import init_db, create_user, verify_user, get_user_portfolio, save_user_portfolio, get_chat_history, save_chat_message, clear_chat_history, reset_password
from components.ui_elements import render_metric_card, plot_candlestick, plot_volatility, plot_sentiment, plot_correlation_heatmap, plot_gauge_chart, plot_normalized_comparison, plot_portfolio_allocation, plot_equity_curve, plot_monte_carlo, plot_fear_greed_gauge
from models.backtester import run_backtest
from models.risk_manager import calculate_portfolio_risk, run_monte_carlo
from models.agent_tools import add_to_portfolio_logic, get_portfolio_summary_logic
from data.market_data import fetch_crypto_data
from data.sentiment import fetch_news_sentiment, fetch_raw_news, fetch_reddit_sentiment
from models.predictor import prepare_data, train_and_predict, predict_future_7_days

# Initialize SQLite Database
init_db()

st.set_page_config(page_title="Crypto AI Analytics", page_icon="🌌", layout="wide")

# Theme Toggle
st.sidebar.markdown("<div style='font-size: 20px; font-weight: bold; padding-bottom: 5px;'>Theme</div>", unsafe_allow_html=True)
theme = st.sidebar.radio("Theme", ["Dark", "Light"], horizontal=True, label_visibility="collapsed")

# Load CSS
if theme == "Dark":
    with open("assets/style.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
else:
    with open("assets/style_light.css") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_id = None

user_container = st.sidebar.container(border=False)
user_container.markdown("<span id='user-account-container-marker'></span>", unsafe_allow_html=True)
user_container.markdown("## 👤 User Account")
if not st.session_state.logged_in:
    tab_login, tab_reg = user_container.tabs(["Login", "Register"])
    with tab_login:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        login_btn = st.button("Login", type="primary", use_container_width=True)
        
        def toggle_reset():
            st.session_state.show_reset = not st.session_state.get('show_reset', False)
        st.button("Forgot Password", on_click=toggle_reset, type="primary", use_container_width=True)
            
        if login_btn:
            uid, msg = verify_user(login_user, login_pass)
            if uid:
                st.session_state.logged_in = True
                st.session_state.username = login_user
                st.session_state.user_id = uid
                st.session_state.portfolio = get_user_portfolio(uid)
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
                
        if st.session_state.get('show_reset'):
            st.markdown("---")
            res_user = st.text_input("Username for Reset", key="res_user")
            res_pass = st.text_input("New Password", type="password", key="res_pass")
            if st.button("Reset Password", type="primary", use_container_width=True):
                if not res_user or not res_pass:
                    st.error("Username and new password are required.")
                else:
                    success, msg = reset_password(res_user, res_pass)
                    if success:
                        st.success(msg)
                        st.session_state.show_reset = False
                        st.rerun()
                    else:
                        st.error(msg)
                    
    with tab_reg:
        reg_user = st.text_input("New Username", key="reg_user")
        reg_pass = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register", type="primary", use_container_width=True):
            if not reg_user or not reg_pass:
                st.error("Username and password are required.")
            else:
                success, msg = create_user(reg_user, reg_pass)
                if success:
                    st.success(msg + " You can now login.")
                else:
                    st.error(msg)
else:
    user_container.success(f"Welcome, {st.session_state.username}!")
    if user_container.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_id = None
        if 'portfolio' in st.session_state:
            del st.session_state.portfolio
        st.rerun()

st.sidebar.markdown("---")

# Sidebar Configuration
st.sidebar.markdown("## ⚙️ Configuration")

coins = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "Ripple": "XRP-USD"
}
selected_coin_name = st.sidebar.selectbox("Select Cryptocurrency", list(coins.keys()))
selected_coin = coins[selected_coin_name]

# Currency Selector
currency = st.sidebar.selectbox("Currency", ["USD", "INR"])
currency_sym = "₹" if currency == "INR" else "$"

# Date / Interval
st.sidebar.markdown("### Timeframe")
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", datetime.today() - timedelta(days=365))
end_date = col2.date_input("End Date", datetime.today())
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "1wk"], index=0)

# Technical Indicators
st.sidebar.markdown("### Technical Indicators")
show_bb = st.sidebar.checkbox("Bollinger Bands")
show_macd = st.sidebar.checkbox("MACD")
show_rsi = st.sidebar.checkbox("RSI")

# Price Alerts
st.sidebar.markdown("### Price Alerts")
price_target = st.sidebar.number_input("Set Target Price", min_value=0.0, value=0.0, step=100.0)

# Main Content
st.markdown(f"<h1 style='text-align: center;'>🌌 {selected_coin_name} Analytics Platform</h1>", unsafe_allow_html=True)

# Live Ticker Marquee
@st.fragment(run_every="60s")
def render_live_ticker(coins_dict, selected_currency, selected_currency_sym):
    ticker_html = "<div style='overflow: hidden; white-space: nowrap; margin-bottom: 20px;'><marquee scrollamount='5' style='font-size: 1.2rem; font-weight: bold; color: #10b981;'>"
    for c_name, c_sym in coins_dict.items():
        try:
            temp_df = fetch_crypto_data(c_sym, period='1d', interval='1d', currency=selected_currency)
            if temp_df is not None and not temp_df.empty:
                c_price = temp_df['Close'].iloc[-1]
                ticker_html += f"&nbsp;&nbsp;&nbsp;&nbsp;{c_name}: {selected_currency_sym}{c_price:,.2f}"
        except:
            pass
    ticker_html += "</marquee></div>"
    st.markdown(ticker_html, unsafe_allow_html=True)

render_live_ticker(coins, currency, currency_sym)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Dashboard", "Multi-Coin Comparison", "Portfolio Tracker", "Chatbot Assistant", "News & Alerts", "Strategy Backtesting"])

with tab1:
    # Fetch Data
    with st.spinner(f"Fetching data for {selected_coin_name}..."):
        market_df = fetch_crypto_data(selected_coin, start_date=start_date, end_date=end_date, interval=interval, currency=currency)
        sentiment_df = fetch_news_sentiment(query=selected_coin_name, days=30)

    if market_df is not None and not market_df.empty:
        latest = market_df.iloc[-1]
        prev = market_df.iloc[-2]
        
        price_change = latest['Close'] - prev['Close']
        price_pct = (price_change / prev['Close']) * 100
        is_positive = price_change >= 0
        
        vol = latest['Volatility']
        
        sentiment_val = sentiment_df.iloc[-1]['Sentiment'] if not sentiment_df.empty else 0.0
        
        def format_volume(vol):
            if vol >= 1e9:
                return f"{vol/1e9:.2f}B"
            elif vol >= 1e6:
                return f"{vol/1e6:.2f}M"
            elif vol >= 1e3:
                return f"{vol/1e3:.2f}K"
            return f"{vol:.2f}"
            
        # Metric Cards
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            render_metric_card("Latest Price", f"{currency_sym}{latest['Close']:,.2f}", f"{price_pct:.2f}%", is_positive, "💰")
        with c2:
            render_metric_card("24h Volume", f"{currency_sym}{format_volume(latest['Volume'])}", "", True, "📊")
        with c3:
            render_metric_card("Volatility", f"{vol*100:.2f}%", "", False, "📈")
        with c4:
            s_color = sentiment_val >= 0
            render_metric_card("Sentiment Score", f"{sentiment_val:.2f}", "", s_color, "🧠")
        with c5:
            trend = "UP" if latest['Trend'] == 1 else "DOWN"
            render_metric_card("Current Trend", trend, "", latest['Trend'] == 1, "⚡")

        # Main Charts
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        
        # Check price alert
        if price_target > 0:
            if (prev['Close'] < price_target <= latest['Close']) or (prev['Close'] > price_target >= latest['Close']):
                st.toast(f"🚨 ALERT: {selected_coin_name} crossed your target of {currency_sym}{price_target:,.2f}!")
                
        st.plotly_chart(plot_candlestick(market_df, selected_coin_name, theme, show_bb=show_bb, show_macd=show_macd, show_rsi=show_rsi, target_price=price_target), use_container_width=True)

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.plotly_chart(plot_volatility(market_df, theme), use_container_width=True)
        with col_chart2:
            st.plotly_chart(plot_sentiment(sentiment_df, theme), use_container_width=True)

        # Advanced ML Pipeline
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        st.markdown("<h2>🤖 AI Prediction Panel</h2>", unsafe_allow_html=True)
        with st.spinner("Training ML Pipeline (Random Forest, XGBoost, Logistic Regression)..."):
            X, y, X_predict, features, final_df = prepare_data(market_df, sentiment_df)
            if len(X) > 30:
                results, best_name, importance_df = train_and_predict(X, y, X_predict, features)
                best_model_data = results[best_name]
                
                # Gauge Chart for Sentiment / Signals
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.plotly_chart(plot_gauge_chart(sentiment_val, "Sentiment Signal", theme), use_container_width=True)
                    
                with col_g2:
                    pred_class = "pred-up" if best_model_data['Prediction'] == "Up" else "pred-down"
                    html_content = f"""
                    <div class='glass-container' style='padding: 20px; box-sizing: border-box; min-height: 280px; display: flex; flex-direction: column; justify-content: center;'>
                        <h3 style='text-align:center;'>Next Day Prediction ({best_name})</h3>
                        <div class='{pred_class}' style='text-align:center; font-size: 2.5rem; font-weight: bold; margin: 15px 0;'>{best_model_data['Prediction']}</div>
                        <p style='text-align:center; word-wrap: break-word;'>Confidence: <b>{best_model_data['Confidence']*100:.1f}%</b> | Accuracy: {best_model_data['Accuracy']*100:.1f}%</p>
                    </div>
                    """
                    st.markdown(html_content, unsafe_allow_html=True)

                # 7-Day Forecast
                st.markdown("<h3>📅 7-Day Future Prediction</h3>", unsafe_allow_html=True)
                forecast_df = predict_future_7_days(final_df, best_model_data['Model'], features)
                
                def color_direction(val):
                    if val == 'Up':
                        return 'color: #10b981; font-weight: bold;'
                    elif val == 'Down':
                        return 'color: #ef4444; font-weight: bold;'
                    return ''
                    
                # Use map if available (newer pandas), else applymap (older pandas)
                if hasattr(forecast_df.style, 'map'):
                    styled_forecast = forecast_df.style.map(color_direction, subset=['Direction'])
                else:
                    styled_forecast = forecast_df.style.applymap(color_direction, subset=['Direction'])
                    
                st.dataframe(styled_forecast, use_container_width=True)
                st.download_button("📥 Export Predictions (CSV)", data=forecast_df.to_csv().encode('utf-8'), file_name=f'{selected_coin_name}_predictions.csv', mime='text/csv')


                
            else:
                st.warning("Not enough data to train models.")
    else:
        st.error("Error fetching data.")

with tab2:
    st.markdown("<h2>Multi-Coin Comparison</h2>", unsafe_allow_html=True)
    st.info("Compare performance of top cryptocurrencies over the selected timeframe.")
    
    # Let user select coins to compare
    all_coins_list = list(coins.keys())
    selected_compare_coins = st.multiselect(
        "Select Coins to Compare",
        options=all_coins_list,
        default=["Bitcoin", "Ethereum", "Solana"]
    )
    
    if selected_compare_coins:
        compare_df_dict = {}
        metrics_data = []
        
        with st.spinner("Fetching comparison data..."):
            for coin_name in selected_compare_coins:
                c_symbol = coins[coin_name]
                df = fetch_crypto_data(c_symbol, start_date=start_date, end_date=end_date, interval=interval, currency=currency)
                if df is not None and not df.empty:
                    compare_df_dict[coin_name] = df
                    
                    # Calculate metrics
                    first_close = df['Close'].iloc[0]
                    last_close = df['Close'].iloc[-1]
                    ret = ((last_close - first_close) / first_close) * 100
                    volatility = df['Volatility'].mean() if 'Volatility' in df.columns else 0
                    avg_volume = df['Volume'].mean()
                    
                    metrics_data.append({
                        'Asset': coin_name,
                        'Current Price': f"{currency_sym}{last_close:,.2f}",
                        'Return (%)': ret,
                        'Avg Volatility': f"{volatility*100:.2f}%",
                        'Avg Daily Volume': f"{currency_sym}{avg_volume/1e6:.2f}M"
                    })
        
        if compare_df_dict:
            # Plotly Line Chart
            st.plotly_chart(plot_normalized_comparison(compare_df_dict, theme), use_container_width=True)
            
            # Metrics Table
            st.markdown("### Comparison Metrics")
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(
                metrics_df.style.background_gradient(cmap='RdYlGn', subset=['Return (%)']).format({'Return (%)': "{:.2f}%"}),
                use_container_width=True,
                hide_index=True
            )
            st.markdown(
                """
                <style>
                /* Target download buttons to increase size and make text black in light theme */
                div[data-testid="stDownloadButton"] button p,
                div[data-testid="stDownloadButton"] button:hover p,
                div[data-testid="stDownloadButton"] button:focus p,
                div[data-testid="stDownloadButton"] button:active p {
                    color: #000000 !important;
                    font-size: 1.3rem !important;
                    font-weight: 600 !important;
                }
                div[data-testid="stDownloadButton"] button,
                div[data-testid="stDownloadButton"] button:hover,
                div[data-testid="stDownloadButton"] button:focus,
                div[data-testid="stDownloadButton"] button:active {
                    border-color: rgba(0, 0, 0, 0.3) !important;
                    height: auto !important;
                    padding-top: 0.5rem !important;
                    padding-bottom: 0.5rem !important;
                    background-color: transparent !important;
                }
                </style>
                """ if theme == "Light" else """
                <style>
                /* Target download buttons to increase size and make text white in dark theme */
                div[data-testid="stDownloadButton"] button p,
                div[data-testid="stDownloadButton"] button:hover p,
                div[data-testid="stDownloadButton"] button:focus p,
                div[data-testid="stDownloadButton"] button:active p {
                    color: #ffffff !important;
                    font-size: 1.3rem !important;
                    font-weight: 600 !important;
                }
                div[data-testid="stDownloadButton"] button,
                div[data-testid="stDownloadButton"] button:hover,
                div[data-testid="stDownloadButton"] button:focus,
                div[data-testid="stDownloadButton"] button:active {
                    border-color: rgba(255, 255, 255, 0.3) !important;
                    height: auto !important;
                    padding-top: 0.5rem !important;
                    padding-bottom: 0.5rem !important;
                    background-color: transparent !important;
                }
                </style>
                """, unsafe_allow_html=True
            )
            col1, col2, col3 = st.columns([1, 1.5, 1])
            with col2:
                st.download_button("📥 Export Metrics (CSV)", data=metrics_df.to_csv(index=False).encode('utf-8'), file_name='comparison_metrics.csv', mime='text/csv', use_container_width=True)
    else:
        st.warning("Please select at least one coin to compare.")

with tab3:
    st.markdown("<h2>Portfolio Tracker</h2>", unsafe_allow_html=True)
    st.info("Manage your portfolio holdings and see live valuation.")
    
    # Initialize session state for portfolio if not exists
    if 'portfolio' not in st.session_state:
        if st.session_state.get('logged_in'):
            st.session_state.portfolio = get_user_portfolio(st.session_state.user_id)
        else:
            st.session_state.portfolio = pd.DataFrame({
                'Asset': ['Bitcoin', 'Ethereum', 'USDT'],
                'Amount': [0.5, 4.2, 1000.0]
            })
            
    if not st.session_state.get('logged_in'):
        st.info("💡 You are viewing a demo portfolio. Create an account in the sidebar to save your own holdings permanently!")
    
    st.markdown("<h3 style='text-align: center;'>Your Holdings</h3>", unsafe_allow_html=True)
    
    # Interactive Editor
    edited_portfolio = st.data_editor(
        st.session_state.portfolio,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Asset": st.column_config.SelectboxColumn(
                "Asset",
                help="Select an asset",
                options=list(coins.keys()) + ["USDT", "USDC"],
                required=True,
            ),
            "Amount": st.column_config.NumberColumn(
                "Amount",
                help="Enter holding amount",
                min_value=0.0,
                format="%.4f",
                required=True,
            )
        }
    )
    
    # Update session state and DB if changed
    if not edited_portfolio.equals(st.session_state.portfolio):
        st.session_state.portfolio = edited_portfolio
        if st.session_state.get('logged_in'):
            save_user_portfolio(st.session_state.user_id, edited_portfolio)
            st.toast("✅ Portfolio saved to database!")

    # Calculate real-time value
    with st.spinner("Calculating live value & risk metrics..."):
        port_eval = []
        total_value = 0.0
        historical_data_dict = {}
        
        for _, row in edited_portfolio.iterrows():
            asset = row['Asset']
            amt = float(row['Amount']) if pd.notnull(row['Amount']) else 0.0
            price = 1.0 # Default for stablecoins like USDT/USDC in USD
            
            if amt > 0:
                # Fetch historical price if trackable
                if asset in coins:
                    # Fetch 365 days for robust risk calculation
                    df = fetch_crypto_data(coins[asset], start_date=datetime.today() - timedelta(days=365), end_date=datetime.today(), interval="1d", currency=currency)
                    if df is not None and not df.empty:
                        price = df['Close'].iloc[-1]
                        historical_data_dict[asset] = df
                elif currency == "INR":
                    # Simple mock conversion for stablecoins if INR selected
                    price = 83.0 # approx USD to INR
                    
                val = amt * price
                total_value += val
                port_eval.append({
                    'Asset': asset,
                    'Amount': amt,
                    'Price': price,
                    'Value': val
                })
            
        eval_df = pd.DataFrame(port_eval) if port_eval else pd.DataFrame(columns=['Asset', 'Amount', 'Price', 'Value'])
        
    # Display Total Value
    st.markdown(f"<div style='text-align: center; font-size: 2.5rem; font-weight: bold; color: {'#10b981' if theme == 'Dark' else '#059669'}; margin-top: 20px; margin-bottom: 30px;'>Total Value: {currency_sym}{total_value:,.2f}</div>", unsafe_allow_html=True)
    
    if not eval_df.empty and total_value > 0:
        st.markdown("---")
        
        # Portfolio Analysis and Allocation (Side by side)
        col_pa1, col_pa2 = st.columns([1, 1])
        with col_pa1:
            st.markdown("### 📊 Portfolio Analysis")
            # Calculate Risk Metrics
            annual_vol, sharpe_ratio, var_95 = calculate_portfolio_risk(eval_df, historical_data_dict, total_value)
            
            r1, r2 = st.columns(2)
            with r1:
                render_metric_card("1-Day VaR (95%)", f"-{currency_sym}{abs(var_95):,.2f}", "", False, "🛡️")
                render_metric_card("Sharpe Ratio", f"{sharpe_ratio:.2f}", "", sharpe_ratio > 1.0, "⚖️")
            with r2:
                render_metric_card("Annual Volatility", f"{annual_vol*100:.2f}%", "", False, "📈")
                
        with col_pa2:
            st.plotly_chart(plot_portfolio_allocation(eval_df, theme), use_container_width=True)
            
        st.markdown("---")
        
        # Monte Carlo and Explanation (Side by side)
        col_mc1, col_mc2 = st.columns([1.5, 1])
        with col_mc1:
            st.markdown("### 🎲 30-Day Monte Carlo Projection")
            st.info("Simulating 1,000 future paths based on historical portfolio volatility.")
            mc_df = run_monte_carlo(eval_df, historical_data_dict, total_value, days=30, simulations=1000)
            if not mc_df.empty:
                st.plotly_chart(plot_monte_carlo(mc_df, theme), use_container_width=True)
                
        with col_mc2:
            st.markdown("### 🤔 What does this chart mean?")
            st.markdown("""
            **Think of this as a 'multiverse' for your money!** 🌌
            
            Because crypto is unpredictable, we can't draw one single line to show where your portfolio is going. Instead, we simulated **1,000 different alternate futures** for the next 30 days based on how wild your coins usually swing (Volatility).
            
            * **Expected Value (Blue Line):** This is the dead-center average of all our simulations. If things go exactly as usual, your portfolio ends up here.
            * **95th Percentile (Green Area):** The "Optimistic" scenario. Only 5% of our simulations did better than this!
            * **5th Percentile (Red Area):** The "Pessimistic" scenario. Your portfolio only dropped below this line in our worst 5% of simulations. This is your likely worst-case scenario.
            
            **TL;DR:** The wider the cone, the riskier your portfolio is!
            """)
            
        st.markdown("---")
        
        # Export Button Bottom Center
        col_ex1, col_ex2, col_ex3 = st.columns([1, 1, 1])
        with col_ex2:
            st.download_button("📥 Export Portfolio (CSV)", data=eval_df.to_csv(index=False).encode('utf-8'), file_name='portfolio.csv', mime='text/csv', use_container_width=True)

with tab4:
    st.markdown("<h2>💬 AI Chatbot Assistant (Gemini Pro)</h2>", unsafe_allow_html=True)
    
    # Define agent tools
    def add_to_portfolio(asset: str, amount: float) -> str:
        """
        Adds a specific amount of a cryptocurrency asset to the user's portfolio.
        If the asset already exists, it increases the amount.
        
        Args:
            asset: The name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Solana').
            amount: The quantity of the asset to add.
        """
        res = add_to_portfolio_logic(st.session_state.get('user_id'), asset, amount)
        if st.session_state.get('logged_in'):
            st.session_state.portfolio = get_user_portfolio(st.session_state.user_id)
        return res
        
    def get_portfolio_summary() -> str:
        """
        Retrieves a summary of the user's current cryptocurrency holdings.
        """
        return get_portfolio_summary_logic(st.session_state.get('user_id'))
        
    # API Key Input
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
        st.markdown("<small>[Get your API Key from Google AI Studio](https://aistudio.google.com/app/apikey)</small><br><small><i>Tip: Save your key in the `.env` file to hide this input and load it automatically!</i></small>", unsafe_allow_html=True)
    
    if api_key:
        genai.configure(api_key=api_key)
        
        # Dynamically find an available model for this API key
        available_model = 'gemini-1.5-flash'
        try:
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            if models:
                # Prefer flash models, otherwise pick the first available
                flash_models = [m for m in models if 'flash' in m]
                available_model = flash_models[0].replace('models/', '') if flash_models else models[0].replace('models/', '')
        except Exception:
            pass # Fallback to default if list_models fails
            
        model = genai.GenerativeModel(
            model_name=available_model,
            tools=[add_to_portfolio, get_portfolio_summary]
        )
        
        # Initialize chat history
        if "messages" not in st.session_state:
            if st.session_state.get('logged_in'):
                st.session_state.messages = get_chat_history(st.session_state.user_id)
            else:
                st.session_state.messages = []
            
        # Chat input at the TOP
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([9, 1])
            with col1:
                prompt = st.text_input("Ask about the crypto market...", label_visibility="collapsed", placeholder="Ask about the crypto market...")
            with col2:
                submit_button = st.form_submit_button("Send 🚀", type="primary", use_container_width=True)
                
        if submit_button and prompt:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            if st.session_state.get('logged_in'):
                save_chat_message(st.session_state.user_id, "user", prompt)
            
            # Prepare context
            context = f"System Context: You are a helpful, professional Crypto Analytics AI. The user is currently viewing data for {selected_coin_name}."
            if 'market_df' in locals() and market_df is not None and not market_df.empty:
                current_price = market_df.iloc[-1]['Close']
                context += f"\n- **Current Price:** {currency_sym}{current_price:,.2f}"
                if 'Volatility' in market_df.columns:
                    context += f"\n- **Volatility:** {market_df.iloc[-1]['Volatility']*100:.2f}%"
                if 'Trend' in market_df.columns:
                    context += f"\n- **Trend:** {'UP' if market_df.iloc[-1]['Trend'] == 1 else 'DOWN'}"
            
            if 'sentiment_val' in locals():
                context += f"\n- **News Sentiment Score:** {sentiment_val:.2f} (-1 to 1 scale)"
                
            if 'best_model_data' in locals():
                context += f"\n- **ML Prediction (Tomorrow):** {best_model_data['Prediction']} (Confidence: {best_model_data['Confidence']*100:.1f}%) using {best_name}"
                
            if 'portfolio' in st.session_state:
                port_df = st.session_state.portfolio
                if not port_df.empty:
                    port_str = ", ".join([f"{float(row['Amount'])} {row['Asset']}" for _, row in port_df.iterrows() if pd.notnull(row['Amount']) and float(row['Amount']) > 0])
                    if port_str:
                        context += f"\n- **User Portfolio Holdings:** {port_str}"
                        
            if 'social_sentiment_score' in st.session_state:
                fg_val = st.session_state.social_sentiment_score
                context += f"\n- **Social Fear & Greed Index (Reddit):** {fg_val:.1f}/100 (0=Extreme Fear, 100=Extreme Greed)"
            
            context += "\n\nPlease provide concise, analytical, and professional answers based on this context. Do not format as a list unless requested."
            
            # Call Gemini
            with st.spinner("Thinking..."):
                try:
                    # Convert history for Gemini
                    gemini_history = []
                    for msg in st.session_state.messages[:-1]:
                        role = "user" if msg["role"] == "user" else "model"
                        gemini_history.append({"role": role, "parts": [msg["content"]]})
                        
                    chat = model.start_chat(history=gemini_history, enable_automatic_function_calling=True)
                    response = chat.send_message(context + "\n\nUser question: " + prompt)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                    if st.session_state.get('logged_in'):
                        save_chat_message(st.session_state.user_id, "assistant", response.text)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error communicating with Gemini API: {e}")

        # Display chat history in a scrollable container at the BOTTOM
        chat_container = st.container(height=400, border=True)
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
        col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
        with col_c2:
            if st.button("🗑️ Clear Chat History", type="primary", use_container_width=True):
                if st.session_state.get('logged_in'):
                    clear_chat_history(st.session_state.user_id)
                st.session_state.messages = []
                st.rerun()
    else:
        st.info("Please enter your Gemini API Key above to start chatting.")

with tab5:
    st.markdown("<h2>📰 News, Sentiment & Alerts Hub</h2>", unsafe_allow_html=True)
    st.info(f"Manage price alerts, view live sentiment, and get AI summaries for {selected_coin_name}.")
    
    # --- Alerts Hub ---
    st.markdown("### 🚨 Active Price Alerts")
    if 'active_alerts' not in st.session_state:
        st.session_state.active_alerts = []
        
    with st.expander("➕ Create New Alert", expanded=False):
        c_alert1, c_alert2, c_alert3 = st.columns([2, 2, 1])
        with c_alert1:
            alert_coin = st.selectbox("Coin", list(coins.keys()), index=list(coins.keys()).index(selected_coin_name))
        with c_alert2:
            alert_price = st.number_input("Target Price", min_value=0.0, step=10.0)
        with c_alert3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Add Alert", type="primary", use_container_width=True):
                if alert_price > 0:
                    st.session_state.active_alerts.append({"coin": alert_coin, "target": alert_price})
                    st.success(f"Alert added for {alert_coin} at {currency_sym}{alert_price:,.2f}")
                
    if st.session_state.active_alerts:
        for idx, alt in enumerate(st.session_state.active_alerts):
            st.markdown(f"**{alt['coin']}** - Target: {currency_sym}{alt['target']:,.2f}")
    else:
        st.caption("No active alerts.")
        
    st.markdown("---")
    
    # --- Sentiment & News ---
    st.markdown("### 📰 Latest Headlines")
    with st.spinner("Fetching latest news..."):
        raw_news_df = fetch_raw_news(query=selected_coin_name, days=7)
        
    if not raw_news_df.empty:
        if 'api_key' in locals() and api_key and 'model' in locals():
            if st.button("🤖 Generate AI News Summary", type="primary"):
                with st.spinner("Gemini is reading the news..."):
                    headlines = "\n".join(raw_news_df['Title'].head(15).tolist())
                    summary_prompt = f"Summarize the market sentiment for {selected_coin_name} based on these recent headlines:\n{headlines}\nProvide 3 concise bullet points."
                    try:
                        summary_response = model.generate_content(summary_prompt)
                        st.success("**AI Market Summary:**")
                        st.markdown(summary_response.text)
                    except Exception as e:
                        st.error(f"Failed to generate summary: {e}")
        else:
            st.caption("Enter your Gemini API key in the Chatbot tab to enable AI summaries.")
        
        # Sentiment Filter
        filter_val = st.radio("Filter News By Sentiment:", ["All", "Bullish", "Bearish", "Neutral"], horizontal=True)
        
        filtered_df = raw_news_df
        if filter_val == "Bullish":
            filtered_df = raw_news_df[raw_news_df['Sentiment_Score'] > 0.05]
        elif filter_val == "Bearish":
            filtered_df = raw_news_df[raw_news_df['Sentiment_Score'] < -0.05]
        elif filter_val == "Neutral":
            filtered_df = raw_news_df[(raw_news_df['Sentiment_Score'] >= -0.05) & (raw_news_df['Sentiment_Score'] <= 0.05)]
            
        if filtered_df.empty:
            st.warning(f"No {filter_val.lower()} news found.")
        else:
            for _, row in filtered_df.iterrows():
                sentiment_score = row['Sentiment_Score']
                if sentiment_score > 0.05:
                    color = "#10b981" # Bullish Green
                    label = "🟢 Bullish"
                elif sentiment_score < -0.05:
                    color = "#ef4444" # Bearish Red
                    label = "🔴 Bearish"
                else:
                    color = "#94a3b8" # Neutral
                    label = "⚪ Neutral"
                    
                html = f"""
                <div class="glass-container" style="padding: 15px; margin-bottom: 10px; border-left: 4px solid {color}; border-radius: 8px;">
                    <a href="{row['Link']}" target="_blank" style="text-decoration: none; color: inherit; font-size: 18px; font-weight: bold;">{row['Title']}</a>
                    <br>
                    <small style="color: #64748b;">{row['Date']} | Sentiment: <b style="color: {color}">{label}</b> (Score: {sentiment_score:.2f})</small>
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)
    else:
        st.warning("No recent news found.")

with tab6:
    st.markdown("<h2>📈 Strategy Backtesting Engine</h2>", unsafe_allow_html=True)
    st.info(f"Simulate trading strategies on historical data for {selected_coin_name}.")
    
    col_bt1, col_bt2 = st.columns([1, 3])
    
    with col_bt1:
        st.markdown("### Backtest Settings")
        strategy = st.selectbox("Select Strategy", ["Buy & Hold", "MACD Crossover", "RSI Mean Reversion"])
        initial_capital = st.number_input("Initial Capital ($)", min_value=100.0, value=10000.0, step=1000.0)
        transaction_fee = st.number_input("Transaction Fee (%)", min_value=0.0, max_value=5.0, value=0.1, step=0.05) / 100.0
        run_btn = st.button("▶ Run Backtest", use_container_width=True)
        
    with col_bt2:
        if run_btn:
            with st.spinner(f"Running {strategy} on {selected_coin_name}..."):
                try:
                    bt_df = fetch_crypto_data(selected_coin, start_date=start_date, end_date=end_date, interval=interval, currency=currency)
                except:
                    bt_df = None
                    
                if bt_df is not None and not bt_df.empty:
                    equity_curve, metrics = run_backtest(bt_df, strategy, initial_capital, transaction_fee)
                    
                    if equity_curve is not None and not equity_curve.empty:
                        # Display Metrics
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            render_metric_card("Total Return", f"{metrics['Total Return']*100:.2f}%", "", metrics['Total Return'] >= 0, "💰")
                        with m2:
                            render_metric_card("Win Rate", f"{metrics['Win Rate (Days)']*100:.1f}%", "", metrics['Win Rate (Days)'] > 0.5, "🎯")
                        with m3:
                            render_metric_card("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%", "", False, "📉")
                        with m4:
                            render_metric_card("Total Trades", f"{metrics['Total Trades']}", "", True, "🔄")
                            
                        # Display Chart
                        st.plotly_chart(plot_equity_curve(equity_curve, theme), use_container_width=True)
                    else:
                        st.error("Failed to generate backtest results.")
                else:
                    st.error("Not enough data to run backtest.")
        else:
            st.info("Configure your settings and click 'Run Backtest' to see results.")

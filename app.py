import numpy as np
import pandas as pd
import ccxt
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta
import time
import pytz
import json
import os

# Streamlit cache to optimize API calls
# @st.cache_data(ttl=60)
def init_exchange():
    exchange = ccxt.binanceusdm({
        'apiKey': os.environ['APIKEY'],  # Replace with your Binance API key
        'secret': os.environ['SECRET'],  # Replace with your Binance secret
        'enableRateLimit': True,
    })
    return exchange

# @st.cache_data(ttl=60)
def fetch_ccxt_prices(_exchange, symbols):
    try:
        _exchange.load_markets()
        prices = {}
        for symbol in symbols:
            ticker = _exchange.fetch_ticker(symbol)
            prices[symbol] = ticker['last'] if ticker else None
        return prices
    except Exception as e:
        st.error(f"Error fetching prices: {e}")
        return {symbol: None for symbol in symbols}

# @st.cache_data(ttl=60)
def fetch_open_positions(_exchange, symbols):
    try:
        positions = _exchange.fetch_positions(symbols)
        position_data = []
        for pos in positions:
            if float(pos['contracts']) > 0: # Only include non-zero positions
                position_data.append({
                    'Symbol': pos['symbol'],
                    'Side': pos['side'],
                    'Contracts': float(pos['contracts']),
                    'EntryPrice': float(pos['entryPrice']),
                    'CurrentPrice': float(pos['markPrice'] or pos['entryPrice']),
                    'UnrealizedPnl': float(pos['unrealizedPnl'] or 0),
                    'Timestamp': pd.to_datetime(datetime.now(pytz.timezone('Asia/Hong_Kong')))
                })
        return pd.DataFrame(position_data)
    except Exception as e:
        st.error(f"Error fetching positions: {e}")
        return pd.DataFrame()

# @st.cache_data(ttl=300)
def fetch_trades_all(_exchange, symbols, since_date):
    try:
        _exchange.load_markets()
        trade_data = []
        since_date_object = datetime.strptime(since_date, "%Y-%m-%d")
        since_timestamp = int(since_date_object.timestamp()) * 1000
        hkt = pytz.timezone('Asia/Hong_Kong')
        current_time = datetime.now(hkt)
        until_timestamp = int(current_time.timestamp()) * 1000
        current_start = since_date_object
        while current_start.timestamp() * 1000 < until_timestamp:
            next_end = current_start + timedelta(days=7)
            current_start_ms = int(current_start.timestamp()) * 1000
            next_end_ms = int(next_end.timestamp()) * 1000
            if next_end_ms > until_timestamp:
                next_end_ms = until_timestamp
            for symbol in symbols:
                trades = _exchange.fetch_my_trades(symbol=symbol, since=current_start_ms, params={'until': next_end_ms})
                for trade in trades:
                    realized_pnl = float(trade.get('info', {}).get('realizedPnl', '0') or 0)
                    trade_data.append({
                        'Timestamp': pd.to_datetime(trade['timestamp'], unit='ms', utc=True),
                        'Symbol': trade['symbol'],
                        'Side': 'Long' if trade['side'] == 'buy' else 'Short',
                        'Price': float(trade['price']),
                        'Amount': float(trade['amount']),
                        'Cost': float(trade['cost']),
                        'Fee': float(trade.get('fee', {}).get('cost', 0)),
                        'Realized P&L': realized_pnl
                    })
                time.sleep(_exchange.rateLimit / 1000)
            current_start = next_end
        df = pd.DataFrame(trade_data)
        if not df.empty:
            df = df.sort_values(by='Timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error fetching trades: {e}")
        return pd.DataFrame()

def calculate_metrics(df_trades, df_positions):
    # Realized P&L from trades
    df_trades = df_trades[df_trades['Realized P&L'] != 0]
    total_trades = len(df_trades)
    realized_pnl = df_trades['Realized P&L'].sum() if not df_trades.empty else 0
    win_trades = len(df_trades[df_trades['Realized P&L'] > 0])
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    avg_pnl = df_trades['Realized P&L'].mean() if total_trades > 0 else 0
    returns = df_trades['Realized P&L']
    sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() != 0 else 0
    cumulative_pnl = df_trades['Realized P&L'].cumsum() if not df_trades.empty else pd.Series()
   
    # Unrealized P&L from open positions
    unrealized_pnl = df_positions['UnrealizedPnl'].sum() if not df_positions.empty else 0
   
    # Total P&L (realized + unrealized)
    total_pnl = realized_pnl + unrealized_pnl
   
    # Drawdown calculation (based on realized P&L only, as unrealized is snapshot-based)
    peak = cumulative_pnl.cummax()
    drawdown = peak - cumulative_pnl
    max_drawdown = drawdown.max() if not drawdown.empty else 0
   
    return {
        'Total Trades': total_trades,
        'Total P&L': total_pnl,
        'Win Rate': win_rate,
        'Average P&L': avg_pnl,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Realized P&L': realized_pnl,
        'Unrealized P&L': unrealized_pnl,
        'Cumulative P&L': cumulative_pnl
    }

def calculate_leaderboard(df_trades, df_positions):
    # Calculate realized P&L per symbol
    realized_pnl = df_trades.groupby('Symbol')['Realized P&L'].sum().reset_index() if not df_trades.empty else pd.DataFrame({'Symbol': [], 'Realized P&L': []})
    # Calculate unrealized P&L per symbol
    unrealized_pnl = df_positions.groupby('Symbol')['UnrealizedPnl'].sum().reset_index() if not df_positions.empty else pd.DataFrame({'Symbol': [], 'UnrealizedPnl': []})
    # Merge realized and unrealized P&L
    leaderboard = pd.merge(realized_pnl, unrealized_pnl, on='Symbol', how='outer').fillna(0)
    # Calculate total P&L
    leaderboard['Total P&L'] = leaderboard['Realized P&L'] + leaderboard['UnrealizedPnl']
    # Sort by total P&L in descending order
    leaderboard = leaderboard.sort_values(by='Total P&L', ascending=False).reset_index(drop=True)
    return leaderboard

def main():
    # Set page config
    st.set_page_config(page_title="Binance Trading Dashboard", layout="wide", initial_sidebar_state="auto")
    # Custom CSS with Tailwind for light and dark mode support
    st.markdown("""
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <style>
            body { font-family: 'Inter', sans-serif; }
            .stApp {
                background-color: #f9fafb;
            }
            @media (prefers-color-scheme: dark) {
                .stApp {
                    background-color: #1f2937;
                }
            }
            .sidebar .sidebar-content {
                background-color: #ffffff;
                border-right: 1px solid #e5e7eb;
            }
            @media (prefers-color-scheme: dark) {
                .sidebar .sidebar-content {
                    background-color: #374151;
                    border-right: 1px solid #4b5563;
                }
            }
            h1, h2, h3, p {
                color: #1f2937;
            }
            @media (prefers-color-scheme: dark) {
                h1, h2, h3, p {
                    color: #f3f4f6;
                }
            }
            .stMetric {
                background-color: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 16px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                color: #1f2937;
            }
            @media (prefers-color-scheme: dark) {
                .stMetric {
                    background-color: #374151;
                    border: 1px solid #4b5563;
                    color: #f3f4f6;
                }
            }
            .stButton>button {
                background-color: #2563eb;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
            }
            .stButton>button:hover {
                background-color: #1d4ed8;
            }
            @media (prefers-color-scheme: dark) {
                .stButton>button {
                    background-color: #3b82f6;
                }
                .stButton>button:hover {
                    background-color: #2563eb;
                }
            }
            .stDataFrame {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                color: #1f2937;
            }
            @media (prefers-color-scheme: dark) {
                .stDataFrame {
                    border: 1px solid #4b5563;
                    color: #f3f4f6;
                }
            }
            .plotly-chart {
                background-color: #ffffff;
                border-radius: 8px;
                padding: 16px;
            }
            @media (prefers-color-scheme: dark) {
                .plotly-chart {
                    background-color: #374151;
                }
            }
            /* Mobile responsiveness */
            @media (max-width: 640px) {
                h1 { font-size: 1.5rem; }
                h2 { font-size: 1.25rem; }
                h3 { font-size: 1rem; }
                .stMetric { padding: 8px; }
                .stButton>button { padding: 6px 12px; }
                .stDataFrame { font-size: 0.875rem; }
            }
        </style>
    """, unsafe_allow_html=True)
    # Sidebar
    with st.sidebar:
        config_path = os.path.join(os.getcwd(), 'instruments.json')
        with open(config_path, 'r') as f:
            data = json.load(f)
        #all_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT', 'HBARUSDT', 'RENDERUSDT', 'SUIUSDT', 'ARBUSDT', '1000PEPEUSDT', 'XRPUSDT', 'ADAUSDT', '1000SHIBUSDT', 'OPUSDT']
        all_symbols = [f"{instr['symbol'].split('/')[0]}USDT" for instr in data['instruments']]
        st.markdown('<h2 class="text-lg font-semibold mb-4">⚙️ Settings</h2>', unsafe_allow_html=True)
        symbols = st.multiselect(
            "Select Symbols",
            options=all_symbols,
            default=[],
            help="Choose one or more trading pairs to analyze."
        )
        since_date = st.date_input(
            "Start Date",
            value=datetime(2025, 8, 1),
            min_value=datetime(2020, 1, 1),
            max_value=datetime.now(),
            help="Select the start date for trade history."
        )
        since_date_str = since_date.strftime("%Y-%m-%d")
        trade_side = st.selectbox("Filter by Trade Side", ["All", "Long", "Short"], help="Filter trades by Long or Short position.")
        refresh = st.button("🔄 Refresh Data", help="Fetch the latest data from Binance.")
    # If no symbols selected, use default
    if not symbols:
        symbols = all_symbols
    # Title and header
    st.markdown('<h1 class="text-3xl font-bold mb-4">📈 Binance USD-M Futures Dashboard</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="mb-6">Real-time trading insights for Binance USD-M Futures. Updated as of {datetime.now(pytz.timezone("Asia/Hong_Kong")).strftime("%I:%M %p HKT, %b %d, %Y")}.</p>', unsafe_allow_html=True)
    # Initialize exchange
    exchange = init_exchange()
    # Fetch and display balance and prices
    st.markdown('<h2 class="text-xl font-semibold mb-4">Overview</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    try:
        balance = exchange.fetch_balance()
        total_balance = balance['total'].get('USDT', 0)
        col1.metric("Account Balance (USDT)", f"${total_balance:,.2f}", delta=None, help="Total USDT balance in your Binance account.")
    except Exception as e:
        col1.error(f"Error fetching balance: {e}")
        if st.button("Retry Balance", key="retry_balance"):
            st.experimental_rerun()
    prices = fetch_ccxt_prices(exchange, symbols)
    for idx, symbol in enumerate(symbols[:3]): # Limit to 3 symbols for layout
        price = prices.get(symbol, None)
        symbol_name = symbol.replace('USDT', '')
        if idx == 0:
            col2.metric(f"{symbol_name} Price (USD)", f"${price:,.2f}" if price else "N/A", delta=None, help=f"Current price of {symbol_name}/USDT.")
        elif idx == 1:
            col3.metric(f"{symbol_name} Price (USD)", f"${price:,.2f}" if price else "N/A", delta=None, help=f"Current price of {symbol_name}/USDT.")
        elif idx == 2:
            col4.metric(f"{symbol_name} Price (USD)", f"${price:,.2f}" if price else "N/A", delta=None, help=f"Current price of {symbol_name}/USDT.")
    # Fetch trades and positions
    # if refresh:
        # st.cache_data.clear()
    df_trades = fetch_trades_all(exchange, symbols=symbols, since_date=since_date_str)
    df_positions = fetch_open_positions(exchange, symbols=symbols)
    # Filter trades by side
    if trade_side != "All":
        df_trades = df_trades[df_trades['Side'] == trade_side]
    # Display metrics and charts in tabs
    if not df_trades.empty or not df_positions.empty:
        metrics = calculate_metrics(df_trades, df_positions)
        st.markdown('<h2 class="text-xl font-semibold mb-4">Key Metrics</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
        col1.metric("Total Trades", metrics['Total Trades'], help="Total number of executed trades.")
        col2.metric("Total P&L (USDT)", f"${metrics['Total P&L']:,.2f}", delta=f"{metrics['Total P&L']:+.2f}", help="Realized + Unrealized P&L.")
        col3.metric("Win Rate", f"{metrics['Win Rate']:.2f}%", help="Percentage of profitable trades.")
        col4.metric("Average P&L (USDT)", f"${metrics['Average P&L']:,.2f}", help="Average profit or loss per trade (realized only).")
        col5.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}", help="Risk-adjusted return metric (realized only).")
        col6.metric("Max Drawdown (USDT)", f"${metrics['Max Drawdown']:,.2f}", help="Maximum loss from peak to trough (realized only).")
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Trade History", "Performance Charts", "Trade Analytics", "Open Positions"])
       
        with tab1:
            st.markdown('<h3 class="text-lg font-semibold mb-2">📜 Trade History</h3>', unsafe_allow_html=True)
            st.dataframe(
                df_trades.sort_values(by='Timestamp', ascending=False).style.format({
                    'Price': '{:,.2f}',
                    'Amount': '{:,.4f}',
                    'Cost': '{:,.2f}',
                    'Fee': '{:,.4f}',
                    'Realized P&L': '{:,.2f}'
                }).set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#2563eb'), ('color', 'white'), ('font-weight', 'bold')]},
                    {'selector': 'td', 'props': [('border', '1px solid #e5e7eb')]},
                    {'selector': 'th, td', 'props': [('color', '#1f2937')]},
                    {'selector': '@media (prefers-color-scheme: dark) th, td', 'props': [('color', '#f3f4f6')]}
                ]),
                use_container_width=True
            )
            with tab2:
                st.markdown('<h3 class="text-lg font-semibold mb-2">📈 Performance Charts</h3>', unsafe_allow_html=True)
                # Cumulative Realized P&L (Daily at 12:00 AM HKT)
                df_plot = df_trades.copy()
                if not df_plot.empty:
                    # Convert timestamps to HKT
                    hkt = pytz.timezone('Asia/Hong_Kong')
                    df_plot['Timestamp'] = df_plot['Timestamp'].dt.tz_convert(hkt)
                    # Floor to 12:00 AM of the next day
                    df_plot['Date'] = df_plot['Timestamp'].apply(lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
                    # Aggregate Realized P&L by date
                    df_daily = df_plot.groupby('Date')['Realized P&L'].sum().reset_index()
                    df_daily['Cumulative Realized P&L'] = df_daily['Realized P&L'].cumsum()
                else:
                    df_daily = pd.DataFrame(columns=['Date', 'Cumulative Realized P&L'])
            
                fig = px.line(
                    df_daily, x='Date', y='Cumulative Realized P&L',
                    title='Cumulative Realized Profit & Loss (Daily at 12:00 AM HKT)',
                    template='plotly_white',
                    color_discrete_map={'Cumulative Realized P&L': '#2563eb'}
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="P&L (USDT)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#1f2937'},
                    title={'font': {'color': '#1f2937'}},
                    hovermode='x unified',
                    legend_title_text='P&L Type'
                )
                fig.update_layout(
                    template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white',
                    font={'color': '#f3f4f6'} if st.get_option('theme.base') == 'dark' else {'color': '#1f2937'},
                    title={'font': {'color': '#f3f4f6'}} if st.get_option('theme.base') == 'dark' else {'font': {'color': '#1f2937'}}
                )
                st.plotly_chart(fig, use_container_width=True)
                # P&L Distribution (Realized only)
                fig = px.histogram(
                    df_trades, x='Realized P&L', nbins=30,
                    title='Realized P&L Distribution',
                    template='plotly_white',
                    color_discrete_sequence=['#ef4444']
                )
                fig.update_layout(
                    xaxis_title="P&L (USDT)",
                    yaxis_title="Count",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#1f2937'},
                    title={'font': {'color': '#1f2937'}},
                    hovermode='x unified'
                )
                fig.update_layout(
                    template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white',
                    font={'color': '#f3f4f6'} if st.get_option('theme.base') == 'dark' else {'color': '#1f2937'},
                    title={'font': {'color': '#f3f4f6'}} if st.get_option('theme.base') == 'dark' else {'font': {'color': '#1f2937'}}
                )
                st.plotly_chart(fig, use_container_width=True)
                # Crypto Leaderboard
                st.markdown('<h3 class="text-lg font-semibold mb-2">🏆 Crypto Leaderboard</h3>', unsafe_allow_html=True)
                leaderboard = calculate_leaderboard(df_trades, df_positions)
                if not leaderboard.empty:
                    st.dataframe(
                        leaderboard.style.format({
                            'Realized P&L': '{:,.2f}',
                            'UnrealizedPnl': '{:,.2f}',
                            'Total P&L': '{:,.2f}'
                        }).set_table_styles([
                            {'selector': 'th', 'props': [('background-color', '#2563eb'), ('color', 'white'), ('font-weight', 'bold')]},
                            {'selector': 'td', 'props': [('border', '1px solid #e5e7eb')]},
                            {'selector': 'th, td', 'props': [('color', '#1f2937')]},
                            {'selector': '@media (prefers-color-scheme: dark) th, td', 'props': [('color', '#f3f4f6')]}
                        ]),
                        use_container_width=True
                    )
                else:
                    st.warning("No P&L data available for the selected symbols.")
        with tab3:
            st.markdown('<h3 class="text-lg font-semibold mb-2">📊 Trade Analytics</h3>', unsafe_allow_html=True)
            # Trade Volume by Symbol (USDT-based)
            volume_by_symbol = df_trades.groupby('Symbol')['Cost'].sum().reset_index()
            fig = px.pie(
                volume_by_symbol, values='Cost', names='Symbol',
                title='Trade Volume by Symbol (USDT)',
                template='plotly_white',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_traces(
                textinfo='percent+label',
                hoverinfo='label+percent+value',
                texttemplate='%{value:,.2f} USDT',
                textfont={'color': '#ffffff' if st.get_option('theme.base') == 'dark' else '#1f2937'}
            )
            fig.update_layout(
                font={'color': '#1f2937'},
                title={'font': {'color': '#1f2937'}}
            )
            fig.update_layout(
                template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white',
                font={'color': '#f3f4f6'} if st.get_option('theme.base') == 'dark' else {'color': '#1f2937'},
                title={'font': {'color': '#f3f4f6'}} if st.get_option('theme.base') == 'dark' else {'font': {'color': '#1f2937'}}
            )
            st.plotly_chart(fig, use_container_width=True)
            # Trades by Position Type (Long/Short)
            side_counts = df_trades['Side'].value_counts().reset_index()
            side_counts.columns = ['Side', 'Count']
            fig = px.bar(
                side_counts, x='Side', y='Count',
                title='Trades by Position Type (Long/Short)',
                template='plotly_white',
                color_discrete_sequence=['#10b981']
            )
            fig.update_layout(
                xaxis_title="Position Type",
                yaxis_title="Number of Trades",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#1f2937'},
                title={'font': {'color': '#1f2937'}},
                hovermode='x unified'
            )
            fig.update_layout(
                template='plotly_dark' if st.get_option('theme.base') == 'dark' else 'plotly_white',
                font={'color': '#f3f4f6'} if st.get_option('theme.base') == 'dark' else {'color': '#1f2937'},
                title={'font': {'color': '#f3f4f6'}} if st.get_option('theme.base') == 'dark' else {'font': {'color': '#1f2937'}}
            )
            st.plotly_chart(fig, use_container_width=True)
        with tab4:
            st.markdown('<h3 class="text-lg font-semibold mb-2">📊 Open Positions</h3>', unsafe_allow_html=True)
            if not df_positions.empty:
                st.dataframe(
                    df_positions.sort_values(by='Timestamp', ascending=False).style.format({
                        'Contracts': '{:,.4f}',
                        'EntryPrice': '{:,.2f}',
                        'CurrentPrice': '{:,.2f}',
                        'UnrealizedPnl': '{:,.2f}'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#2563eb'), ('color', 'white'), ('font-weight', 'bold')]},
                        {'selector': 'td', 'props': [('border', '1px solid #e5e7eb')]},
                        {'selector': 'th, td', 'props': [('color', '#1f2937')]},
                        {'selector': '@media (prefers-color-scheme: dark) th, td', 'props': [('color', '#f3f4f6')]}
                    ]),
                    use_container_width=True
                )
            else:
                st.warning("No open positions for selected symbols.")
    else:
        st.warning("No trade or position data available for the selected symbols or date range. Try adjusting your filters or refreshing the data.")

if __name__ == "__main__":
    main()
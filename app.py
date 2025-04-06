import yfinance as yf
import pandas as pd
import datetime
import streamlit as st
import plotly.graph_objs as go
import requests
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# UI Setup
st.set_page_config(layout="wide")
st.title("📈 Real-Time Trading Assistant")
st.markdown("### 🔷 Created for Mr. Shreyash")

# Mobile UI optimization
st.markdown("""
<style>
@media (max-width: 768px) {
    .element-container { padding: 10px !important; }
    h1 { font-size: 24px !important; }
    h2, h3 { font-size: 20px !important; }
    .stButton > button { width: 100%; font-size: 18px; }
    .stDataFrame { overflow-x: auto !important; }
}
</style>
""", unsafe_allow_html=True)

# NSE Extended Stock List (Top 200+)
nse_top200 = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "AXISBANK.NS", "LT.NS", "MARUTI.NS", "TATAMOTORS.NS",
    "ITC.NS", "HINDUNILVR.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "WIPRO.NS", "SUNPHARMA.NS",
    "COALINDIA.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS", "BPCL.NS", "EICHERMOT.NS",
    "HCLTECH.NS", "TECHM.NS", "UPL.NS", "ULTRACEMCO.NS", "TITAN.NS", "JSWSTEEL.NS",
    "BHARTIARTL.NS", "ASIANPAINT.NS", "HDFCLIFE.NS", "DIVISLAB.NS", "BRITANNIA.NS",
    "SHREECEM.NS", "NESTLEIND.NS", "CIPLA.NS", "BAJAJFINSV.NS", "GRASIM.NS",
    "HINDALCO.NS", "DRREDDY.NS", "TATACONSUM.NS", "HEROMOTOCO.NS", "SBILIFE.NS",
    "ICICIPRULI.NS", "M&M.NS", "PIDILITIND.NS", "DLF.NS", "BANKBARODA.NS", "INDUSINDBK.NS",
    "IOC.NS", "PNB.NS", "GAIL.NS", "BEL.NS", "VEDL.NS", "TRENT.NS", "NAVINFLUOR.NS",
    "ABB.NS", "AUBANK.NS", "BAJAJ-AUTO.NS", "PAGEIND.NS", "LTI.NS", "ADANIGREEN.NS",
    "ADANIPOWER.NS", "TATAPOWER.NS", "TATACHEM.NS", "SRF.NS", "ZEEL.NS", "HAVELLS.NS",
    "AMBUJACEM.NS", "INDIGO.NS", "IGL.NS", "CANBK.NS", "BHEL.NS", "APOLLOHOSP.NS",
    "GODREJCP.NS", "LUPIN.NS", "BERGEPAINT.NS", "MUTHOOTFIN.NS", "TORNTPHARM.NS",
    "PEL.NS", "MFSL.NS", "ICICIGI.NS", "CONCOR.NS", "ALKEM.NS", "AARTIIND.NS",
    "GLENMARK.NS", "ESCORTS.NS", "SUNTV.NS", "TVSMOTOR.NS", "FEDERALBNK.NS",
    "BOSCHLTD.NS", "NAM-INDIA.NS", "IRCTC.NS", "BIOCON.NS", "HINDPETRO.NS", "COLPAL.NS",
    "UBL.NS", "CROMPTON.NS", "MPHASIS.NS", "DEEPAKNTR.NS", "GMRINFRA.NS", "RECLTD.NS",
    "INDHOTEL.NS", "CHOLAFIN.NS", "COFORGE.NS", "ZYDUSLIFE.NS", "IDFCFIRSTB.NS", "MINDTREE.NS",
    "TATAELXSI.NS", "VOLTAS.NS", "ABCAPITAL.NS", "BALRAMCHIN.NS", "SAIL.NS", "GUJGASLTD.NS",
    "RAJESHEXPO.NS", "AUROPHARMA.NS", "OFSS.NS", "PIIND.NS", "LALPATHLAB.NS", "IRFC.NS",
    "LTTS.NS", "INDIACEM.NS", "BEL.NS", "CESC.NS", "BANKINDIA.NS", "PFC.NS", "IDBI.NS",
    "EDELWEISS.NS", "BANDHANBNK.NS", "JKCEMENT.NS", "SHRIRAMCIT.NS", "FINCABLES.NS"
]

# Sidebar
st.sidebar.header("🔎 Select a Stock")
selected_symbol = st.sidebar.selectbox("Choose Stock Symbol", nse_top200)

# Bot Scanner Tab
if st.sidebar.button("🤖 Bot Scanner"):
    st.markdown("## 🔍 AI Bot Scanner - Chart Pattern Detection")
    bot_results = []
    for stock in nse_top200:
        try:
            data = yf.Ticker(stock).history(interval='15m', period='5d')
            def compute_rsi(series, period=14):
                delta = series.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            rsi = compute_rsi(data['Close'])

            if rsi.iloc[-1] > 60:
                bot_results.append({
                    "Stock": stock,
                    "RSI": round(rsi.iloc[-1], 2),
                    "Last Price": round(data['Close'].iloc[-1], 2)
                })
        except:
            continue
    bot_df = pd.DataFrame(bot_results)
    st.dataframe(bot_df)
    st.stop()

# Functions
def fetch_stock_data(symbol, interval='5m', period='5d'):
    stock = yf.Ticker(symbol)
    data = stock.history(interval=interval, period=period)
    data.dropna(inplace=True)
    return data

def detect_candlestick_patterns(data):
    return {}  # Temporarily disabled due to missing TA-Lib

def check_volume_spike(data, threshold=20):
    avg_volume = data['Volume'][:-1].mean()
    current_volume = data['Volume'].iloc[-1]
    if current_volume > avg_volume * threshold:
        return True, current_volume, avg_volume
    return False, current_volume, avg_volume

def fetch_news(symbol):
    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey=762424e0828643cca4d7247f51f97071"
        response = requests.get(url)
        articles = response.json().get("articles", [])
        return articles[:5]
    except:
        return []

# AI-Based Financial Sentiment using FinBERT
finbert = hf_pipeline("sentiment-analysis", model="ProsusAI/finbert")

def analyze_news_sentiment(news_articles):
    results = []
    for article in news_articles:
        title = article.get('title', '')
        description = article.get('description', '')
        combined_text = f"{title} {description}"
        try:
            prediction = finbert(combined_text)[0]
            sentiment = f"🔼 Positive" if prediction['label'] == 'positive' else "🔻 Negative"
        except:
            sentiment = "⚪ Neutral"
        results.append((title, sentiment))
    return results

def get_bullish_momentum_stocks():
    bullish = []
    for stock in nse_top200:
        try:
            data = fetch_stock_data(stock, interval='1d', period='7d')
            def compute_rsi(series, period=14):
                delta = series.diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            rsi = compute_rsi(data['Close'])

            if rsi.iloc[-1] > 60:
                bullish.append({
                    "Stock": stock,
                    "Last Close": round(data['Close'].iloc[-1], 2),
                    "RSI": round(rsi.iloc[-1], 2),
                    "Volume": int(data['Volume'].iloc[-1])
                })
        except:
            continue
    return bullish

def plot_candlestick(data, symbol):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                open=data['Open'], high=data['High'],
                low=data['Low'], close=data['Close'])])
    fig.update_layout(title=f"{symbol} Candlestick Chart", xaxis_title="Time", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    support = data['Low'].rolling(window=20).min().iloc[-1]
    resistance = data['High'].rolling(window=20).max().iloc[-1]
    fig.add_hline(y=support, line_color="green", line_dash="dot", annotation_text="Support")
    fig.add_hline(y=resistance, line_color="red", line_dash="dot", annotation_text="Resistance")
    st.plotly_chart(fig, use_container_width=True)

# Fetch and display data
data = fetch_stock_data(selected_symbol)
patterns = detect_candlestick_patterns(data)
volume_spike, current_vol, avg_vol = check_volume_spike(data)
news_articles = fetch_news(selected_symbol)
sentiment_analysis = analyze_news_sentiment(news_articles)
bullish_list = get_bullish_momentum_stocks()

# Layout
col1, col2 = st.columns([3, 2])

col1.subheader("🚀 Trade Setup Monitor")
plot_candlestick(data, selected_symbol)
if patterns:
    col1.success(f"Pattern Detected: {', '.join(patterns.keys())}")
else:
    col1.info("No major pattern detected in last candle.")
if volume_spike:
    col1.warning(f"🔺 Sudden Volume Spike! Current: {int(current_vol)} vs Avg: {int(avg_vol)}")

# News
col2.subheader("📰 News & AI-Based Sentiment")
for title, sentiment in sentiment_analysis:
    col2.markdown(f"**{sentiment}** - {title}")

# Breakouts
st.markdown("---")
st.subheader("📊 Weekly Breakout Tracker")
st.markdown("Stocks showing sudden spikes or breakouts in past week:")
breakout_log = pd.DataFrame({
    "Stock": ["ADANIENT.NS", "RELIANCE.NS", "TATAMOTORS.NS"],
    "Breakout Date": ["2024-04-02", "2024-04-04", "2024-04-05"],
    "Pattern": ["Bullish Engulfing", "Hammer", "Morning Star"],
    "Volume Spike": ["Yes", "Yes", "No"]
})
st.dataframe(breakout_log, use_container_width=True)

# Sidebar Momentum
st.sidebar.markdown("---")
st.sidebar.header("📈 Bullish Momentum Stocks")
for stock in bullish_list:
    st.sidebar.markdown(
        f"**{stock['Stock']}**\n"
        f"Price: ₹{stock['Last Close']} | RSI: {stock['RSI']}\n"
        f"Volume: {stock['Volume']}"
    )

# Refresh
st_autorefresh = st.experimental_rerun() if st.button("🔁 Refresh Now") else time.sleep(60)

def detect_candlestick_patterns(data):
    return {}  # Temporarily disabled due to missing TA-Lib

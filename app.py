import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import math
import plotly.graph_objects as go
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Master AI Trader", page_icon="🧠", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🎛️ Steuerzentrale")
market_mode = st.sidebar.selectbox("Markt / Börse", ["Deutschland (Xetra)", "USA (Original)", "Weltweit"])
ticker_input = st.sidebar.text_input("Symbol", value="LIN").upper()
strategy_mode = st.sidebar.selectbox("Strategie", ["Swing Trading (Kurz)", "Value Investing (Langzeit)"])
konto = st.sidebar.number_input("Dein Kontostand (€)", value=10000.0, step=500.0)
risk_pct = st.sidebar.slider("Risiko pro Trade (%)", 0.5, 5.0, 2.0)

# --- HELPER FUNKTIONEN ---

def get_exchange_rate(from_currency):
    if from_currency == "EUR": return 1.0
    try:
        if from_currency == "USD":
            tick = yf.Ticker("EUR=X")
            return tick.history(period="1d")['Close'].iloc[-1]
        return 1.0 
    except: return 1.0 

@st.cache_resource
def load_ai_model():
    try: return pipeline("text-classification", model="ProsusAI/finbert")
    except: return None

# --- HIER WAR DER FEHLER (JETZT KORRIGIERT) ---
def get_symbol_and_currency(user_input, mode):
    symbol = user_input.strip()
    currency = "USD" # Standard-Annahme
    
    if mode == "Deutschland (Xetra)":
        if not symbol.endswith(".DE"): symbol += ".DE"
        currency = "EUR"
    elif mode == "USA (Original)":
        symbol = symbol.replace(".DE", "")
        currency = "USD"
        
    # Wir geben jetzt BEIDES zurück: Symbol UND Währung
    return symbol, currency 

def fetch_data(symbol, period):
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return None

        # Indikatoren
        df['RSI'] = df.ta.rsi(length=14)
        df['ATR'] = df.ta.atr(length=14)
        df['SMA50'] = df.ta.sma(length=50)
        df['SMA200'] = df.ta.sma(length=200)
        
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_SIGNAL'] = macd.iloc[:, 2]

        bands = df.ta.bbands(length=20, std=2)
        upper = [c for c in bands.columns if c.startswith("BBU")][0]
        lower = [c for c in bands.columns if c.startswith("BBL")][0]
        df['BB_UPPER'] = bands[upper]
        df['BB_LOWER'] = bands[lower]
        
        return df
    except: return None

def get_fundamentals(symbol):
    t = yf.Ticker(symbol)
    try:
        i = t.info
        sector = i.get('sector', i.get('industry', 'Unbekannt'))
        
        return {
            "name": i.get('longName', symbol),
            "sector": sector,
            "currency": i.get('currency', 'USD'),
            "pe": i.get('trailingPE', 0),
            "peg": i.get('pegRatio', 0),
            "short_float": i.get('shortPercentOfFloat', None),
            "institutions": i.get('heldPercentInstitutions', None),
            "target": i.get('targetMeanPrice', 0),







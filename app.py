import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import math
import plotly.graph_objects as go # NEU: Für Profi-Charts
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

def get_symbol_and_currency(user_input, mode):
    symbol = user_input.strip()
    if mode == "Deutschland (Xetra)":
        if not symbol.endswith(".DE"): symbol += ".DE"
    elif mode == "USA (Original)":
        symbol = symbol.replace(".DE", "")
    return symbol

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
        # Robusterer Sektor-Check
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
            "rating": i.get('recommendationKey', 'none').upper()
        }
    except: return None

def get_ai_analysis(symbol, pipeline):
    """Holt News und lässt die KI bewerten"""
    if not pipeline: return "Neutral", 0, []
    
    # Basis Ticker für News (ohne .DE oft besser)
    base = symbol.replace(".DE", "")
    t = yf.Ticker(base)
    
    try:
        news = t.news
        headlines = []
        for n in news[:3]: # Top 3 News
            title = n.get('title', n.get('content', ''))
            if title: headlines.append(title)
        
        if not headlines: return "Neutral", 0, []

        results = pipeline(headlines)
        score = 0
        for res in results:
            if res['label'] == 'positive': score += 1
            elif res['label'] == 'negative': score -= 1
            
        stimmung = "Positiv" if score > 0 else ("Negativ" if score < 0 else "Neutral")
        return stimmung, score, headlines
    except:
        return "Neutral", 0, []

# --- APP START ---
st.title("🧠 Master AI Investment Terminal")

real_symbol, _ = get_symbol_and_currency(ticker_input, market_mode)

if st.button("Vollanalyse starten 🚀"):
    
    with st.spinner(f"Lade KI, Kurse & News für {real_symbol}..."):
        ai_pipeline = load_ai_model()
        period = "2y" if strategy_mode == "Value Investing (Langzeit)" else "1y"
        df = fetch_data(real_symbol, period)
        fund = get_fundamentals(real_symbol)
        
        # KI Analyse JETZT AKTIVIEREN
        ai_stimmung, ai_score, headlines = get_ai_analysis(real_symbol, ai_pipeline)

    if df is not None and fund is not None:
        stock_currency = fund['currency']
        fx_rate = get_exchange_rate(stock_currency)
        last = df.iloc[-1]
        curr_price_eur = last['Close'] * fx_rate
        
        # --- HEADER ---
        st.header(f"{fund['name']} ({fund['sector']})")
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Preis", f"{curr_price_eur:.2f} €")
        
        # Smart Money Anzeige
        inst_display = f"{fund['institutions']*100:.1f}%" if fund['institutions'] else "n/a"
        k2.metric("Institutionen", inst_display)
        
        short_display = f"{fund['short_float']*100:.2f}%" if fund['short_float'] else "n/a"
        k3.metric("Short Quote", short_display)
        
        # KI Anzeige oben
        k4.metric("KI Stimmung", ai_stimmung, f"Score: {ai_score}")

        st.divider()

        # --- SCORING SYSTEM ---
        score = 0
        reasons_pro = []
        reasons_con = []
        
        # 1. Trend
        if pd.notna(last['SMA200']):
            if last['Close'] > last['SMA200']:
                score += 1
                reasons_pro.append("Aufwärtstrend (> SMA200)")
            else:
                if strategy_mode == "Swing Trading (Kurz)":
                    score -= 0.5 # Weniger Strafe bei Swing
                    reasons_con.append("Unter SMA200 (Gegen Trend)")
                else:
                    score -= 2 # Harte Strafe bei Investing
                    reasons_con.append("Abwärtstrend (No-Go für Long-Term)")

        # 2. Bollinger Rebound (Swing Special)
        if last['Close'] > last['BB_UPPER']:
            reasons_con.append("Am oberen Band (Überhitzt)")
            if strategy_mode == "Swing Trading (Kurz)": score -= 2
        elif last['Close'] <= last['BB_LOWER'] * 1.01:
            reasons_pro.append("Am unteren Band (Rebound Chance)")
            if strategy_mode == "Swing Trading (Kurz)": score += 3 # Starker Boost!
            else: score += 1

        # 3. RSI
        if last['RSI'] < 30:
            reasons_pro.append(f"RSI Panik ({last['RSI']:.0f})")
            score += 1
        elif last['RSI'] > 70:
            reasons_con.append(f"RSI Hype ({last['RSI']:.0f})")
            score -= 1

        # 4. KI Einfluss
        if ai_stimmung == "Positiv":
            score += 1
            reasons_pro.append("KI News sind positiv")
        elif ai_stimmung == "Negativ":
            score -= 1
            reasons_con.append("KI News sind negativ")

        # --- ANZEIGE ARGUMENTE ---
        c_pro, c_con = st.columns(2)
        with c_pro:
            st.success("✅ BULLISH")
            for r in reasons_pro: st.write(f"• {r}")
        with c_con:
            st.error("❌ BEARISH")
            for r in reasons_con: st.write(f"• {r}")
            
        # News Dropdown
        with st.expander("📰 Gelesene Schlagzeilen ansehen"):
            for h in headlines: st.write(f"- {h}")

        # --- INTERAKTIVER PROFI CHART (PLOTLY) ---
        st.subheader("📊 Profi-Chart")
        
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'], name="Kurs")])
        
        # Indikatoren hinzufügen
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], line=dict(color='gray', width=1), name="Upper Band"))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], line=dict(color='gray', width=1), name="Lower Band"))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='blue', width=2), name="SMA 200"))
        
        fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # --- TRADING PLAN ---
        st.divider()
        st.subheader("📋 Trading Plan (in Euro)")
        
        action = "WARTEN"
        tp_orig = last['Close']
        sl_orig = last['Close']
        
        if strategy_mode == "Swing Trading (Kurz)":
            # Rebound Long
            if score >= 1.5:
                action = "LONG (Rebound)"
                tp_orig = last['SMA50'] if pd.notna(last['SMA50']) else last['BB_UPPER']
                sl_orig = last['BB_LOWER'] * 0.98
            # Short
            elif score <= -1.5:
                action = "SHORT"
                tp_orig = last['BB_LOWER']
                sl_orig = last['BB_UPPER'] * 1.02
        else:
            # Investing
            if score >= 3:
                action = "INVESTIEREN"
                tp_orig = fund['target'] if fund['target'] > 0 else last['Close']*1.3
                sl_orig = last['Close'] * 0.85

        # Umrechnen
        tp_eur = tp_orig * fx_rate
        sl_eur = sl_orig * fx_rate
        curr_eur = last['Close'] * fx_rate
        
        # Risiko Check
        if "SHORT" in action:
            risk = sl_eur - curr_eur
            chance = curr_eur - tp_eur
        else:
            risk = curr_eur - sl_eur
            chance = tp_eur - curr_eur
            
        crv = chance / risk if risk > 0 else 0
        budget = konto * (risk_pct / 100)
        qty = math.floor(budget / risk) if risk > 0 else 0
        
        # Anzeige
        if action == "WARTEN":
            st.warning("✋ Keine klare Chance (Score zu niedrig)")
        elif crv < 1.2:
            st.warning(f"✋ Signal {action}, aber CRV ({crv:.2f}) lohnt nicht.")
        else:
            color = "red" if "SHORT" in action else "green"
            st.markdown(f":{color}[## Empfehlung: {action}]")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stop Loss", f"{sl_eur:.2f} €")
            c2.metric("Take Profit", f"{tp_eur:.2f} €")
            c3.metric("CRV", f"{crv:.2f}")
            c4.metric("Stückzahl", f"{qty}")
            
            st.info(f"💰 Invest: {qty*curr_eur:.2f} € (Risiko: {budget:.2f} €)")

    else:
        st.error("Fehler beim Laden. Symbol prüfen!")






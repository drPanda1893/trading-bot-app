import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import math
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wall Street Insider Bot", page_icon="🏦", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Konfiguration")
market_mode = st.sidebar.selectbox("Markt", ["USA (Original)", "Deutschland (Xetra)", "Weltweit"])
ticker_input = st.sidebar.text_input("Symbol", value="NVDA").upper()
strategy_mode = st.sidebar.selectbox("Strategie", ["Value Investing (Langzeit)", "Swing Trading (Kurz)"])
konto = st.sidebar.number_input("Kontostand", value=10000.0, step=500.0)
risk_pct = st.sidebar.slider("Risiko (%)", 0.5, 5.0, 1.0)

# --- HELPER ---
def get_clean_symbol(user_input, mode):
    symbol = user_input.strip()
    if mode == "Deutschland (Xetra)":
        if not symbol.endswith(".DE"): symbol += ".DE"
    elif mode == "USA (Original)":
        symbol = symbol.replace(".DE", "")
    return symbol

@st.cache_resource
def load_ai_model():
    try: return pipeline("text-classification", model="ProsusAI/finbert")
    except: return None

def get_smart_money_data(symbol):
    """Holt Daten über Insider, Short-Seller und Institutionen"""
    t = yf.Ticker(symbol)
    try:
        info = t.info
        
        # Short Ratio: Wie viele wetten auf Absturz?
        short_percent = info.get('shortPercentOfFloat', 0)
        if short_percent is None: short_percent = 0
        
        # Institutional: Wie viel gehört den Profis?
        institutional_holdings = info.get('heldPercentInstitutions', 0)
        if institutional_holdings is None: institutional_holdings = 0
        
        data = {
            "name": info.get('longName', symbol),
            "sector": info.get('sector', 'Unbekannt'),
            "currency": info.get('currency', 'USD'),
            "pe_ratio": info.get('trailingPE', 0),
            "price_to_sales": info.get('priceToSalesTrailing12Months', 0), # Wichtig bei Tech!
            "short_percent": short_percent * 100, # In Prozent umrechnen
            "institutions": institutional_holdings * 100,
            "beta": info.get('beta', 1.0),
            "52_week_high": info.get('fiftyTwoWeekHigh', 0),
            "current_price": info.get('currentPrice', 0)
        }
        return data
    except: return None

def analyze_market(symbol):
    try:
        df = yf.download(symbol, period="2y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return None

        df['RSI'] = df.ta.rsi(length=14)
        df['ATR'] = df.ta.atr(length=14)
        df['SMA200'] = df.ta.sma(length=200) # Langfristiger Trend
        df['SMA50'] = df.ta.sma(length=50)   # Kurzfristiger Trend (Momentum)
        
        bands = df.ta.bbands(length=20, std=2)
        upper = [c for c in bands.columns if c.startswith("BBU")][0]
        df['BB_UPPER'] = bands[upper]
        return df
    except: return None

# --- APP START ---
st.title("🏦 Wall Street Insider & Smart Money Tracker")

real_symbol = get_clean_symbol(ticker_input, market_mode)

if st.button(f"Insider-Check für {real_symbol} 🚀"):
    
    with st.spinner('Analysiere Short-Seller & Institutionen...'):
        df = analyze_market(real_symbol)
        fund = get_smart_money_data(real_symbol)

    if df is not None and fund is not None:
        last = df.iloc[-1]
        curr = last['Close']
        cur_sym = "$" if fund['currency'] == "USD" else "€"
        
        st.header(f"{fund['name']} ({fund['sector']})")
        
        # --- TEIL 1: SMART MONEY FLOW (Wer kauft?) ---
        st.subheader("🕵️ Smart Money & Risiko")
        
        c1, c2, c3, c4 = st.columns(4)
        
        # A. Institutionen (Stabilität)
        inst_val = f"{fund['institutions']:.1f}%"
        inst_delta = "Stark" if fund['institutions'] > 60 else "Schwach"
        c1.metric("Institutionen Besitz", inst_val, inst_delta)
        
        # B. Short Seller (Wetten auf Crash)
        short_val = f"{fund['short_percent']:.2f}%"
        # Wenn mehr als 5% geshortet sind, ist das ein Warnsignal!
        short_state = "Normal"
        short_color = "normal"
        if fund['short_percent'] > 5:
            short_state = "⚠️ Viele wetten dagegen!"
            short_color = "inverse"
        c2.metric("Short Interest (Wetten auf Fall)", short_val, short_state, delta_color=short_color)
        
        # C. Bewertung (KGV & P/S)
        pe = fund['pe_ratio']
        pe_msg = "Fair"
        if pe > 50: pe_msg = "Teuer / Hype"
        c3.metric("KGV (P/E)", f"{pe:.1f}", pe_msg, delta_color="inverse")
        
        # D. Abstand zum Allzeithoch
        ath_dist = ((curr - fund['52_week_high']) / fund['52_week_high']) * 100
        c4.metric("Abstand zum Hoch", f"{ath_dist:.1f}%")

        # --- TEIL 2: BUBBLE CHECK (Chart Technik) ---
        st.subheader("📉 Bubble- & Trend-Check")
        
        reasons_neg = []
        reasons_pos = []
        score = 0 # Skala von -5 bis +5
        
        # 1. Abstand zur 200-Tage-Linie (Bubble Indikator)
        if pd.notna(last['SMA200']):
            dist_200 = ((curr - last['SMA200']) / last['SMA200']) * 100
            st.write(f"Abstand zur 200-Tage-Linie: **{dist_200:.1f}%**")
            
            if dist_200 > 50:
                score -= 3
                reasons_neg.append(f"EXTREM ÜBERHITZT: Kurs ist {dist_200:.0f}% über dem Durchschnitt (Bubble-Gefahr!).")
            elif dist_200 > 20:
                score -= 1
                reasons_neg.append("Aktie ist historisch teuer (weit weg von der Linie).")
            elif dist_200 < 0:
                score -= 2
                reasons_neg.append("Negativer Langzeittrend (Unter 200er Linie).")
            else:
                score += 1
                reasons_pos.append("Gesunder Aufwärtstrend.")

        # 2. SMA 50 vs SMA 200 (Momentum)
        if pd.notna(last['SMA50']) and pd.notna(last['SMA200']):
            if curr < last['SMA50'] and curr > last['SMA200']:
                score -= 2
                reasons_neg.append("⚠️ WARNUNG: Momentum verloren! Kurs fällt unter die 50-Tage-Linie.")
        
        # 3. Bewertung (Fundamental)
        if fund['pe_ratio'] > 70:
            score -= 2
            reasons_neg.append(f"Fundamental extrem teuer (KGV {fund['pe_ratio']:.0f}).")
        
        # 4. Smart Money
        if fund['institutions'] > 70:
            score += 1
            reasons_pos.append("Big Player (Fonds) halten die Aktie stabil.")
        
        if fund['short_percent'] > 5:
            score -= 2
            reasons_neg.append("Wall Street wettet vermehrt auf fallende Kurse!")

        # 5. RSI
        if last['RSI'] > 70:
            score -= 2
            reasons_neg.append("RSI Überkauft (Alle haben schon gekauft).")
        elif last['RSI'] < 30:
            score += 2
            reasons_pos.append("RSI Überverkauft (Panik im Markt = Chance).")


        # --- ANZEIGE ---
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ FÜR KAUF SPRICHT")
            for p in reasons_pos: st.write(f"• {p}")
        with col2:
            st.error("❌ DAGEGEN SPRICHT (RISIKO)")
            for n in reasons_neg: st.write(f"• {n}")
            
        st.line_chart(df[['Close', 'SMA200', 'SMA50']])

        # --- FAZIT ---
        st.divider()
        st.subheader("🤖 Algorithmus Fazit")
        
        if score >= 2:
            st.success(f"EMPFEHLUNG: KAUFEN (Score: {score})")
            st.write("Die positiven Faktoren überwiegen.")
        elif score <= -2:
            st.error(f"EMPFEHLUNG: VERKAUFEN / SHORT (Score: {score})")
            st.write("Zu viele Warnsignale (Überhitzung, fallendes Momentum).")
        else:
            st.warning(f"EMPFEHLUNG: NEUTRAL / HOLD (Score: {score})")
            st.write("Markt ist unentschlossen. Risiko zu hoch für Neueinstieg.")




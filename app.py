import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import math
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Global Pro Trader", page_icon="🌍", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Konfiguration")

# 1. BÖRSENPLATZ WÄHLEN
market_mode = st.sidebar.selectbox("Börsenplatz / Markt", ["Deutschland (Xetra .DE)", "USA (Original)", "Weltweit (Manuell)"])

ticker_input = st.sidebar.text_input("Symbol (z.B. SAP, NVDA, 9984.T)", value="NVDA").upper()
strategy_mode = st.sidebar.selectbox("Strategie", ["Value Investing (Langzeit)", "Swing Trading (Kurz)"])
konto = st.sidebar.number_input("Kontostand", value=10000.0, step=500.0) # Währung dynamisch
risk_pct = st.sidebar.slider("Risiko pro Trade (%)", 0.5, 5.0, 1.0)

# --- HELPER: SYMBOL LOGIK ---
def get_clean_symbol(user_input, mode):
    symbol = user_input.strip()
    if mode == "Deutschland (Xetra .DE)":
        if not symbol.endswith(".DE"):
            symbol += ".DE"
    elif mode == "USA (Original)":
        # US-Symbole haben keine Endung (Apple = AAPL)
        symbol = symbol.replace(".DE", "") 
    return symbol

# --- CACHING & KI ---
@st.cache_resource
def load_ai_model():
    try: return pipeline("text-classification", model="ProsusAI/finbert")
    except: return None

def get_detailed_fundamentals(symbol):
    # Bei US-Aktien ist das Symbol gleich dem Base-Symbol
    t = yf.Ticker(symbol)
    try:
        info = t.info
        curr_price = info.get('currentPrice', 0)
        target_price = info.get('targetMeanPrice', 0)
        currency = info.get('currency', 'EUR') # Währung holen!
        
        analyst_upside = 0
        if curr_price > 0 and target_price > 0:
            analyst_upside = ((target_price - curr_price) / curr_price) * 100

        data = {
            "name": info.get('longName', symbol),
            "sector": info.get('sector', 'Unbekannt'),
            "currency": currency,
            "pe_ratio": info.get('trailingPE', 0),
            "peg_ratio": info.get('pegRatio', None),
            "profit_margin": info.get('profitMargins', 0) * 100,
            "target_price": target_price,
            "analyst_rating": info.get('recommendationKey', 'none').upper(),
            "analyst_upside": analyst_upside,
            "dividend_yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            "description": info.get('longBusinessSummary', 'Keine Info.')
        }
        return data
    except: return None

def analyze_market(symbol, mode):
    period = "2y" if mode == "Value Investing (Langzeit)" else "1y"
    
    try:
        # Bei US Aktien ist Yahoo manchmal zickig mit "auto_adjust"
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.empty: return None

        df['RSI'] = df.ta.rsi(length=14)
        df['ATR'] = df.ta.atr(length=14)
        df['SMA200'] = df.ta.sma(length=200)
        
        bands = df.ta.bbands(length=20, std=2)
        upper = [c for c in bands.columns if c.startswith("BBU")][0]
        df['BB_UPPER'] = bands[upper]
        return df
    except: return None

# --- APP START ---
st.title("🌍 Global Pro Trader Dashboard")

# Symbol korrekt formatieren
real_symbol = get_clean_symbol(ticker_input, market_mode)

if st.button(f"Analyse starten für {real_symbol} 🚀"):
    
    with st.spinner(f'Lade Daten für {real_symbol}...'):
        ai_pipeline = load_ai_model()
        df = analyze_market(real_symbol, strategy_mode)
        fund = get_detailed_fundamentals(real_symbol)

    if df is not None and fund is not None:
        last = df.iloc[-1]
        curr_price = last['Close']
        cur_sym = "$" if fund['currency'] == "USD" else "€" # Währungszeichen
        
        st.header(f"{fund['name']} ({fund['sector']})")
        
        # --- LISTEN FÜR PRO & CONTRA ---
        pros = []
        cons = []
        quality_score = 0
        
        # 1. Analysten Check
        if "BUY" in fund['analyst_rating']:
            pros.append(f"Analysten Rating: {fund['analyst_rating']}")
            quality_score += 2
        elif "SELL" in fund['analyst_rating']:
            cons.append("Analysten raten zum VERKAUF")
        
        if fund['analyst_upside'] > 10:
            pros.append(f"Kurspotenzial: +{fund['analyst_upside']:.1f}%")
            quality_score += 2
        elif fund['analyst_upside'] < 0:
            cons.append(f"Kurs über Analysten-Ziel ({fund['analyst_upside']:.1f}%)")

        # 2. Bewertung
        peg = fund['peg_ratio']
        if peg is not None and peg > 0:
            if peg < 1.5:
                pros.append(f"Günstig (PEG: {peg:.2f})")
                quality_score += 2
            elif peg > 2.5:
                cons.append(f"Teuer (PEG: {peg:.2f})")

        # 3. Profitabilität
        if fund['profit_margin'] > 15:
            pros.append(f"Top Marge ({fund['profit_margin']:.1f}%)")
            quality_score += 1
        elif fund['profit_margin'] < 0:
            cons.append("Macht Verluste")

        # 4. Trend
        if pd.notna(last['SMA200']):
            if curr_price > last['SMA200']:
                pros.append("Aufwärtstrend (> SMA200)")
                quality_score += 1
            else:
                cons.append("Abwärtstrend (< SMA200)")

        # 5. RSI
        if last['RSI'] < 35:
            pros.append(f"RSI überverkauft ({last['RSI']:.0f})")
        elif last['RSI'] > 70:
            cons.append(f"RSI überhitzt ({last['RSI']:.0f})")

        # --- ANZEIGE ---
        c_pro, c_con = st.columns(2)
        with c_pro:
            st.success("✅ POSITIV")
            for p in pros: st.write(f"• {p}")
        with c_con:
            st.error("❌ NEGATIV")
            for c in cons: st.write(f"• {c}")

        st.divider()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Preis", f"{curr_price:.2f} {cur_sym}")
        k2.metric("Ziel", f"{fund['target_price']:.2f} {cur_sym}")
        k3.metric("Währung", fund['currency'])
        k4.metric("Quality Score", f"{quality_score}/8")

        st.line_chart(df['Close'])

        # --- TRADING PLAN ---
        st.subheader("📋 Plan")
        
        if strategy_mode == "Value Investing (Langzeit)":
            target = fund['target_price']
            stop = curr_price * 0.88 
        else:
            target = last['BB_UPPER']
            stop = curr_price - (2 * last['ATR'])

        # Money Management
        risk_per_share = curr_price - stop
        
        # Währungsumrechnung für Konto-Risiko (Vereinfacht: Wir nehmen an Konto ist in gleicher Währung wie Aktie)
        # In einer echten App müsste man EUR/USD umrechnen. 
        # Hier nehmen wir an: 10.000 Einheiten der Aktien-Währung.
        risk_money = konto * (risk_pct / 100)
        
        qty = math.floor(risk_money / risk_per_share) if risk_per_share > 0 else 0
        invest = qty * curr_price
        
        buy_signal = quality_score >= 5
        
        if buy_signal:
            st.success(f"EMPFEHLUNG: KAUFEN ({real_symbol})")
            c1, c2, c3 = st.columns(3)
            c1.metric("Stop Loss", f"{stop:.2f} {cur_sym}")
            c2.metric("Take Profit", f"{target:.2f} {cur_sym}")
            
            chance = target - curr_price
            crv = chance / risk_per_share if risk_per_share > 0 else 0
            c3.metric("CRV", f"{crv:.2f}")

            if qty == 0:
                st.warning("Konto zu klein für das Risiko.")
            else:
                st.info(f"Kaufe **{qty} Stück** (Invest: {invest:.2f} {cur_sym})")
        else:
            st.error("WARTEN")
            st.write(f"Score {quality_score} zu niedrig.")
            
    else:
        st.error("Konnte Daten nicht laden. Ist das Symbol korrekt? Versuch USA Mode.")



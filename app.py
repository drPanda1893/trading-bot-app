import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import math
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro Value Investor", page_icon="🏛️", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("⚙️ Konfiguration")
ticker_input = st.sidebar.text_input("Symbol (z.B. LIN, SAP, MSFT)", value="LIN").upper()
strategy_mode = st.sidebar.selectbox("Strategie", ["Value Investing (Langzeit)", "Swing Trading (Kurz)"])
konto = st.sidebar.number_input("Kontostand (€)", value=10000.0, step=500.0)
risk_pct = st.sidebar.slider("Risiko pro Trade (%)", 0.5, 5.0, 1.0)

# --- CACHING & KI ---
@st.cache_resource
def load_ai_model():
    return pipeline("text-classification", model="ProsusAI/finbert")

def get_detailed_fundamentals(symbol):
    """Holt Profi-Daten für Value Investing"""
    base = symbol.replace(".DE", "")
    t = yf.Ticker(base)
    
    try:
        info = t.info
        curr_price = info.get('currentPrice', 0)
        target_price = info.get('targetMeanPrice', 0) # Durchschnittliches Analysten-Ziel
        
        # Berechnung des Upside-Potenzials laut Analysten
        analyst_upside = 0
        if curr_price > 0 and target_price > 0:
            analyst_upside = ((target_price - curr_price) / curr_price) * 100

        data = {
            "name": info.get('longName', symbol),
            "sector": info.get('sector', 'Unbekannt'),
            "beta": info.get('beta', 1.0), # Risiko-Faktor
            "pe_ratio": info.get('trailingPE', 0),
            "peg_ratio": info.get('pegRatio', 0), # WICHTIG: Bewertung inkl. Wachstum
            "profit_margin": info.get('profitMargins', 0) * 100,
            "revenue_growth": info.get('revenueGrowth', 0) * 100,
            "target_price": target_price,
            "analyst_rating": info.get('recommendationKey', 'none').upper(), # z.B. BUY, HOLD
            "analyst_upside": analyst_upside,
            "dividend_yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            "description": info.get('longBusinessSummary', 'Keine Info.')
        }
        return data
    except Exception as e:
        return None

def analyze_market(symbol, mode):
    if not symbol.endswith(".DE") and not symbol.endswith("-USD"):
        symbol += ".DE"
    
    period = "5y" if mode == "Value Investing (Langzeit)" else "1y"
    
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty: return None

        # Indikatoren
        df['RSI'] = df.ta.rsi(length=14)
        df['ATR'] = df.ta.atr(length=14)
        df['SMA200'] = df.ta.sma(length=200)
        
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_SIGNAL'] = macd.iloc[:, 2]

        bands = df.ta.bbands(length=20, std=2)
        upper = [c for c in bands.columns if c.startswith("BBU")][0]
        df['BB_UPPER'] = bands[upper]
        
        return df
    except Exception:
        return None

def get_ai_score(symbol, ai_pipeline):
    base = symbol.replace(".DE", "")
    t = yf.Ticker(base)
    try:
        news = t.news
        if not news: return "Neutral", 0, []
        headlines = [n.get('title', '') for n in news[:3] if n.get('title')]
        if not headlines: return "Neutral", 0, []
        
        results = ai_pipeline(headlines)
        score = 0
        for res in results:
            if res['label'] == 'positive': score += 1
            elif res['label'] == 'negative': score -= 1
            
        stimmung = "Positiv" if score > 0 else ("Negativ" if score < 0 else "Neutral")
        return stimmung, score, headlines
    except:
        return "Neutral", 0, []

# --- APP LAYOUT ---

st.title("🏛️ Ultimate Value & Trading Dashboard")

if st.button("Tiefenanalyse starten 🚀"):
    
    with st.spinner('Analysiere Fundamentaldaten & KI...'):
        ai_pipeline = load_ai_model()
        df = analyze_market(ticker_input, strategy_mode)
        fund = get_detailed_fundamentals(ticker_input)

    if df is not None and fund is not None:
        last = df.iloc[-1]
        curr_price = last['Close']
        
        # --- TEIL 1: UNTERNEHMENS-QUALITÄT (Buffett Check) ---
        st.header(f"{fund['name']} ({fund['sector']})")
        
        # Wir berechnen einen "Quality Score" (0 bis 10)
        quality_score = 0
        quality_reasons = []
        
        # 1. Analysten Meinung
        if "BUY" in fund['analyst_rating'] or "STRONG_BUY" in fund['analyst_rating']:
            quality_score += 2
            quality_reasons.append("Analysten raten zum KAUF")
        
        # 2. Potenzial zum Analysten-Ziel
        if fund['analyst_upside'] > 10:
            quality_score += 2
            quality_reasons.append(f"Analysten sehen +{fund['analyst_upside']:.1f}% Potenzial")
        elif fund['analyst_upside'] > 0:
            quality_score += 1
            
        # 3. PEG Ratio (Wachstums-Bewertung)
        # Unter 1.0 ist sehr billig, unter 2.0 ist okay für Tech
        if fund['peg_ratio'] > 0 and fund['peg_ratio'] < 1.5:
            quality_score += 2
            quality_reasons.append(f"Günstige Bewertung (PEG {fund['peg_ratio']:.2f})")
        
        # 4. Profitabilität
        if fund['profit_margin'] > 15:
            quality_score += 2
            quality_reasons.append(f"Hohe Marge ({fund['profit_margin']:.1f}%)")
            
        # 5. Dividende
        if fund['dividend_yield'] > 2.0:
            quality_score += 1
            quality_reasons.append(f"Zahlt Dividende ({fund['dividend_yield']:.1f}%)")

        # 6. Trend (SMA 200) - Nur wenn wir nicht im Bärenmarkt sind
        trend_ok = False
        if pd.notna(last['SMA200']):
            if curr_price > last['SMA200']:
                quality_score += 1
                quality_reasons.append("Langfristiger Aufwärtstrend")
                trend_ok = True
            else:
                quality_reasons.append("⚠️ Unter 200-Tage-Linie (Vorsicht)")

        # Anzeige der KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Analysten Ziel", f"{fund['target_price']:.2f} €", f"{fund['analyst_upside']:.1f}%")
        c2.metric("Bewertung (PEG)", f"{fund['peg_ratio']:.2f}", delta="Gut" if fund['peg_ratio'] < 2 else "Teuer", delta_color="inverse")
        c3.metric("Marge", f"{fund['profit_margin']:.1f}%")
        c4.metric("Rating", fund['analyst_rating'])
        
        st.progress(quality_score / 10, text=f"Quality Score: {quality_score}/10")

        # --- TEIL 2: TRADING LOGIK (Timing) ---
        
        st.subheader("⏱️ Timing & Risiko")
        stimmung, ai_score, headlines = get_ai_score(ticker_input, ai_pipeline)
        
        # Chart anzeigen
        st.line_chart(df['Close'])
        
        # Entscheidung
        buy_signal = False
        
        if strategy_mode == "Value Investing (Langzeit)":
            # Wir kaufen, wenn Qualität hoch ist UND der Preis fair ist
            if quality_score >= 6: 
                buy_signal = True
                
                # ZIEL: Analysten Kursziel
                target_price = fund['target_price']
                # STOP: Weiter weg (1.5x ATR oder 15%), da wir lange halten wollen
                stop_loss = curr_price * 0.88 # 12% Puffer
                
        else: # Swing Trading
            if last['RSI'] < 40 and trend_ok:
                buy_signal = True
                target_price = last['BB_UPPER']
                stop_loss = curr_price - (2 * last['ATR'])

        # --- ERGEBNIS ---
        if buy_signal:
            st.success("✅ EMPFEHLUNG: KAUFEN")
            
            # Chance-Risiko Berechnung
            risk_share = curr_price - stop_loss
            reward_share = target_price - curr_price
            
            # CRV (Chance Risk Ratio)
            crv = reward_share / risk_share if risk_share > 0 else 0
            
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Stop Loss", f"{stop_loss:.2f} €", f"-{risk_share:.2f} €")
            col_res2.metric("Kursziel", f"{target_price:.2f} €", f"+{reward_share:.2f} €")
            col_res3.metric("CRV (Faktor)", f"{crv:.2f}")
            
            # Money Management
            risk_euro = konto * (risk_pct / 100)
            qty = math.floor(risk_euro / risk_share) if risk_share > 0 else 0
            invest = qty * curr_price
            
            st.info(f"💰 Kaufe **{qty} Stück** für {invest:.2f} €.")
            st.caption(f"Wenn der Stop-Loss greift, verlierst du exakt {risk_euro:.2f} € ({risk_pct}% deines Kontos).")
            
            if crv < 2.0:
                st.warning(f"Achtung: Das CRV ist nur {crv:.2f}. Analysten sehen aktuell wenig Luft nach oben im Vergleich zum Risiko.")
        
        else:
            st.error("⛔ EMPFEHLUNG: WARTEN / NICHT KAUFEN")
            st.write(f"Der Quality Score ({quality_score}) oder das technische Setup passen noch nicht.")
            st.write("**Positive Punkte:** " + ", ".join(quality_reasons))


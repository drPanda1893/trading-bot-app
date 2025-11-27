import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import math
from transformers import pipeline

# --- PAGE SETUP ---
st.set_page_config(page_title="Pro AI Trader", page_icon="💎", layout="wide")

# --- SIDEBAR CONFIG ---
st.sidebar.title("⚙️ Konfiguration")
ticker_input = st.sidebar.text_input("Symbol (z.B. SAP, LIN, TSLA)", value="LIN").upper()
strategy_mode = st.sidebar.selectbox("Strategie-Modus", ["Swing Trading (Kurz)", "Investing (Lang)"])
konto = st.sidebar.number_input("Kontostand (€)", value=10000.0, step=500.0)
risk_pct = st.sidebar.slider("Risiko pro Position (%)", 0.5, 5.0, 1.0)

# --- FUNKTIONEN ---

@st.cache_resource
def load_ai_model():
    """Lädt FinBERT (Sentiment Analysis)"""
    return pipeline("text-classification", model="ProsusAI/finbert")

def get_fundamentals(symbol):
    """Holt Finanzkennzahlen (Gewinn, KGV, etc.)"""
    # Basis-Ticker ohne .DE für Fundamentals (oft bessere Daten)
    base = symbol.replace(".DE", "")
    t = yf.Ticker(base)
    
    try:
        info = t.info
        # Wir suchen wichtige Kennzahlen
        data = {
            "name": info.get('longName', symbol),
            "sector": info.get('sector', 'Unbekannt'),
            "pe_ratio": info.get('trailingPE', 0), # Kurs-Gewinn-Verhältnis
            "forward_pe": info.get('forwardPE', 0), # Erwartetes KGV
            "profit_margin": info.get('profitMargins', 0) * 100, # Gewinnmarge in %
            "roe": info.get('returnOnEquity', 0) * 100, # Eigenkapitalrendite
            "revenue_growth": info.get('revenueGrowth', 0) * 100, # Umsatzwachstum
            "debt_to_equity": info.get('debtToEquity', 0), # Verschuldung
            "summary": info.get('longBusinessSummary', 'Keine Beschreibung verfügbar.')
        }
        return data
    except Exception:
        return None

def analyze_market(symbol):
    if not symbol.endswith(".DE") and not symbol.endswith("-USD"):
        symbol += ".DE"
    
    # Für Investing brauchen wir mehr Historie (SMA 200)
    period = "2y" if strategy_mode == "Investing (Lang)" else "1y"
    
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty: return None

        # Indikatoren
        df['RSI'] = df.ta.rsi(length=14)
        df['ATR'] = df.ta.atr(length=14)
        df['SMA200'] = df.ta.sma(length=200) # Wichtig für Langzeit
        
        # MACD
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_SIGNAL'] = macd.iloc[:, 2]

        # Bollinger (für Swing)
        bands = df.ta.bbands(length=20, std=2)
        upper = [c for c in bands.columns if c.startswith("BBU")][0]
        df['BB_UPPER'] = bands[upper]
        
        return df
    except Exception as e:
        st.error(f"Fehler beim Chart-Laden: {e}")
        return None

def get_ai_score(symbol, ai_pipeline):
    base = symbol.replace(".DE", "")
    t = yf.Ticker(base)
    try:
        news = t.news
    except: return "Neutral", 0, []
    
    if not news: return "Neutral", 0, []
    
    headlines = []
    for n in news[:3]:
        title = n.get('title', n.get('content', ''))
        if title: headlines.append(title)
        
    if not headlines: return "Neutral", 0, []
    
    try:
        results = ai_pipeline(headlines)
        score = 0
        for res in results:
            if res['label'] == 'positive': score += 1
            elif res['label'] == 'negative': score -= 1
        
        stimmung = "Positiv" if score > 0 else ("Negativ" if score < 0 else "Neutral")
        return stimmung, score, headlines
    except:
        return "Neutral", 0, headlines

# --- HAUPT APP ---

st.title("💎 Ultimate AI Investment Dashboard")

if st.button("Analyse starten 🚀"):
    
    # 1. KI Laden
    with st.spinner('Lade KI...'):
        ai_pipeline = load_ai_model()

    # 2. Daten laden
    df = analyze_market(ticker_input)
    fund_data = get_fundamentals(ticker_input)

    if df is not None:
        last = df.iloc[-1]
        curr_price = last['Close']
        
        # --- TEIL 1: FUNDAMENTALANALYSE (Finanz-Check) ---
        if fund_data:
            st.header(f"🏢 {fund_data['name']} ({fund_data['sector']})")
            
            # 4 Spalten für Finanz-KPIs
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            # KGV (PE Ratio) Bewertung
            pe = fund_data['pe_ratio']
            pe_color = "normal"
            if pe > 0 and pe < 20: pe_color = "normal" # Günstig
            elif pe > 40: pe_color = "off" # Teuer
            
            kpi1.metric("KGV (P/E)", f"{pe:.1f}", delta="-Günstig" if pe < 20 else "Teuer", delta_color=pe_color)
            kpi2.metric("Gewinnmarge", f"{fund_data['profit_margin']:.1f}%", delta="Profitabel" if fund_data['profit_margin']>10 else None)
            kpi3.metric("Umsatzwachstum", f"{fund_data['revenue_growth']:.1f}%")
            kpi4.metric("Schulden/Ek", f"{fund_data['debt_to_equity']:.2f}")
            
            with st.expander("Unternehmens-Beschreibung lesen"):
                st.write(fund_data['summary'])
        else:
            st.warning("Keine Fundamentaldaten verfügbar.")

        # --- TEIL 2: STRATEGIE CHECK ---
        st.divider()
        st.subheader(f"📊 Strategie-Analyse: {strategy_mode}")
        
        stimmung, ai_score, headlines = get_ai_score(ticker_input, ai_pipeline)
        
        score = 0
        reasons = []
        
        # --- LOGIK WEICHE ---
        if strategy_mode == "Swing Trading (Kurz)":
            # Kurzfristige Logik (RSI, MACD)
            target_price = last['BB_UPPER']
            stop_loss = curr_price - (2 * last['ATR'])
            
            if last['RSI'] < 40: 
                score += 1
                reasons.append("RSI günstig (<40)")
            if last['MACD'] > last['MACD_SIGNAL']:
                score += 1
                reasons.append("Momentum positiv (MACD)")
            if stimmung == "Positiv":
                score += 1
                reasons.append("KI News positiv")

        else: # Investing (Lang)
            # Langfristige Logik (Trend + Fundamentals)
            
            # Ziel: Wir halten lange. Ziel ist z.B. +20% pauschal oder das Allzeithoch
            target_price = curr_price * 1.30 # 30% Upside Ziel
            stop_loss = curr_price * 0.85    # 15% Trailing Stop (weiter weg!)
            
            # A. Trend Check (Über 200 Tage Linie?)
            if pd.notna(last['SMA200']):
                if curr_price > last['SMA200']:
                    score += 1
                    reasons.append("Langzeittrend intakt (> SMA200)")
                else:
                    reasons.append("WARNUNG: Unter 200-Tage-Linie (Bärenmarkt)")
            
            # B. Fundamental Check
            if fund_data:
                if fund_data['profit_margin'] > 10:
                    score += 1
                    reasons.append("Hohe Profitabilität (>10%)")
                if fund_data['revenue_growth'] > 5:
                    score += 0.5
                    reasons.append("Unternehmen wächst")
            
            # C. KI Check (Langfristig sind News weniger wichtig, aber gut zu wissen)
            if stimmung == "Negativ":
                score -= 0.5 # Nur Abzug bei schlechten News

        # --- ERGEBNIS ANZEIGE ---
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.line_chart(df['Close'])
        
        with col2:
            st.write("### KI Stimmung")
            st.info(f"{stimmung} (Score: {ai_score})")
            for h in headlines:
                st.caption(f"• {h}")

        # --- HANDELSANWEISUNG & PERFORMANCE ---
        st.subheader("💡 Ergebnis & Plan")
        
        min_score = 2.0 if strategy_mode == "Swing Trading (Kurz)" else 2.5
        
        if score >= min_score:
            st.success(f"✅ KAUF-SIGNAL (Score: {score})")
            st.write(f"**Begründung:** {', '.join(reasons)}")
            
            # Money Management Berechnung
            risk_per_share = curr_price - stop_loss
            risk_euro = konto * (risk_pct / 100)
            
            qty = 0
            if risk_per_share > 0:
                qty = math.floor(risk_euro / risk_per_share)
            
            invest_total = qty * curr_price
            potential_win = (target_price - curr_price) * qty
            potential_loss = (curr_price - stop_loss) * qty
            
            # Die "Trading Karte"
            c1, c2, c3 = st.columns(3)
            c1.metric("Einstieg", f"{curr_price:.2f} €")
            c2.metric("Stop Loss", f"{stop_loss:.2f} €", f"-{potential_loss:.2f} €")
            c3.metric("Take Profit (Ziel)", f"{target_price:.2f} €", f"+{potential_win:.2f} €")
            
            st.info(f"🛒 **Order-Vorschlag:** Kaufe **{qty} Stück** (Investition: {invest_total:.2f} €)")
            
        else:
            st.warning(f"✋ WARTEN (Score: {score}/{min_score})")
            st.write("Aktuell keine idealen Bedingungen für diese Strategie.")
            st.write(f"Gründe dagegen/neutral: {', '.join(reasons) if reasons else 'Keine positiven Signale.'}")

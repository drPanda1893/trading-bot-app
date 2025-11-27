import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import math
from transformers import pipeline

# --- KONFIGURATION DER SEITE ---
st.set_page_config(page_title="AI Trading Bot", page_icon="📈")

# --- TITEL & SIDEBAR ---
st.title("📈 AI & Algo Trading Dashboard")
st.write("Analysiert Aktien mit RSI, MACD, Bollinger Bändern & Künstlicher Intelligenz (FinBERT).")

# Sidebar für Eingaben
st.sidebar.header("Einstellungen")
ticker_input = st.sidebar.text_input("Aktien-Symbol (z.B. SAP, LIN, TSLA)", value="LIN")
konto = st.sidebar.number_input("Kontostand (€)", value=10000.0, step=100.0)
risk_pct = st.sidebar.slider("Risiko pro Trade (%)", 0.5, 5.0, 1.0)


# --- FUNKTIONEN (Mit Caching für Speed) ---

@st.cache_resource
def load_ai_model():
    """Lädt die KI nur EINMAL, damit die App schnell bleibt."""
    return pipeline("text-classification", model="ProsusAI/finbert")


def analyze_market(symbol):
    # .DE Anhängen für deutsche Aktien, falls vergessen
    if not symbol.endswith(".DE") and not symbol.endswith("-USD"):
        # Einfache Heuristik: Wenn keine Punkte im Namen, probieren wir DE
        # (Für US Ticker kann man das weglassen)
        symbol += ".DE"

    status_text = st.empty()
    status_text.text(f"Lade Daten für {symbol}...")

    # Daten laden
    try:
        df = yf.download(symbol, period="1y", interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            st.error("Keine Daten gefunden. Ticker prüfen!")
            return None

        # Indikatoren
        df['RSI'] = df.ta.rsi(length=14)
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_SIGNAL'] = macd.iloc[:, 2]
        df['ATR'] = df.ta.atr(length=14)
        bands = df.ta.bbands(length=20, std=2)
        upper_col = [c for c in bands.columns if c.startswith("BBU")][0]
        df['BB_UPPER'] = bands[upper_col]

        status_text.empty()
        return df
    except Exception as e:
        st.error(f"Fehler: {e}")
        return None


def get_ai_score(symbol, ai_pipeline):
    # Basis Symbol für News (ohne .DE)
    base = symbol.replace(".DE", "")
    ticker = yf.Ticker(base)
    news = ticker.news

    if not news:
        return "Neutral", 0, []

    headlines = [n['title'] for n in news[:3]]
    results = ai_pipeline(headlines)

    score = 0
    for res in results:
        if res['label'] == 'positive':
            score += 1
        elif res['label'] == 'negative':
            score -= 1

    stimmung = "Neutral"
    if score > 0: stimmung = "Positiv"
    if score < 0: stimmung = "Negativ"

    return stimmung, score, headlines


# --- HAUPT LOGIK ---

if st.button("Analyse Starten 🚀"):

    # 1. KI Laden (Mit Ladebalken)
    with st.spinner('Lade KI-Modell (beim ersten Mal dauert es kurz)...'):
        ai_pipeline = load_ai_model()

    # 2. Daten & Technik
    df = analyze_market(ticker_input)

    if df is not None:
        last = df.iloc[-1]
        curr_price = last['Close']

        # 3. KI Analyse
        stimmung, ai_score, headlines = get_ai_score(ticker_input, ai_pipeline)

        # 4. Score Berechnung
        tech_score = 0
        reasons = []

        # RSI
        if last['RSI'] < 35:
            tech_score += 1
            reasons.append("RSI günstig (<35)")

        # MACD
        if last['MACD'] > last['MACD_SIGNAL']:
            tech_score += 1
            reasons.append("Trend positiv (MACD)")

        # KI Einfluss
        total_score = tech_score
        if stimmung == "Positiv":
            total_score += 1
            reasons.append("KI News positiv")
        elif stimmung == "Negativ":
            total_score -= 1
            reasons.append("KI News negativ")

        # --- ANZEIGE ---

        # Spalten für Metriken
        col1, col2, col3 = st.columns(3)
        col1.metric("Preis", f"{curr_price:.2f} €")
        col2.metric("RSI", f"{last['RSI']:.2f}")
        col3.metric("KI Stimmung", stimmung)

        # Chart
        st.line_chart(df['Close'])

        # Ergebnis Box
        st.subheader("🤖 Ergebnis")

        upside = ((last['BB_UPPER'] - curr_price) / curr_price) * 100

        if total_score >= 2:
            st.success(f"STRONG BUY (Score: {total_score})")
            st.write(f"**Gründe:** {', '.join(reasons)}")
            st.write(f"**Chance:** +{upside:.2f}% (bis oberes Band)")

            # Risk Management
            stop_loss = curr_price - (2 * last['ATR'])
            risk_amt = konto * (risk_pct / 100)
            risk_share = curr_price - stop_loss
            qty = math.floor(risk_amt / risk_share) if risk_share > 0 else 0

            st.info(f"💰 Money Management: Kaufe **{qty} Stück** (Invest: {qty * curr_price:.2f}€)")

        else:
            st.warning(f"WAIT / NEUTRAL (Score: {total_score})")
            st.write("Aktuell kein klares Signal.")

        # News anzeigen
        with st.expander("Gelesene Schlagzeilen ansehen"):
            for h in headlines:
                st.write(f"- {h}")
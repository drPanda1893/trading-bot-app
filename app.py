import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import math
import numpy as np
import plotly.graph_objects as go
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Master AI Trader Pro", page_icon="🧠", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🎛️ Steuerzentrale")
market_mode = st.sidebar.selectbox("Markt / Börse", ["Deutschland (Xetra)", "USA (Original)", "Weltweit"])
ticker_input = st.sidebar.text_input("Symbol", value="LIN").upper()
strategy_mode = st.sidebar.selectbox("Strategie", ["Swing Trading (Kurz)", "Value Investing (Langzeit)"])

st.sidebar.divider()
st.sidebar.subheader("🧠 Bot-Persönlichkeit")
aggro_mode = st.sidebar.select_slider(
    "Risikobereitschaft:",
    options=["Sicherheits-Fanatiker 🛡️", "Ausgewogen ⚖️", "Risiko-Freudig (Degen) 🚀"],
    value="Ausgewogen ⚖️"
)

st.sidebar.divider()
konto = st.sidebar.number_input("Dein Kontostand (€)", value=10000.0, step=500.0)
risk_pct = st.sidebar.slider("Risiko pro Trade (%)", 0.5, 10.0, 2.0)

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
    currency = "USD" 
    if mode == "Deutschland (Xetra)":
        if not symbol.endswith(".DE"): symbol += ".DE"
        currency = "EUR"
    elif mode == "USA (Original)":
        symbol = symbol.replace(".DE", "")
        currency = "USD"
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
        
        # SIGMA (Volatilität) Berechnung
        # Wir berechnen die Standardabweichung der täglichen Returns
        df['Returns'] = df['Close'].pct_change()
        # Annualisierte Volatilität (Sigma) = StdDev * Wurzel(252 Handelstage)
        sigma = df['Returns'].std() * (252 ** 0.5)
        
        return df, sigma
    except: return None, None

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
            "rating": i.get('recommendationKey', 'none').upper()
        }
    except: return None

def get_ai_analysis(symbol, pipeline):
    if not pipeline: return "Neutral", 0, []
    base = symbol.replace(".DE", "")
    t = yf.Ticker(base)
    try:
        news = t.news
        headlines = []
        for n in news[:3]:
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
        df, sigma_val = fetch_data(real_symbol, period)
        fund = get_fundamentals(real_symbol)
        ai_stimmung, ai_score, headlines = get_ai_analysis(real_symbol, ai_pipeline)

    if df is not None and fund is not None:
        stock_currency = fund['currency']
        fx_rate = get_exchange_rate(stock_currency)
        last = df.iloc[-1]
        curr_price_eur = last['Close'] * fx_rate
        
        st.header(f"{fund['name']} ({fund['sector']})")
        
        # Top KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Preis", f"{curr_price_eur:.2f} €")
        
        # Smart Money
        inst_display = f"{fund['institutions']*100:.1f}%" if fund['institutions'] else "n/a"
        k2.metric("Institutionen", inst_display)
        
        # Sigma Anzeige (NEU)
        sigma_display = f"{sigma_val*100:.1f}%" if sigma_val else "n/a"
        # Färbung: Hohes Sigma (>40%) ist rot (riskant), niedriges (<20%) ist grün
        sigma_color = "normal"
        if sigma_val and sigma_val > 0.4: sigma_color = "inverse" 
        k3.metric("Sigma (Volatilität)", sigma_display, "Risiko p.a.", delta_color=sigma_color)
        
        k4.metric("KI Stimmung", ai_stimmung, f"Score: {ai_score}")

        st.divider()

        # --- SCORE LOGIK MIT PROTOKOLLIERUNG ---
        score = 0
        score_log = [] # Hier speichern wir alle Details
        
        # Helper für Log
        def add_log(criteria, value, points, reason):
            score_log.append({"Kriterium": criteria, "Wert": value, "Punkte": points, "Erklärung": reason})

        # 1. Trend
        if pd.notna(last['SMA200']):
            trend_val = f"{last['SMA200']:.2f}"
            if last['Close'] > last['SMA200']:
                score += 1
                add_log("Trend (SMA200)", trend_val, "+1", "Kurs ist über der 200-Tage-Linie (Aufwärtstrend)")
            else:
                if strategy_mode == "Swing Trading (Kurz)":
                    score -= 0.5
                    add_log("Trend (SMA200)", trend_val, "-0.5", "Kurs unter Linie (Gegen Trend, bei Swing ok)")
                else:
                    score -= 2
                    add_log("Trend (SMA200)", trend_val, "-2.0", "Kurs unter Linie (Bärenmarkt)")

        # 2. Bollinger
        bb_low = last['BB_LOWER']
        bb_up = last['BB_UPPER']
        
        if last['Close'] > bb_up:
            if strategy_mode == "Swing Trading (Kurz)": 
                score -= 2
                add_log("Bollinger Band", "Oben", "-2.0", "Kurs durchbricht oberes Band (Überkauft)")
            else:
                add_log("Bollinger Band", "Oben", "0", "Kurs ist hoch (für Value weniger relevant)")
                
        elif last['Close'] <= bb_low * 1.02: 
            if strategy_mode == "Swing Trading (Kurz)": 
                score += 3
                add_log("Bollinger Band", "Unten", "+3.0", "Kurs am unteren Band (Perfekter Rebound Einstieg)")
            else: 
                score += 1
                add_log("Bollinger Band", "Unten", "+1.0", "Kurs ist günstig (am unteren Band)")
        else:
            add_log("Bollinger Band", "Mitte", "0", "Kurs läuft innerhalb der Bänder")

        # 3. RSI
        rsi_val = f"{last['RSI']:.1f}"
        if last['RSI'] < 35:
            score += 1
            add_log("RSI", rsi_val, "+1.0", "RSI ist niedrig (Überverkauft)")
        elif last['RSI'] > 70:
            score -= 1
            add_log("RSI", rsi_val, "-1.0", "RSI ist zu hoch (Überkauft)")
        else:
            add_log("RSI", rsi_val, "0", "RSI ist neutral")

        # 4. KI
        if ai_stimmung == "Positiv":
            score += 1
            add_log("KI News", "Positiv", "+1.0", "KI bewertet Schlagzeilen positiv")
        elif ai_stimmung == "Negativ":
            score -= 1
            add_log("KI News", "Negativ", "-1.0", "KI bewertet Schlagzeilen negativ")
        else:
            add_log("KI News", "Neutral", "0", "Keine eindeutige Stimmung in den News")

        # 5. Fundamental (Nur bei Investing)
        if strategy_mode == "Value Investing (Langzeit)":
            peg = fund['peg']
            if peg is not None and 0 < peg < 1.5:
                score += 1
                add_log("PEG Ratio", f"{peg:.2f}", "+1.0", "Aktie ist fundamental günstig bewertet")
            elif peg > 2.5:
                score -= 1
                add_log("PEG Ratio", f"{peg:.2f}", "-1.0", "Aktie ist fundamental teuer")

        # --- ANZEIGE SCORE DETAILS ---
        col_chart, col_score = st.columns([2, 1])
        
        with col_chart:
            st.subheader("📊 Profi-Chart")
            fig = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'], high=df['High'],
                            low=df['Low'], close=df['Close'], name="Kurs")])
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], line=dict(color='gray', width=1), name="Upper Band"))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], line=dict(color='gray', width=1), name="Lower Band"))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], line=dict(color='blue', width=2), name="SMA 200"))
            fig.update_layout(height=400, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with col_score:
            st.subheader(f"🔍 Score-Analyse ({score})")
            # Schöne Tabelle erstellen
            df_log = pd.DataFrame(score_log)
            st.dataframe(df_log, hide_index=True, use_container_width=True)
            
            with st.expander("🔢 Alle Rohdaten (Deep Dive)"):
                st.write(f"**Aktueller Kurs:** {last['Close']:.2f}")
                st.write(f"**SMA 200:** {last['SMA200']:.2f}")
                st.write(f"**SMA 50:** {last['SMA50']:.2f}")
                st.write(f"**RSI:** {last['RSI']:.2f}")
                st.write(f"**ATR:** {last['ATR']:.2f}")
                st.write(f"**Bollinger Oben:** {last['BB_UPPER']:.2f}")
                st.write(f"**Bollinger Unten:** {last['BB_LOWER']:.2f}")
                st.write(f"**Sigma (Volatilität):** {sigma_val*100:.2f}%")

        st.divider()
        
        # --- TRADING PLAN ---
        st.subheader(f"📋 Trading Plan (in Euro)")
        
        # Dynamische Schwellen
        min_score_long = 2.0
        min_score_short = -2.0
        min_crv = 1.2
        
        if aggro_mode == "Sicherheits-Fanatiker 🛡️":
            min_score_long = 3.0
            min_score_short = -3.0
            min_crv = 1.5
        elif aggro_mode == "Risiko-Freudig (Degen) 🚀":
            min_score_long = 0.5 
            min_score_short = -0.5
            min_crv = 0.6 
        
        action = "WARTEN"
        tp_orig = last['Close']
        sl_orig = last['Close']
        
        # Logik
        if strategy_mode == "Swing Trading (Kurz)":
            if score >= min_score_long:
                action = "LONG (Rebound)"
                if "Risiko-Freudig" in aggro_mode:
                    tp_orig = last['BB_UPPER']
                else:
                    tp_orig = last['SMA50'] if pd.notna(last['SMA50']) else last['BB_UPPER']
                sl_orig = last['BB_LOWER'] * 0.98
            elif score <= min_score_short:
                action = "SHORT"
                if "Risiko-Freudig" in aggro_mode:
                    tp_orig = last['BB_LOWER']
                else:
                    tp_orig = last['SMA50'] if pd.notna(last['SMA50']) else last['BB_LOWER']
                sl_orig = last['BB_UPPER'] * 1.02
        else:
            if score >= (min_score_long + 1):
                action = "INVESTIEREN"
                tp_orig = fund['target'] if fund['target'] > 0 else last['Close']*1.3
                sl_orig = last['Close'] * 0.85

        # Umrechnen
        tp_eur = tp_orig * fx_rate
        sl_eur = sl_orig * fx_rate
        curr_eur = last['Close'] * fx_rate
        
        if "SHORT" in action:
            risk = sl_eur - curr_eur
            chance = curr_eur - tp_eur
        else:
            risk = curr_eur - sl_eur
            chance = tp_eur - curr_eur
            
        crv = chance / risk if risk > 0 else 0
        budget = konto * (risk_pct / 100)
        qty = math.floor(budget / risk) if risk > 0 else 0
        invest_sum = qty * curr_eur
        
        # MÜ & SIGMA BERECHNUNG (Das Finale)
        base_prob = 0.50
        prob_boost = abs(score) * 0.05 
        win_prob = min(0.85, base_prob + prob_boost)
        loss_prob = 1.0 - win_prob
        
        total_risk = qty * risk
        total_reward = qty * chance
        
        # Mü (Erwartungswert in Euro)
        mu_value = (total_reward * win_prob) - (total_risk * loss_prob)
        
        # ANZEIGE
        if action == "WARTEN":
            st.warning(f"✋ Keine klare Chance (Score {score} zu niedrig für '{aggro_mode}')")
        elif crv < min_crv:
            st.warning(f"✋ Signal {action}, aber CRV ({crv:.2f}) lohnt nicht.")
        else:
            color = "red" if "SHORT" in action else "green"
            st.markdown(f":{color}[## Empfehlung: {action}]")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stop Loss", f"{sl_eur:.2f} €")
            c2.metric("Take Profit", f"{tp_eur:.2f} €")
            c3.metric("CRV", f"{crv:.2f}")
            c4.metric("Stückzahl", f"{qty}")
            
            st.divider()
            
            # MÜ & SIGMA BOX
            col_math1, col_math2 = st.columns(2)
            
            with col_math1:
                st.info(f"🧮 **Erwartungswert (Mü): {mu_value:.2f} €**")
                st.caption("Durchschnittlicher Gewinn pro Trade bei dieser Strategie.")
                
            with col_math2:
                # Sigma des Trades (nicht der Aktie) ist schwer exakt zu berechnen ohne Monte Carlo,
                # aber wir nehmen das Aktien-Sigma als Risiko-Indikator
                st.warning(f"⚡ **Risiko-Faktor (Sigma): {sigma_val*100:.1f}%**")
                st.caption("Jährliche Schwankungsbreite der Aktie. Je höher, desto wilder der Ritt.")

            st.success(f"💰 Investiere: **{invest_sum:.2f} €** (Risiko: {budget:.2f} €)")

    else:
        st.error("Fehler beim Laden.")









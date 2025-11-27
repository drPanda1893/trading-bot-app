import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import math
from transformers import pipeline

# --- PAGE CONFIG ---
st.set_page_config(page_title="Master AI Trader", page_icon="🧠", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🎛️ Steuerzentrale")

# Modus Wahl
market_mode = st.sidebar.selectbox("Markt / Börse", ["USA (Original)", "Deutschland (Xetra)", "Weltweit"])
ticker_input = st.sidebar.text_input("Symbol", value="NVDA").upper()
strategy_mode = st.sidebar.selectbox("Strategie", ["Value Investing (Langzeit)", "Swing Trading (Kurz)"])
konto = st.sidebar.number_input("Dein Kontostand (€)", value=10000.0, step=500.0)
risk_pct = st.sidebar.slider("Risiko pro Trade (%)", 0.5, 5.0, 1.0)

# --- HELPER FUNKTIONEN ---

def get_exchange_rate(from_currency):
    """Holt den aktuellen Wechselkurs zu Euro"""
    if from_currency == "EUR": return 1.0
    
    try:
        # Yahoo Ticker für Währungen: EUR=X bedeutet "Wie viel USD ist 1 Euro wert?"
        # Wir brauchen aber andersrum: Wie viel Euro ist 1 USD wert?
        if from_currency == "USD":
            pair = "EUR=X" # Wert von 1 USD in EUR (oft ca 0.95)
            # Achtung: Yahoo Notation ist oft verwirrend. Wir prüfen den Wert.
            tick = yf.Ticker(pair)
            rate = tick.history(period="1d")['Close'].iloc[-1]
            return rate 
        else:
            # Fallback für andere Währungen (vereinfacht 1:1, um Fehler zu vermeiden)
            return 1.0 
    except:
        return 1.0 # Fallback bei Fehler

@st.cache_resource
def load_ai_model():
    try: return pipeline("text-classification", model="ProsusAI/finbert")
    except: return None

def get_symbol_and_currency(user_input, mode):
    symbol = user_input.strip()
    currency_code = "USD" # Default Annahme
    
    if mode == "Deutschland (Xetra)":
        if not symbol.endswith(".DE"): symbol += ".DE"
        currency_code = "EUR"
    elif mode == "USA (Original)":
        symbol = symbol.replace(".DE", "")
        currency_code = "USD"
    
    return symbol, currency_code

def fetch_data(symbol, period):
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        if df.empty: return None

        # Indikatoren Berechnung
        df['RSI'] = df.ta.rsi(length=14)
        df['ATR'] = df.ta.atr(length=14)
        df['SMA50'] = df.ta.sma(length=50)
        df['SMA200'] = df.ta.sma(length=200)
        
        # MACD
        macd = df.ta.macd(fast=12, slow=26, signal=9)
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_SIGNAL'] = macd.iloc[:, 2]

        # Bollinger Bänder (WICHTIG für Swing)
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
        return {
            "name": i.get('longName', symbol),
            "sector": i.get('sector', 'Unbekannt'),
            "currency": i.get('currency', 'USD'),
            "pe": i.get('trailingPE', 0),
            "peg": i.get('pegRatio', 0),
            "beta": i.get('beta', 0),
            "short_float": i.get('shortPercentOfFloat', None), # Kann None sein!
            "institutions": i.get('heldPercentInstitutions', None), # Kann None sein!
            "target": i.get('targetMeanPrice', 0),
            "rating": i.get('recommendationKey', 'none').upper()
        }
    except: return None

# --- APP START ---
st.title("🧠 Master AI Investment Terminal")

real_symbol, estimated_currency = get_symbol_and_currency(ticker_input, market_mode)
ai_pipeline = load_ai_model()

if st.button("Vollanalyse starten 🚀"):
    
    # Ladebalken
    with st.spinner(f"Analysiere {real_symbol} & berechne Währungskurse..."):
        # Zeitrahmen je nach Strategie
        period = "2y" if strategy_mode == "Value Investing (Langzeit)" else "1y"
        df = fetch_data(real_symbol, period)
        fund = get_fundamentals(real_symbol)
        
        # KI Sentiment
        news_score = 0
        ai_msg = "Keine News"
        if ai_pipeline:
            # Fake News Abruf (vereinfacht für Speed im Beispiel)
            # In echter App hier get_ai_score nutzen
            pass 

    if df is not None and fund is not None:
        # --- WÄHRUNGS-UMRECHNUNG ---
        # Wir holen den Kurs von der Aktie zur UI (Euro)
        stock_currency = fund['currency'] # z.B. USD
        fx_rate = get_exchange_rate(stock_currency) # z.B. 0.95
        
        last = df.iloc[-1]
        curr_price_orig = last['Close']
        curr_price_eur = curr_price_orig * fx_rate
        
        # --- HEADER ---
        st.header(f"{fund['name']} ({fund['sector']})")
        
        # Top KPIs in Euro
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Preis (Live)", f"{curr_price_eur:.2f} €", help=f"Original: {curr_price_orig:.2f} {stock_currency}")
        
        # Smart Money (Nur anzeigen wenn Daten da sind!)
        if fund['institutions'] is not None and fund['institutions'] > 0:
            val = fund['institutions'] * 100
            k2.metric("Institutionen", f"{val:.1f}%", "Stark" if val > 60 else "Schwach")
        else:
            k2.metric("Institutionen", "n/a", help="Daten für diesen Markt nicht verfügbar")
            
        # Short Interest
        if fund['short_float'] is not None:
            val = fund['short_float'] * 100
            k3.metric("Short Quote", f"{val:.2f}%", "Hoch!" if val > 5 else "Normal", delta_color="inverse")
        else:
            k3.metric("Short Quote", "n/a")
            
        # PE / KGV
        pe_val = f"{fund['pe']:.1f}" if fund['pe'] > 0 else "k.A."
        k4.metric("KGV (P/E)", pe_val)

        st.divider()

        # --- ANALYSE LOGIK (Score Berechnung) ---
        score = 0
        reasons_pro = []
        reasons_con = []
        
        # 1. Trend (SMA)
        if pd.notna(last['SMA200']):
            if curr_price_orig > last['SMA200']:
                score += 1
                reasons_pro.append("Aufwärtstrend (Kurs > 200-Tage-Linie)")
                trend_direction = "LONG"
            else:
                score -= 1
                reasons_con.append("Abwärtstrend (Kurs < 200-Tage-Linie)")
                trend_direction = "SHORT"
        
        # 2. Momentum (MACD)
        if last['MACD'] > last['MACD_SIGNAL']:
            score += 1
            reasons_pro.append("MACD Momentum positiv")
        else:
            score -= 1
            reasons_con.append("MACD Momentum negativ")
            
        # 3. Bollinger Bänder (Mean Reversion)
        # Ist der Kurs am oberen Band? (Überkauft -> Short Chance)
        if curr_price_orig > last['BB_UPPER']:
            reasons_con.append("Kurs am oberen Bollinger Band (Überhitzt)")
            if strategy_mode == "Swing Trading (Kurz)": score -= 2 
        # Ist der Kurs am unteren Band? (Überverkauft -> Long Chance)
        elif curr_price_orig < last['BB_LOWER']:
            reasons_pro.append("Kurs am unteren Bollinger Band (Rebound Chance)")
            if strategy_mode == "Swing Trading (Kurz)": score += 2

        # 4. RSI
        if last['RSI'] > 70:
            reasons_con.append(f"RSI extrem hoch ({last['RSI']:.0f})")
            score -= 1
        elif last['RSI'] < 30:
            reasons_pro.append(f"RSI extrem niedrig ({last['RSI']:.0f})")
            score += 1

        # --- ANZEIGE DER ARGUMENTE ---
        c_pro, c_con = st.columns(2)
        with c_pro:
            st.success("✅ BULLISH (FÜR KAUF/LONG)")
            for r in reasons_pro: st.write(f"• {r}")
        with c_con:
            st.error("❌ BEARISH (FÜR VERKAUF/SHORT)")
            for r in reasons_con: st.write(f"• {r}")

        # --- CHART ---
        st.subheader("📊 Chart Analyse (Bollinger & SMA)")
        # Wir zeigen den Chart in Original-Währung (sauberer), aber Rechenwerte in Euro
        chart_data = df[['Close', 'SMA200', 'BB_UPPER', 'BB_LOWER']]
        st.line_chart(chart_data)

       # --- TRADING PLAN (DAS HERZSTÜCK) ---
        st.divider()
        st.subheader("📋 Dein Trading Plan (in Euro)")
        
        # Ziele setzen basierend auf Strategie
        stop_loss_eur = 0.0
        take_profit_eur = 0.0
        action = "NEUTRAL"
        
        # Logik für SWING TRADING (Bollinger Bänder)
        if strategy_mode == "Swing Trading (Kurz)":
            # LONG SZENARIO (Wir kaufen unten)
            if score >= 1: 
                action = "LONG (Kaufen)"
                take_profit_orig = last['BB_UPPER']
                stop_loss_orig = curr_price_orig - (1.5 * last['ATR'])
                
            # SHORT SZENARIO (Wir wetten auf Fall)
            elif score <= -1:
                action = "SHORT (Auf Fall wetten)"
                take_profit_orig = last['BB_LOWER']
                stop_loss_orig = curr_price_orig + (1.5 * last['ATR'])
            else:
                action = "WARTEN"
                take_profit_orig = curr_price_orig
                stop_loss_orig = curr_price_orig
                
        # Logik für VALUE INVESTING (Langzeit)
        else:
            if score >= 2:
                action = "INVESTIEREN (Long)"
                take_profit_orig = fund['target'] if fund['target'] > 0 else (curr_price_orig * 1.3)
                stop_loss_orig = curr_price_orig * 0.85
            else:
                action = "NICHT INVESTIEREN"
                take_profit_orig = curr_price_orig
                stop_loss_orig = curr_price_orig

        # UMRECHNUNG IN EURO
        tp_eur = take_profit_orig * fx_rate
        sl_eur = stop_loss_orig * fx_rate
        
        # RISIKO BERECHNUNG & CRV
        if "SHORT" in action:
            # Bei Short: Risiko entsteht wenn Preis STEIGT (SL > Einstieg)
            risk_per_share_eur = sl_eur - curr_price_eur
            chance_per_share_eur = curr_price_eur - tp_eur
        else: # LONG
            # Bei Long: Risiko entsteht wenn Preis FÄLLT (SL < Einstieg)
            risk_per_share_eur = curr_price_eur - sl_eur
            chance_per_share_eur = tp_eur - curr_price_eur
            
        # CRV Berechnung (Sicherheit vor Division durch 0)
        if risk_per_share_eur > 0:
            crv = chance_per_share_eur / risk_per_share_eur
        else:
            crv = 0
        
        # Stückzahl
        risk_budget = konto * (risk_pct / 100)
        qty = math.floor(risk_budget / risk_per_share_eur) if risk_per_share_eur > 0 else 0
        invest_sum = qty * curr_price_eur

        # --- FINALE BOX MIT CRV-FILTER ---
        
        # Wir filtern Trades raus, die sich mathematisch nicht lohnen (CRV < 1.0)
        is_profitable_trade = crv > 1.2  # Mindestanforderung: 20% mehr Gewinn als Risiko
        
        if action == "WARTEN" or action == "NICHT INVESTIEREN":
            st.warning(f"✋ Empfehlung: {action}")
            st.write("Kein klares Signal vom Algorithmus.")
            
        elif not is_profitable_trade:
            # NEU: Warnung statt Empfehlung, wenn CRV schlecht ist
            st.warning(f"✋ Signal vorhanden ({action}), aber LOHNT NICHT.")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Risiko", f"-{risk_per_share_eur:.2f} €/Aktie")
            c2.metric("Chance", f"+{chance_per_share_eur:.2f} €/Aktie")
            c3.metric("CRV", f"{crv:.2f}", delta="Schlecht", delta_color="inverse")
            
            st.error("Der Kurs ist bereits zu nah am Ziel. Das Risiko ist größer als der mögliche Gewinn!")
            
        else:
            # GUTER TRADE
            if "SHORT" in action:
                st.error(f"📉 Empfehlung: {action}")
                st.caption("Bei SHORT: Stop Loss liegt ÜBER dem aktuellen Preis.")
            else:
                st.success(f"📈 Empfehlung: {action}")
                
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Stop Loss", f"{sl_eur:.2f} €", help="Verkaufen wenn erreicht")
            c2.metric("Take Profit", f"{tp_eur:.2f} €", help="Ziel erreicht")
            c3.metric("CRV", f"{crv:.2f}", delta="Gut" if crv > 2 else "Ok")
            c4.metric("Stückzahl", f"{qty}", help=f"Budget: {risk_budget:.2f}€")
            
            st.info(f"💰 Einsatz: **{invest_sum:.2f} €** (Max. Verlust: -{risk_budget:.2f} € | Ziel-Gewinn: +{qty*chance_per_share_eur:.2f} €)")





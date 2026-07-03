"""
Seasonal Week Scanner v2
------------------------
Five-pillar scoring for candidate stocks over the next 1-2 weeks:
  Seasonal 40% | Trend 25% | Setup 15% | Risk 10% | News 10%

RUN LOCALLY:  streamlit run seasonal_scanner.py
DEPLOY:       replace seasonal_scanner.py on GitHub; Render redeploys itself.
"""

import datetime as dt
import math

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Seasonal Week Scanner v2", layout="wide")

DEFAULT_TICKERS = (
    "AAPL,MSFT,AMZN,GOOGL,NVDA,META,ORCL,IBM,INTC,CSCO,"
    "ADBE,CRM,TXN,QCOM,AMD,MU,ACN,INTU,"
    "JPM,BAC,WFC,C,GS,MS,AXP,BLK,SCHW,USB,"
    "JNJ,PFE,MRK,ABT,LLY,BMY,AMGN,GILD,MDT,UNH,"
    "PG,KO,PEP,WMT,COST,MCD,NKE,SBUX,HD,LOW,"
    "XOM,CVX,COP,SLB,OXY,GE,HON,CAT,DE,MMM,"
    "BA,LMT,RTX,UPS,FDX,UNP,CSX,DIS,CMCSA,VZ,"
    "T,TMO,DHR,SYK,ISRG,SO,DUK,NEE,D,LIN,"
    # Playbook additions
    "TEVA,CAH,GH,"                       # healthcare
    "NOW,RBRK,CRWD,ANET,APH,NFLX,CRDO,PANW,"  # technology
    "KTOS,RDW,RKLB,"                     # space & defense
    "FCX,PR,MP,CEG,VST,"                 # energy, metals, minerals
    "O,EPD,TMUS"                         # defensive tilt
)
BENCH, VIX = "^GSPC", "^VIX"

POS_WORDS = {"beat", "beats", "surge", "surges", "record", "upgrade", "upgraded",
             "raises", "raised", "strong", "growth", "rally", "gain", "gains",
             "buy", "outperform", "bullish", "profit", "wins", "award", "approval",
             "expands", "partnership", "breakthrough", "tops", "higher", "jump"}
NEG_WORDS = {"miss", "misses", "plunge", "plunges", "downgrade", "downgraded",
             "cuts", "cut", "weak", "lawsuit", "probe", "investigation", "recall",
             "sell", "underperform", "bearish", "loss", "losses", "layoff",
             "layoffs", "warns", "warning", "falls", "lower", "drop", "fraud",
             "bankruptcy", "delay", "halt", "slump", "fears", "debt"}

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
anchor = st.sidebar.date_input("Anchor date (window starts after this day)",
                               value=dt.date.today())
lookback = st.sidebar.slider("Lookback years", 10, 25, 25)
hold_days = st.sidebar.slider("Holding period (trading days)", 3, 10, 10)
tickers_text = st.sidebar.text_area("Tickers (comma separated)",
                                    DEFAULT_TICKERS, height=150)
min_years = st.sidebar.slider(
    "Minimum years of history", 4, 25, 8,
    help="Lower this to include young stocks (RKLB, RDW, CEG, PR, CRDO...). "
         "Warning: seasonal stats on <10 samples are close to meaningless — "
         "for young names, trust the Trend/Setup/Risk pillars instead.")
check_extras = st.sidebar.slider("Earnings+news lookups for top N (slower)",
                                 5, 30, 15)
run = st.sidebar.button("Run scan", type="primary")

st.title("Seasonal Week Scanner v2")
st.caption(
    "Five-pillar score: Seasonal 40% · Trend 25% · Setup 15% · Risk 10% · "
    "News 10%. Educational tool, NOT investment advice. Survivorship bias "
    "applies: today's tickers exclude past delistings."
)


# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def download_prices(tickers: tuple, start: str) -> pd.DataFrame:
    return yf.download(list(tickers), start=start, interval="1d",
                       auto_adjust=True, progress=False)


def week_returns(px, month, day, first_year, last_year, hold):
    out = {}
    for yr in range(first_year, last_year + 1):
        try:
            ts = pd.Timestamp(dt.date(yr, month, day))
        except ValueError:
            ts = pd.Timestamp(dt.date(yr, month, day - 1))
        w = px[(px.index > ts) & (px.index <= ts + pd.Timedelta(days=30))]
        if len(w) >= hold + 1:
            out[yr] = float(w.iloc[hold] / w.iloc[0] - 1) * 100
    return out


def rsi(close: pd.Series, period: int) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1 / period, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period).mean()
    v = (100 - 100 / (1 + ag / al)).dropna()
    return float(v.iloc[-1]) if len(v) else float("nan")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def scale(x, lo, hi):
    """Map x in [lo, hi] -> [0, 100]."""
    if pd.isna(x):
        return 50.0
    return 100.0 * (clamp(x, lo, hi) - lo) / (hi - lo)


def grade(score):
    return ("A" if score >= 75 else "B" if score >= 65 else
            "C" if score >= 55 else "D" if score >= 45 else "F")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def earnings_in_window(ticker: str, start: dt.date, end: dt.date):
    """True/False/None(unknown) if earnings falls inside [start, end]."""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = []
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date") or []
        elif cal is not None and hasattr(cal, "loc"):
            row = cal.loc["Earnings Date"] if "Earnings Date" in cal.index else []
            dates = list(row) if hasattr(row, "__iter__") else [row]
        for d in dates:
            d = pd.Timestamp(d).date()
            if start <= d <= end:
                return True
        return False if dates else None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 3)
def news_sentiment(ticker: str):
    """(score -1..1, n_headlines, worst_negative_title) from recent headlines."""
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return None, 0, ""
    titles = []
    for it in items[:12]:
        t = it.get("title") or (it.get("content") or {}).get("title") or ""
        if t:
            titles.append(t)
    if not titles:
        return None, 0, ""
    total, worst_neg, worst_title = 0, 0, ""
    for t in titles:
        words = {w.strip(".,!?:;()'\"").lower() for w in t.split()}
        pos = len(words & POS_WORDS)
        neg = len(words & NEG_WORDS)
        total += pos - neg
        if neg - pos > worst_neg:
            worst_neg, worst_title = neg - pos, t
    return clamp(total / len(titles), -1, 1), len(titles), worst_title


# ---------------- Main ----------------
if not run:
    st.info("Set your date and tickers in the sidebar, then click **Run scan**.")
    st.stop()

tickers = tuple(sorted({t.strip().upper() for t in tickers_text.split(",")
                        if t.strip()}))
this_year = anchor.year
first_year = this_year - lookback
start = f"{first_year - 1}-01-01"

with st.spinner(f"Downloading {len(tickers)} tickers + S&P 500 + VIX..."):
    raw = download_prices(tickers + (BENCH, VIX), start)
closes, volumes = raw["Close"], raw["Volume"]

# ----- Market regime banner -----
spx_px = closes[BENCH].dropna()
spx_200 = float(spx_px.tail(200).mean())
spx_last = float(spx_px.iloc[-1])
try:
    vix_last = float(closes[VIX].dropna().iloc[-1])
except Exception:
    vix_last = float("nan")
regime_ok = spx_last > spx_200 and (pd.isna(vix_last) or vix_last < 25)
regime_txt = (f"S&P 500 {'ABOVE' if spx_last > spx_200 else 'BELOW'} its "
              f"200-day average ({spx_last:,.0f} vs {spx_200:,.0f}); "
              f"VIX at {vix_last:.1f}" if not pd.isna(vix_last) else "")
if regime_ok:
    st.success(f"Market regime: FAVORABLE for long setups — {regime_txt}")
else:
    st.warning(f"Market regime: CAUTION — {regime_txt}. Seasonal long signals "
               "are historically less reliable in weak regimes.")

spx_seasonal = week_returns(spx_px, anchor.month, anchor.day,
                            first_year, this_year - 1, hold_days)
spx_wins = sum(1 for r in spx_seasonal.values() if r > 0)

# ----- Per-stock pillars -----
rows = []
rets_map, price_map = {}, {}
for t in tickers:
    if t not in closes.columns:
        continue
    px = closes[t].dropna()
    if len(px) < 210:
        continue
    stock = week_returns(px, anchor.month, anchor.day,
                         first_year, this_year - 1, hold_days)
    common = [yr for yr in stock if yr in spx_seasonal]
    if len(common) < min_years:
        continue
    n = len(common)
    rets = [stock[yr] for yr in common]
    excess = [stock[yr] - spx_seasonal[yr] for yr in common]
    win_pct = 100 * sum(1 for r in rets if r > 0) / n
    beat_pct = 100 * sum(1 for e in excess if e > 0) / n
    avg_ret = sum(rets) / n
    avg_ex = sum(excess) / n
    sd = (sum((e - avg_ex) ** 2 for e in excess) / (n - 1)) ** 0.5
    tstat = avg_ex / (sd / math.sqrt(n)) if sd > 0 else 0.0

    # Trend
    last = float(px.iloc[-1])
    dma50 = float(px.tail(50).mean())
    dma200 = float(px.tail(200).mean())
    mom1m = (last / float(px.iloc[-22]) - 1) * 100 if len(px) > 22 else float("nan")
    mom3m = (last / float(px.iloc[-64]) - 1) * 100 if len(px) > 64 else float("nan")

    # Setup
    rsi14 = rsi(px.tail(160), 14)
    rsi2 = rsi(px.tail(60), 2)
    vol = volumes[t].dropna()
    vol_ratio = float(vol.iloc[-1] / vol.tail(30).mean()) if len(vol) >= 30 else float("nan")

    # Risk (volatility)
    dr = px.tail(31).pct_change().dropna()
    ann_vol = float(dr.std() * math.sqrt(252) * 100) if len(dr) > 5 else float("nan")

    # ----- Pillar scores (0-100) -----
    seasonal_s = 0.35 * win_pct + 0.35 * beat_pct + 0.30 * scale(tstat, 0, 3)
    trend_s = (25 * (last > dma50) + 25 * (last > dma200)
               + 0.25 * scale(mom1m, -10, 10) + 0.25 * scale(mom3m, -20, 20))
    # Setup: reward neutral-to-oversold RSI in context, normal-to-elevated volume
    if pd.isna(rsi14):
        rsi_s = 50
    elif rsi14 >= 75:
        rsi_s = 20
    elif rsi14 >= 65:
        rsi_s = 45
    elif rsi14 >= 40:
        rsi_s = 75
    elif rsi14 >= 30:
        rsi_s = 65 if last > dma200 else 40
    else:
        rsi_s = 80 if last > dma200 else 25   # oversold in uptrend = opportunity
    timing_bonus = 10 if (not pd.isna(rsi2) and rsi2 < 10 and last > dma200) else 0
    vol_s = 50 if pd.isna(vol_ratio) else (70 if 0.8 <= vol_ratio <= 1.8
                                           else 45 if vol_ratio < 0.8 else 35)
    setup_s = clamp(0.7 * rsi_s + 0.3 * vol_s + timing_bonus, 0, 100)
    risk_s = 100 - scale(ann_vol, 15, 60)  # low vol -> high score

    rets_map[t], price_map[t] = sorted(rets), last
    rows.append({
        "Ticker": t, "Years": n, "Win %": round(win_pct, 1),
        "Beat %": round(beat_pct, 1), "Avg Ret %": round(avg_ret, 2),
        "Avg Excess %": round(avg_ex, 2), "t-stat": round(tstat, 2),
        ">50DMA": "Y" if last > dma50 else "N",
        ">200DMA": "Y" if last > dma200 else "N",
        "1M %": round(mom1m, 1), "3M %": round(mom3m, 1),
        "RSI14": round(rsi14, 1), "RSI2": round(rsi2, 1),
        "Vol vs 30d": round(vol_ratio, 2) if not pd.isna(vol_ratio) else None,
        "AnnVol %": round(ann_vol, 1),
        "_seasonal": seasonal_s, "_trend": trend_s,
        "_setup": setup_s, "_risk": risk_s,
    })

if not rows:
    st.warning("No tickers had enough history — lower minimum years.")
    st.stop()

df = pd.DataFrame(rows)
df["_prelim"] = (0.40 * df["_seasonal"] + 0.25 * df["_trend"]
                 + 0.15 * df["_setup"] + 0.10 * df["_risk"] + 0.10 * 50)
df = df.sort_values("_prelim", ascending=False).reset_index(drop=True)

# ----- Earnings + news for top N only (slow lookups) -----
win_start = anchor + dt.timedelta(days=1)
win_end = anchor + dt.timedelta(days=hold_days * 2 + 4)
earn_flags, news_scores, news_notes = {}, {}, {}
prog = st.progress(0.0, text="Checking earnings dates and news headlines...")
top_n = df.head(check_extras)["Ticker"].tolist()
for i, t in enumerate(top_n):
    earn_flags[t] = earnings_in_window(t, win_start, win_end)
    s, n_headlines, worst = news_sentiment(t)
    news_scores[t] = s
    news_notes[t] = worst
    prog.progress((i + 1) / len(top_n))
prog.empty()

def final_row(r):
    t = r["Ticker"]
    earn = earn_flags.get(t)          # True / False / None
    news = news_scores.get(t)         # -1..1 / None
    risk = r["_risk"] - (50 if earn else 0)
    news_s = 50 if news is None else scale(news, -1, 1)
    total = (0.40 * r["_seasonal"] + 0.25 * r["_trend"] + 0.15 * r["_setup"]
             + 0.10 * clamp(risk, 0, 100) + 0.10 * news_s)
    return pd.Series({
        "Score": round(total, 1), "Grade": grade(total),
        "Seasonal": round(r["_seasonal"], 0), "Trend": round(r["_trend"], 0),
        "Setup": round(r["_setup"], 0), "Risk": round(clamp(risk, 0, 100), 0),
        "News": round(news_s, 0),
        "Earnings in window": ("YES ⚠" if earn else
                               "No" if earn is False else "?"),
        "News flag": (news_notes.get(t, "")[:60] if news is not None
                      and news < -0.2 else ""),
    })

extras = df.apply(final_row, axis=1)
out = pd.concat([df, extras], axis=1)
out = out.sort_values("Score", ascending=False).reset_index(drop=True)

# ----- Display -----
c1, c2, c3, c4 = st.columns(4)
c1.metric("S&P up in this window",
          f"{spx_wins}/{len(spx_seasonal)} yrs "
          f"({100 * spx_wins / len(spx_seasonal):.0f}%)")
c2.metric("S&P avg return", f"{sum(spx_seasonal.values())/len(spx_seasonal):+.2f}%")
c3.metric("Stocks scanned", f"{len(out)}")
c4.metric("Regime", "Favorable" if regime_ok else "Caution")

# ----- Top 10 with historical-implied price projections -----
st.subheader(f"Top 10 picks — {hold_days}-trading-day outlook")
proj_rows = []
for _, r in out.head(10).iterrows():
    t = r["Ticker"]
    rets, price = rets_map[t], price_map[t]
    s = pd.Series(rets)
    med, p25, p75 = s.median(), s.quantile(0.25), s.quantile(0.75)
    proj_rows.append({
        "Rank": len(proj_rows) + 1, "Ticker": t, "Grade": r["Grade"],
        "Score": r["Score"],
        "Hist up odds": f"{r['Win %']:.0f}%",
        "Price now": round(price, 2),
        "Target (hist median)": round(price * (1 + med / 100), 2),
        "Range (25th-75th pct)":
            f"{price * (1 + p25 / 100):,.2f} - {price * (1 + p75 / 100):,.2f}",
        "Earnings in window": r["Earnings in window"],
    })
st.dataframe(pd.DataFrame(proj_rows), use_container_width=True,
             hide_index=True)
st.caption(
    "⚠ 'Target' = current price moved by the stock's MEDIAN historical "
    "return for this exact calendar window; the range spans its 25th-75th "
    "percentile outcomes. These are historical echoes, not forecasts — "
    "roughly half of past years finished outside the range shown. "
    "'Hist up odds' is the past win rate, not a true probability."
)

st.subheader("Top 10 detail")
show_cols = ["Ticker", "Score", "Grade", "Seasonal", "Trend", "Setup", "Risk",
             "News", "Win %", "Beat %", "t-stat", ">50DMA", ">200DMA",
             "1M %", "RSI14", "RSI2", "AnnVol %", "Earnings in window",
             "News flag"]
st.dataframe(out.head(10)[show_cols], use_container_width=True, hide_index=True)

st.subheader("Full results")
full_cols = [c for c in out.columns if not c.startswith("_")]
st.dataframe(out[full_cols], use_container_width=True, hide_index=True)
st.download_button("Download CSV", out[full_cols].to_csv(index=False),
                   file_name="seasonal_scan_v2.csv")

with st.expander("How the score works"):
    st.markdown(
        "**Score = 40% Seasonal + 25% Trend + 15% Setup + 10% Risk + 10% News** "
        "(each pillar 0-100; Grade: A ≥75, B ≥65, C ≥55, D ≥45, else F)\n\n"
        "- **Seasonal** — win rate, beat-the-S&P rate, and the t-statistic of "
        "average excess return. The t-stat separates reliable patterns from "
        "lucky streaks; above ~2 is statistically meaningful, below 1 is noise\n"
        "- **Trend** — above the 50- and 200-day moving averages, plus 1- and "
        "3-month momentum. Stocks in uptrends continue up more often\n"
        "- **Setup** — RSI(14) zone (neutral or oversold-in-uptrend scores "
        "best, overbought worst), RSI(2) as a short-term timing bonus, and "
        "volume vs its 30-day average\n"
        "- **Risk** — lower volatility scores higher; an earnings report "
        "inside the holding window halves the pillar (event risk dominates "
        "2-week horizons)\n"
        "- **News** — crude keyword sentiment on recent headlines (top-ranked "
        "stocks only). A negative flag shows the most concerning headline. "
        "Treat as a tripwire to investigate, not a verdict\n\n"
        "**Honest limits:** ~25 seasonal samples per stock is thin; news "
        "scoring is keyword-based, not true language understanding; and no "
        "score makes a 2-week move predictable — this tilts odds and flags "
        "hazards, nothing more. Always verify the ⚠ flags before acting."
    )

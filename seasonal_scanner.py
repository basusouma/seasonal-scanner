"""
Seasonal Week Scanner
---------------------
Pick any date. The app looks at the NEXT 5 trading days after that
calendar date in each of the past N years and shows which stocks:
  - rose most often in that week (win rate)
  - beat the S&P 500 most often (beat rate, avg excess)
and pairs it with TODAY's technical context: RSI(14) and volume vs 30-day avg.

RUN:
  pip install streamlit yfinance pandas
  streamlit run seasonal_scanner.py
"""

import datetime as dt

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Seasonal Week Scanner", layout="wide")

DEFAULT_TICKERS = (
    "AAPL,MSFT,AMZN,GOOGL,NVDA,META,ORCL,IBM,INTC,CSCO,"
    "ADBE,CRM,TXN,QCOM,AMD,MU,ACN,INTU,"
    "JPM,BAC,WFC,C,GS,MS,AXP,BLK,SCHW,USB,"
    "JNJ,PFE,MRK,ABT,LLY,BMY,AMGN,GILD,MDT,UNH,"
    "PG,KO,PEP,WMT,COST,MCD,NKE,SBUX,HD,LOW,"
    "XOM,CVX,COP,SLB,OXY,GE,HON,CAT,DE,MMM,"
    "BA,LMT,RTX,UPS,FDX,UNP,CSX,DIS,CMCSA,VZ,"
    "T,TMO,DHR,SYK,ISRG,SO,DUK,NEE,D,LIN"
)
BENCH = "^GSPC"

# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")
anchor = st.sidebar.date_input("Anchor date (week starts after this day)",
                               value=dt.date.today())
lookback = st.sidebar.slider("Lookback years", 10, 25, 25)
hold_days = st.sidebar.slider("Holding period (trading days)", 3, 10, 5)
tickers_text = st.sidebar.text_area("Tickers (comma separated)",
                                    DEFAULT_TICKERS, height=150)
min_years = st.sidebar.slider("Minimum years of history required", 10, 25, 15)
run = st.sidebar.button("Run scan", type="primary")

st.title("Seasonal Week Scanner")
st.caption(
    f"Historical behavior of the {hold_days} trading days after "
    f"{anchor.strftime('%B %d')} in each of the past {lookback} years, "
    "combined with current RSI and volume. Educational tool, not advice. "
    "Beware survivorship bias: today's tickers exclude past delistings."
)


# ---------------- Helpers ----------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def download_prices(tickers: tuple, start: str) -> pd.DataFrame:
    raw = yf.download(list(tickers), start=start, interval="1d",
                      auto_adjust=True, progress=False)
    return raw


def week_returns(px: pd.Series, month: int, day: int,
                 first_year: int, last_year: int, hold: int) -> dict:
    """{year: % return} for `hold` trading days after (month, day)."""
    out = {}
    for yr in range(first_year, last_year + 1):
        try:
            anchor_ts = pd.Timestamp(dt.date(yr, month, day))
        except ValueError:  # Feb 29 in a non-leap year
            anchor_ts = pd.Timestamp(dt.date(yr, month, day - 1))
        window = px[(px.index > anchor_ts)
                    & (px.index <= anchor_ts + pd.Timedelta(days=25))]
        if len(window) >= hold + 1:
            out[yr] = float(window.iloc[hold] / window.iloc[0] - 1) * 100
    return out


def rsi14(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    val = 100 - 100 / (1 + rs)
    return float(val.dropna().iloc[-1]) if not val.dropna().empty else float("nan")


def rsi_flag(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    if v >= 70:
        return "Overbought"
    if v <= 30:
        return "Oversold"
    return "Neutral"


# ---------------- Main ----------------
if run:
    tickers = tuple(sorted({t.strip().upper() for t in tickers_text.split(",")
                            if t.strip()}))
    this_year = anchor.year
    first_year = this_year - lookback
    start = f"{first_year - 1}-01-01"

    with st.spinner(f"Downloading {len(tickers)} tickers + S&P 500..."):
        raw = download_prices(tickers + (BENCH,), start)
    closes, volumes = raw["Close"], raw["Volume"]

    spx = week_returns(closes[BENCH].dropna(), anchor.month, anchor.day,
                       first_year, this_year - 1, hold_days)
    spx_wins = sum(1 for r in spx.values() if r > 0)

    rows = []
    for t in tickers:
        if t not in closes.columns:
            continue
        px = closes[t].dropna()
        stock = week_returns(px, anchor.month, anchor.day,
                             first_year, this_year - 1, hold_days)
        common = [yr for yr in stock if yr in spx]
        if len(common) < min_years:
            continue
        rets = [stock[yr] for yr in common]
        excess = [stock[yr] - spx[yr] for yr in common]
        wins = sum(1 for r in rets if r > 0)
        beats = sum(1 for e in excess if e > 0)
        n = len(common)

        # Current technicals
        recent_close = px.tail(120)
        rsi_val = rsi14(recent_close)
        vol = volumes[t].dropna()
        vol_ratio = (float(vol.iloc[-1] / vol.tail(30).mean())
                     if len(vol) >= 30 else float("nan"))

        win_pct = 100 * wins / n
        beat_pct = 100 * beats / n
        avg_excess = sum(excess) / n
        # Transparent composite: seasonality only (technicals shown as context)
        score = round(0.4 * win_pct + 0.4 * beat_pct
                      + 20 * max(min(avg_excess, 3), -3) / 3, 1)

        rows.append({
            "Ticker": t, "Years": n,
            "Up Years": wins, "Win %": round(win_pct, 1),
            "Beat SPX": beats, "Beat %": round(beat_pct, 1),
            "Avg Ret %": round(sum(rets) / n, 2),
            "Avg Excess %": round(avg_excess, 2),
            "Score": score,
            "RSI(14)": round(rsi_val, 1) if not pd.isna(rsi_val) else None,
            "RSI Zone": rsi_flag(rsi_val),
            "Vol vs 30d": round(vol_ratio, 2) if not pd.isna(vol_ratio) else None,
        })

    if not rows:
        st.warning("No tickers had enough history. Lower the minimum years "
                   "or check the ticker list.")
        st.stop()

    df = pd.DataFrame(rows).sort_values("Score", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("S&P 500 up in this week",
              f"{spx_wins}/{len(spx)} yrs ({100 * spx_wins / len(spx):.0f}%)")
    c2.metric("S&P 500 avg return", f"{sum(spx.values()) / len(spx):+.2f}%")
    c3.metric("Stocks scanned", f"{len(df)}")

    st.subheader("Top seasonal candidates")
    top = df.head(10)
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.subheader("Full results")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", df.to_csv(index=False),
                       file_name="seasonal_scan.csv")

    with st.expander("How to read this"):
        st.markdown(
            "- **Win %** — how often the stock rose in this week historically\n"
            "- **Beat % / Avg Excess %** — how often and by how much it beat "
            "the S&P 500 in the same week (this separates a real seasonal "
            "edge from general market drift)\n"
            "- **Score** — 40% win rate + 40% beat rate + 20% scaled excess. "
            "Seasonality only; technicals are context, not part of the score\n"
            "- **RSI(14)** — today's momentum. Below 30 = oversold (bounce "
            "candidates), above 70 = overbought (chase risk)\n"
            "- **Vol vs 30d** — today's volume relative to its 30-day "
            "average. Above ~1.5 means unusual activity: check the news "
            "before trusting the seasonal pattern\n\n"
            "With ~25 samples per stock, even an 80% win rate can be luck. "
            "Use this as a screen for further research, never as a signal "
            "by itself."
        )
else:
    st.info("Set your date and tickers in the sidebar, then click **Run scan**.")

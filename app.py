import io
from datetime import datetime
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="RSI + Candle Scanner", page_icon="📈", layout="wide")

NSE_URL = "https://archives.nseindia.com/content/indices/ind_{}list.csv"
UNIVERSES = {"Nifty 50": "nifty50", "Nifty 100": "nifty100",
             "Nifty 200": "nifty200", "Nifty 500": "nifty500"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
CHUNK = 100
try:
    pd.Series([1], index=pd.to_datetime(["2024-01-31"])).resample("ME").last()
    MONTH = "ME"
except ValueError:
    MONTH = "M"


# ---------------------------------------------------------------- universe
@st.cache_data(ttl=86400, show_spinner=False)
def load_universe(key: str) -> pd.DataFrame:
    try:
        r = requests.get(NSE_URL.format(key), headers=HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
    except Exception:
        df = pd.read_csv(f"universe/{key}.csv")  # optional local fallback
    if "Industry" not in df:
        df["Industry"] = ""
    return df[["Symbol", "Industry"]].dropna(subset=["Symbol"]).drop_duplicates("Symbol")


# ---------------------------------------------------------------- data
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_chunk(symbols: tuple) -> dict:
    tk = [s + ".NS" for s in symbols]
    raw = yf.download(tk, period="8y", interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    out = {}
    if raw is None or raw.empty:
        return out
    if not isinstance(raw.columns, pd.MultiIndex):
        raw = pd.concat({tk[0]: raw}, axis=1)
    avail = set(raw.columns.get_level_values(0))
    for s, t in zip(symbols, tk):
        if t in avail:
            d = raw[t][["Open", "High", "Low", "Close"]].dropna()
            if len(d):
                if d.index.tz is not None:
                    d.index = d.index.tz_localize(None)
                out[s] = d
    return out


# ---------------------------------------------------------------- indicators
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan)).fillna(np.inf)


def candle_info(df: pd.DataFrame):
    """Returns (bullish patterns string, candle type) for the latest daily candle."""
    o, h, l, c = (df[k].to_numpy(float)[-12:] for k in ("Open", "High", "Low", "Close"))
    if len(c) < 7:
        return "", ""
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-9)
    up = h - np.maximum(o, c)
    lo = np.minimum(o, c) - l
    avg = max(body[-11:-1].mean(), 1e-9)
    g, r = c > o, c < o
    down = c[-2] < c[-6]                      # short-term decline before the candle
    p = []

    if down and body[-1] / rng[-1] > 0.1 and lo[-1] >= 2 * body[-1] and up[-1] <= 0.15 * rng[-1]:
        p.append("Hammer")
    if down and body[-1] / rng[-1] <= 0.1 and lo[-1] >= 0.65 * rng[-1] and up[-1] <= 0.1 * rng[-1]:
        p.append("Dragonfly Doji")
    if r[-2] and g[-1] and c[-1] >= o[-2] and o[-1] <= c[-2] and body[-1] > body[-2]:
        p.append("Bullish Engulfing")
    if r[-2] and body[-2] >= avg and g[-1] and o[-1] < c[-2] and (o[-2] + c[-2]) / 2 < c[-1] < o[-2]:
        p.append("Piercing Line")
    if r[-2] and body[-2] >= avg and g[-1] and o[-1] > c[-2] and c[-1] < o[-2] and body[-1] < 0.6 * body[-2]:
        p.append("Bullish Harami")
    if r[-3] and body[-3] >= avg and body[-2] <= 0.4 * body[-3] and max(o[-2], c[-2]) < c[-3] \
            and g[-1] and c[-1] > (o[-3] + c[-3]) / 2:
        p.append("Morning Star")
    if g[-1] and g[-2] and g[-3] and c[-1] > c[-2] > c[-3] \
            and o[-3] < o[-2] < c[-3] and o[-2] < o[-1] < c[-2] \
            and all(up[-k] <= 0.3 * body[-k] for k in (1, 2, 3)):
        p.append("Three White Soldiers")
    if down and r[-2] and g[-1] and abs(l[-1] - l[-2]) <= 0.002 * c[-1]:
        p.append("Tweezer Bottom")
    marubozu = body[-1] >= 0.9 * rng[-1] and body[-1] >= 1.2 * avg
    if g[-1] and marubozu:
        p.append("Bullish Marubozu")

    b = body[-1] / rng[-1]
    if b <= 0.1:
        t = ("Dragonfly Doji" if lo[-1] >= 0.65 * rng[-1]
             else "Gravestone Doji" if up[-1] >= 0.65 * rng[-1] else "Doji")
    elif marubozu:
        t = "Bullish Marubozu" if g[-1] else "Bearish Marubozu"
    elif b <= 0.3 and up[-1] >= body[-1] and lo[-1] >= body[-1]:
        t = "Spinning Top"
    elif lo[-1] >= 2 * body[-1] and up[-1] <= 0.15 * rng[-1]:
        t = "Hammer-shape"
    elif up[-1] >= 2 * body[-1] and lo[-1] <= 0.15 * rng[-1]:
        t = "Shooting Star-shape"
    elif g[-1]:
        t = "Long Green" if body[-1] >= 1.5 * avg else "Green"
    else:
        t = "Long Red" if body[-1] >= 1.5 * avg else "Red"
    return ", ".join(p), t


def analyse(sym: str, df: pd.DataFrame, n: int):
    close = df["Close"]
    if len(close) < 300:
        return None
    d = rsi(close, n).iloc[-1]
    w = rsi(close.resample("W-FRI").last().dropna(), n).iloc[-1]
    m = rsi(close.resample(MONTH).last().dropna(), n).iloc[-1]
    if np.isnan([d, w, m]).any():
        return None
    pat, typ = candle_info(df)
    return {"Symbol": sym, "Close": round(close.iloc[-1], 2),
            "Chg %": round((close.iloc[-1] / close.iloc[-2] - 1) * 100, 2),
            "Monthly RSI": round(m, 1), "Weekly RSI": round(w, 1), "Daily RSI": round(d, 1),
            "Bullish Pattern": pat, "Candle Type": typ,
            "From 52W High %": round((close.iloc[-1] / close.tail(252).max() - 1) * 100, 1)}


def run_scan(symbols, n, bar):
    rows, chunks = [], [symbols[i:i + CHUNK] for i in range(0, len(symbols), CHUNK)]
    for i, ch in enumerate(chunks):
        data = fetch_chunk(tuple(ch))
        for s in ch:
            res = analyse(s, data[s], n) if s in data else None
            if res:
                rows.append(res)
        bar.progress((i + 1) / len(chunks), text=f"Scanned {min((i + 1) * CHUNK, len(symbols))}/{len(symbols)}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- UI
st.title("📈 NSE RSI + Candlestick Scanner")
st.caption("Monthly & Weekly RSI strong, Daily RSI pulled back — with bullish candle detection")

with st.sidebar:
    st.header("Segment")
    seg = st.selectbox("Universe", list(UNIVERSES) + ["Custom list"], index=3)
    custom = st.text_area("Custom symbols (comma separated)", "RELIANCE, TCS, INFY") if seg == "Custom list" else ""
    st.header("RSI filters")
    rsi_len = st.number_input("RSI length", 5, 30, 14)
    m_min = st.slider("Monthly RSI above", 40, 90, 60)
    w_min = st.slider("Weekly RSI above", 40, 90, 60)
    d_lo, d_hi = st.slider("Daily RSI between", 20, 70, (35, 45))
    only_pat = st.checkbox("Only stocks with a bullish pattern", False)
    go = st.button("🔍 Scan", type="primary", use_container_width=True)
    if st.button("Clear cache", use_container_width=True):
        st.cache_data.clear()

if go:
    try:
        if seg == "Custom list":
            syms = sorted({s.strip().upper().replace(".NS", "") for s in custom.replace("\n", ",").split(",") if s.strip()})
            meta = pd.DataFrame({"Symbol": syms, "Industry": ""})
        else:
            meta = load_universe(UNIVERSES[seg])
            syms = meta["Symbol"].tolist()
    except Exception:
        st.error("Could not load the NSE list. Use 'Custom list' or add universe/<name>.csv to the repo.")
        st.stop()
    bar = st.progress(0, text="Starting…")
    df = run_scan(syms, int(rsi_len), bar)
    bar.empty()
    if not df.empty:
        df.insert(1, "Sector", df["Symbol"].map(meta.set_index("Symbol")["Industry"]))
    st.session_state["res"] = (df, seg, len(syms), datetime.now().strftime("%d %b %Y %H:%M"))

if "res" not in st.session_state:
    st.info("Choose a segment and click **Scan**. First scan of Nifty 500 takes ~30–60 s; repeats are cached.")
    st.stop()

df, seg_name, total, ts = st.session_state["res"]
if df.empty:
    st.warning("No data returned (Yahoo may be rate-limiting). Try again in a minute.")
    st.stop()

sectors = sorted(s for s in df["Sector"].unique() if s)
pick = st.multiselect("Sector filter (optional)", sectors)
mask = (df["Monthly RSI"] >= m_min) & (df["Weekly RSI"] >= w_min) & df["Daily RSI"].between(d_lo, d_hi)
if only_pat:
    mask &= df["Bullish Pattern"] != ""
if pick:
    mask &= df["Sector"].isin(pick)
out = df[mask].sort_values("Monthly RSI", ascending=False).copy()
out["Chart"] = out["Symbol"].map(
    lambda s: "https://in.tradingview.com/chart/?symbol=NSE%3A" + quote(s.replace("&", "_").replace("-", "_")))

c1, c2, c3 = st.columns(3)
c1.metric("Segment", seg_name)
c2.metric("Scanned / matched", f"{len(df)} / {len(out)}", help=f"{total - len(df)} symbols skipped (no data / short history)")
c3.metric("Last scan", ts)

st.dataframe(out, use_container_width=True, hide_index=True,
             column_config={"Chart": st.column_config.LinkColumn("Chart", display_text="Open"),
                            "Close": st.column_config.NumberColumn(format="₹%.2f")})
st.download_button("⬇️ Download CSV", out.drop(columns="Chart").to_csv(index=False),
                   f"scan_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
with st.expander("All scanned stocks (unfiltered)"):
    st.dataframe(df, use_container_width=True, hide_index=True)

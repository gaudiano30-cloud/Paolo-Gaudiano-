import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ==========================================================
# CONFIG
# ==========================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

FILES = {
    "iv": "iv_surface_all.csv",
    "iv_smile": "iv_smile_wide_all.csv",
    "crash": "crash_probabilities_all.csv",
    "rnd": "rnd_mode_all.csv",
    "mnd": "mnd_mode_all.csv",
    "opt": "option_pricing_all.csv",
    "sens": "sensitivities_all.csv",
}


# ==========================================================
# HELPERS (LOAD / CLEAN)
# ==========================================================
def _is_float_like(x: str) -> bool:
    try:
        float(str(x).strip().replace(",", "."))
        return True
    except Exception:
        return False


def _numeric(s):
    return pd.to_numeric(s, errors="coerce")


def _to_datetime_safe(s: pd.Series, dayfirst=False):
    return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)


def _std_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Rende sempre disponibile la colonna 'ticker' (t piccola)."""
    if df is None or df.empty:
        return df
    if "ticker" in df.columns:
        return df
    if "Ticker" in df.columns:
        df = df.rename(columns={"Ticker": "ticker"})
    return df


def _parse_deltaT_to_years(deltaT: pd.Series) -> pd.Series:
    def one(x):
        if pd.isna(x):
            return np.nan
        x = str(x).strip().upper()
        if x.endswith("Y"):
            return float(x[:-1])
        if x.endswith("M"):
            return float(x[:-1]) / 12.0
        if x.endswith("D"):
            return float(x[:-1]) / 365.0
        return np.nan
    return deltaT.map(one)


@st.cache_data(show_spinner=False)
def _read_csv_smart(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_all_data():
    out = {}
    for k, fname in FILES.items():
        path = os.path.join(DATA_DIR, fname)
        df = _read_csv_smart(path)
        df = _std_ticker(df)
        out[k] = df
    return out


def available_tickers(data_dict):
    tickers = set()
    for df in data_dict.values():
        if df is not None and not df.empty and "ticker" in df.columns:
            tickers |= set(df["ticker"].dropna().astype(str).str.upper().unique())
    return sorted(tickers) if tickers else ["(nessun ticker)"]


def by_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "ticker" not in df.columns:
        st.error("Manca la colonna 'ticker' (o 'Ticker') nel CSV.")
        st.stop()
    return df[df["ticker"].astype(str).str.upper() == str(ticker).upper()].copy()


# ==========================================================
# CHARTS – IV SURFACE
# ==========================================================
def _prep_iv_surface(df: pd.DataFrame) -> pd.DataFrame:
    df = _std_ticker(df)
    if df is None or df.empty:
        return df

    out = df.copy()
    out["Expiry_dt"] = _to_datetime_safe(out.get("Expiry"), dayfirst=False)
    out["Data_dt"]   = _to_datetime_safe(out.get("Data"),   dayfirst=True)
    out["Moneyness"] = _numeric(out.get("Moneyness"))
    out["IV"]        = _numeric(out.get("IV"))

    if "Expiry_dt" in out.columns and "Data_dt" in out.columns:
        out["T_years"] = (out["Expiry_dt"] - out["Data_dt"]).dt.days / 365.0
    else:
        out["T_years"] = np.nan

    out = out.dropna(subset=["Moneyness", "IV"])
    return out


def fig_iv_surface_3d(df_surface: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df_surface = _prep_iv_surface(df_surface)
    if df_surface is None or df_surface.empty:
        fig.update_layout(title=f"IV Surface 3D – {ticker} (nessun dato)")
        return fig

    piv = (
        df_surface.pivot_table(index="T_years", columns="Moneyness", values="IV", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    if piv.empty:
        fig.update_layout(title=f"IV Surface 3D – {ticker} (dati non pivotabili)")
        return fig

    fig.add_trace(go.Surface(x=piv.columns.values, y=piv.index.values, z=piv.values, showscale=True))
    fig.update_layout(
        title=f"IV Surface 3D – {ticker}",
        scene=dict(xaxis_title="Moneyness (K/F)", yaxis_title="T (anni)", zaxis_title="IV"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig

# ==========================================================
# CHARTS – CRASH PROB
# ==========================================================
def fig_crash(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Crash Prob – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["DeltaT_years"] = _parse_deltaT_to_years(d.get("DeltaT"))
    d["P_Q"] = _numeric(d.get("P_crash_Q (RND)"))
    d["P_P"] = _numeric(d.get("P_crash_P (MND)"))
    d = d.sort_values("DeltaT_years")

    fig.add_trace(go.Bar(x=d.get("DeltaT"), y=d["P_Q"], name="P_crash_Q (RND)"))
    fig.add_trace(go.Bar(x=d.get("DeltaT"), y=d["P_P"], name="P_crash_P (MND)"))

    fig.update_layout(
        title=f"Crash Probabilities – {ticker}",
        barmode="group",
        xaxis_title="DeltaT",
        yaxis_title="Probabilità",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ==========================================================
# CHARTS – RND/MND MODE
# ==========================================================
def _prep_mode(df: pd.DataFrame) -> pd.DataFrame:
    df = _std_ticker(df)
    if df is None or df.empty:
        return df
    out = df.copy()
    out["DeltaT_years"] = _parse_deltaT_to_years(out.get("DeltaT"))
    out["Mode_S_T"] = _numeric(out.get("Prezzo a scadenza del sottostante(S)"))
    out = out.dropna(subset=["DeltaT_years", "Mode_S_T"]).sort_values("DeltaT_years")
    return out


def fig_rnd(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    d = _prep_mode(df)
    if d is None or d.empty:
        fig.update_layout(title=f"RND Mode – {ticker} (nessun dato)")
        return fig

    if "Modello" in d.columns:
        for model, g in d.groupby("Modello"):
            fig.add_trace(go.Scatter(x=g["DeltaT_years"], y=g["Mode_S_T"], mode="lines+markers", name=str(model)))
    else:
        fig.add_trace(go.Scatter(x=d["DeltaT_years"], y=d["Mode_S_T"], mode="lines+markers", name="Mode"))

    fig.update_layout(
        title=f"RND Mode vs Horizon – {ticker}",
        xaxis_title="T (anni)",
        yaxis_title="Mode S(T)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_mnd(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    d = _prep_mode(df)
    if d is None or d.empty:
        fig.update_layout(title=f"MND Mode – {ticker} (nessun dato)")
        return fig

    if "Modello" in d.columns:
        for model, g in d.groupby("Modello"):
            fig.add_trace(go.Scatter(x=g["DeltaT_years"], y=g["Mode_S_T"], mode="lines+markers", name=str(model)))
    else:
        fig.add_trace(go.Scatter(x=d["DeltaT_years"], y=d["Mode_S_T"], mode="lines+markers", name="Mode"))

    fig.update_layout(
        title=f"MND Mode vs Horizon – {ticker}",
        xaxis_title="T (anni)",
        yaxis_title="Mode S(T)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ==========================================================
# CHARTS – OPTION PRICING
# ==========================================================
def fig_opt_timeseries(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Option Pricing – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Date_dt"] = _to_datetime_safe(d.get("Date"), dayfirst=False)
    d = d.sort_values("Date_dt")
    d["Market_Price"] = _numeric(d.get("Market_Price"))
    d["BS_Price"] = _numeric(d.get("BS_Price"))
    d["Rab_Price"] = _numeric(d.get("Rab_Price"))

    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["Market_Price"], mode="lines", name="Market"))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["BS_Price"], mode="lines", name="BS"))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["Rab_Price"], mode="lines", name="Rabinovitch"))

    fig.update_layout(
        title=f"Option Pricing – Market vs Models – {ticker}",
        xaxis_title="Date",
        yaxis_title="Price",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig



def fig_opt_error_hist(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Pricing Errors – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Diff_BS_Market"] = _numeric(d.get("Diff_BS_Market"))
    d["Diff_Rab_Market"] = _numeric(d.get("Diff_Rab_Market"))

    fig.add_trace(go.Histogram(x=d["Diff_BS_Market"].dropna(), nbinsx=60, name="BS - Market", opacity=0.6))
    fig.add_trace(go.Histogram(x=d["Diff_Rab_Market"].dropna(), nbinsx=60, name="Rab - Market", opacity=0.6))

    fig.update_layout(
        title=f"Pricing Error Distribution – {ticker}",
        barmode="overlay",
        xaxis_title="Model - Market",
        yaxis_title="Count",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ==========================================================
# CHARTS – SENSITIVITIES
# ==========================================================
def fig_sens_diff_pct(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Sensitivities – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["days"] = _numeric(d.get("days"))
    d["Rab_minus_BS_pct"] = _numeric(d.get("Rab_minus_BS_pct"))
    d = d.sort_values("days")

    fig.add_trace(go.Scatter(x=d["days"], y=d["Rab_minus_BS_pct"], mode="lines+markers", name="(Rab-BS)%"))
    fig.update_layout(
        title=f"Sensitivities: (Rabinovitch - BS)% vs days – {ticker}",
        xaxis_title="days",
        yaxis_title="%",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig

# ==========================================================
# STREAMLIT UI
# ==========================================================
st.set_page_config(page_title="Thesis Dashboard", layout="wide")
st.title("Paolo Gaudiano – Dal pricing teorico alla realtà di mercato")

data = load_all_data()
tickers = available_tickers(data)

with st.sidebar:
    st.header("Filtri")
    ticker = st.selectbox("ticker", tickers, index=0)
    tab = st.radio(
        "Sezione",
        ["IV", "Option Pricing", "Sensitivities", "RND", "MND", "Crash Prob"],
        index=0
    )
    st.divider()
    st.caption(f"Data directory: {DATA_DIR}")

if ticker == "(nessun ticker)":
    st.warning("Non ho trovato ticker nei CSV. Controlla che esista la colonna 'ticker' o 'Ticker'.")
    st.stop()

# ---- Render ----
if tab == "IV":
    df_surf = by_ticker(data["iv"], ticker)
    df_smile = by_ticker(data["iv_smile"], ticker)

    st.plotly_chart(fig_iv_surface_3d(df_surf, ticker), use_container_width=True)

elif tab == "Option Pricing":
    df = by_ticker(data["opt"], ticker)
    st.plotly_chart(fig_opt_timeseries(df, ticker), use_container_width=True)
    st.plotly_chart(fig_opt_error_hist(df, ticker), use_container_width=True)

elif tab == "Sensitivities":
    df = by_ticker(data["sens"], ticker)
    st.plotly_chart(fig_sens_diff_pct(df, ticker), use_container_width=True)

elif tab == "RND":
    df = by_ticker(data["rnd"], ticker)
    st.plotly_chart(fig_rnd(df, ticker), use_container_width=True)

elif tab == "MND":
    df = by_ticker(data["mnd"], ticker)
    st.plotly_chart(fig_mnd(df, ticker), use_container_width=True)

elif tab == "Crash Prob":
    df = by_ticker(data["crash"], ticker)
    st.plotly_chart(fig_crash(df, ticker), use_container_width=True)


#st.divider()
#st.caption("Se qualche grafico è vuoto: controlla i nomi delle colonne nei CSV (Date/Expiry/Data ecc.).")

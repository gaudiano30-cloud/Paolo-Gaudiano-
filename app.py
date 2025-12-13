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
# IV (TESI): heatmap + smile (niente 3D/termstructure se non in tesi)
# ==========================================================
def _prep_iv_surface(df: pd.DataFrame) -> pd.DataFrame:
    df = _std_ticker(df)
    if df is None or df.empty:
        return df

    out = df.copy()
    out["Expiry_dt"] = _to_datetime_safe(out.get("Expiry"), dayfirst=False)
    out["Data_dt"] = _to_datetime_safe(out.get("Data"), dayfirst=True)
    out["Moneyness"] = _numeric(out.get("Moneyness"))
    out["IV"] = _numeric(out.get("IV"))

    if "Expiry_dt" in out.columns and "Data_dt" in out.columns:
        out["T_years"] = (out["Expiry_dt"] - out["Data_dt"]).dt.days / 365.0
    else:
        out["T_years"] = np.nan

    out = out.dropna(subset=["Moneyness", "IV"])
    return out


def fig_iv_heatmap(df_surface: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df_surface = _prep_iv_surface(df_surface)
    if df_surface is None or df_surface.empty:
        fig.update_layout(title=f"IV Heatmap – {ticker} (nessun dato)")
        return fig

    piv = (
        df_surface.pivot_table(index="T_years", columns="Moneyness", values="IV", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    if piv.empty:
        fig.update_layout(title=f"IV Heatmap – {ticker} (dati non pivotabili)")
        return fig

    fig.add_trace(go.Heatmap(x=piv.columns.values, y=piv.index.values, z=piv.values, colorbar=dict(title="IV")))
    fig.update_layout(
        title=f"IV Surface (heatmap) – {ticker}",
        xaxis_title="Moneyness (K/F)",
        yaxis_title="T (anni)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_iv_smile(df_smile_wide: pd.DataFrame, ticker: str, max_expiries=6) -> go.Figure:
    fig = go.Figure()
    df_smile_wide = _std_ticker(df_smile_wide)
    if df_smile_wide is None or df_smile_wide.empty:
        fig.update_layout(title=f"IV Smile – {ticker} (nessun dato)")
        return fig

    df = df_smile_wide.copy()
    df["Expiry_dt"] = _to_datetime_safe(df.get("Expiry"), dayfirst=False)

    smile_cols = [c for c in df.columns if _is_float_like(c)]
    if not smile_cols:
        fig.update_layout(title=f"IV Smile – {ticker} (colonne smile non trovate)")
        return fig

    df = df.sort_values("Expiry_dt").head(max_expiries)

    long = df.melt(
        id_vars=["Expiry", "Expiry_dt"],
        value_vars=smile_cols,
        var_name="Moneyness",
        value_name="IV",
    )
    long["Moneyness"] = _numeric(long["Moneyness"])
    long["IV"] = _numeric(long["IV"])
    long = long.dropna(subset=["Moneyness", "IV"]).sort_values(["Expiry_dt", "Moneyness"])

    for exp, g in long.groupby("Expiry"):
        fig.add_trace(go.Scatter(x=g["Moneyness"], y=g["IV"], mode="lines+markers", name=str(exp)))

    fig.update_layout(
        title=f"IV Smile (prime {max_expiries} expiry) – {ticker}",
        xaxis_title="Moneyness (K/F)",
        yaxis_title="IV",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ==========================================================
# CRASH (TESI): Q vs P
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
# RND/MND MODE (TESI): Mode S(T) vs Horizon (da df_mode/df_mode_mnd)
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


def fig_mode(df: pd.DataFrame, ticker: str, title_prefix: str) -> go.Figure:
    fig = go.Figure()
    d = _prep_mode(df)
    if d is None or d.empty:
        fig.update_layout(title=f"{title_prefix} Mode – {ticker} (nessun dato)")
        return fig

    if "Modello" in d.columns:
        for model, g in d.groupby("Modello"):
            fig.add_trace(go.Scatter(x=g["DeltaT_years"], y=g["Mode_S_T"], mode="lines+markers", name=str(model)))
    else:
        fig.add_trace(go.Scatter(x=d["DeltaT_years"], y=d["Mode_S_T"], mode="lines+markers", name="Mode"))

    fig.update_layout(
        title=f"{title_prefix} Mode vs Horizon – {ticker}",
        xaxis_title="T (anni)",
        yaxis_title="Mode S(T)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ==========================================================
# OPTION PRICING (TESI): i grafici esatti di Option_pricing.py
# ==========================================================
def fig_opt_market_vs_models(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Market vs Models – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Date_dt"] = _to_datetime_safe(d.get("Date"), dayfirst=False)
    d = d.sort_values("Date_dt")

    d["Market_Price"] = _numeric(d.get("Market_Price"))
    d["BS_Price"] = _numeric(d.get("BS_Price"))
    d["Rab_Price"] = _numeric(d.get("Rab_Price"))

    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["Market_Price"], mode="lines", name="Market", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["BS_Price"], mode="lines", name="Black–Scholes"))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["Rab_Price"], mode="lines", name="Rabinovitch"))

    fig.update_layout(
        title=f"{ticker} – Market vs BS vs Rabinovitch",
        xaxis_title="Date",
        yaxis_title="Option Price",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_opt_mispricing_pct(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Mispricing % – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Date_dt"] = _to_datetime_safe(d.get("Date"), dayfirst=False)
    d = d.sort_values("Date_dt")

    # in tesi: DiffRel_* (decimali) -> *100
    d["DiffRel_BS_Market"] = _numeric(d.get("DiffRel_BS_Market")) * 100.0
    d["DiffRel_Rab_Market"] = _numeric(d.get("DiffRel_Rab_Market")) * 100.0

    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["DiffRel_BS_Market"], mode="lines", name="(BS-Mkt)/Mkt"))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["DiffRel_Rab_Market"], mode="lines", name="(Rab-Mkt)/Mkt"))

    fig.update_layout(
        title=f"Mispricing Model to Market (%) – {ticker}",
        xaxis_title="Date",
        yaxis_title="Δ (%)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_opt_mispricing_vs_return(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Mispricing vs Return – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Date_dt"] = _to_datetime_safe(d.get("Date"), dayfirst=False)
    d = d.sort_values("Date_dt")

    # tesi: return del sottostante da colonna Price
    d["Price"] = _numeric(d.get("Price"))
    ret = d["Price"].pct_change() * 100.0

    d["Diff_Rab_Market"] = _numeric(d.get("Diff_Rab_Market"))

    fig.add_trace(go.Scatter(x=d["Date_dt"], y=ret, mode="lines", name="Return (%)"))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["Diff_Rab_Market"], mode="lines", name="Delta Rabinovitch ($)"))

    fig.update_layout(
        title=f"Mispricing vs Return – {ticker}",
        xaxis_title="Date",
        yaxis_title="Value",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_opt_rho_sensitivity(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Rho sensitivity – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Date_dt"] = _to_datetime_safe(d.get("Date"), dayfirst=False)
    d = d.sort_values("Date_dt")

    # in tesi: DiffRel_Rab_Market / DiffRel_Rab_rho_-1 / DiffRel_Rab_rho_+1
    cols = ["DiffRel_Rab_Market", "DiffRel_Rab_rho_-1", "DiffRel_Rab_rho_+1"]
    for c in cols:
        if c in d.columns:
            d[c] = _numeric(d[c]) * 100.0

    if not all(c in d.columns for c in cols):
        fig.update_layout(title=f"Rho sensitivity – {ticker} (colonne rho mancanti)")
        return fig

    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["DiffRel_Rab_Market"], mode="lines", name="Rabinovitch (rho stimato)"))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["DiffRel_Rab_rho_-1"], mode="lines", name="Rabinovitch (rho = -1)"))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["DiffRel_Rab_rho_+1"], mode="lines", name="Rabinovitch (rho = +1)"))

    fig.update_layout(
        title=f"Mispricing Sensitivity to ρ – {ticker}",
        xaxis_title="Date",
        yaxis_title="Δ (%)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_opt_bs_vs_lsm(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"BS vs LSM – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Date_dt"] = _to_datetime_safe(d.get("Date"), dayfirst=False)
    d = d.sort_values("Date_dt")
    d["BS_Price"] = _numeric(d.get("BS_Price"))
    d["LSM_Price"] = _numeric(d.get("LSM_Price"))

    if "LSM_Price" not in d.columns or d["LSM_Price"].notna().sum() == 0:
        fig.update_layout(title=f"BS vs LSM – {ticker} (LSM_Price mancante)")
        return fig

    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["BS_Price"], mode="lines", name="BS"))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["LSM_Price"], mode="lines", name="LSM"))

    fig.update_layout(
        title=f"{ticker} – BS vs LSM",
        xaxis_title="Date",
        yaxis_title="Option Price",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_opt_delta_bs_lsm_pct(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Delta BS vs LSM (%) – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Date_dt"] = _to_datetime_safe(d.get("Date"), dayfirst=False)
    d = d.sort_values("Date_dt")
    d["BS_Price"] = _numeric(d.get("BS_Price"))
    d["LSM_Price"] = _numeric(d.get("LSM_Price"))

    if d["LSM_Price"].notna().sum() == 0:
        fig.update_layout(title=f"Delta BS vs LSM (%) – {ticker} (LSM_Price mancante)")
        return fig

    y = (d["BS_Price"] - d["LSM_Price"]) / d["LSM_Price"] * 100.0
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=y, mode="lines", name="(BS-LSM)/LSM %"))

    fig.update_layout(
        title=f"{ticker} – Delta BS vs LSM (%)",
        xaxis_title="Date",
        yaxis_title="Δ (%)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_opt_delta_bs_lsm_abs(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Delta BS vs LSM ($) – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Date_dt"] = _to_datetime_safe(d.get("Date"), dayfirst=False)
    d = d.sort_values("Date_dt")
    d["BS_Price"] = _numeric(d.get("BS_Price"))
    d["LSM_Price"] = _numeric(d.get("LSM_Price"))

    if d["LSM_Price"].notna().sum() == 0:
        fig.update_layout(title=f"Delta BS vs LSM ($) – {ticker} (LSM_Price mancante)")
        return fig

    y = d["BS_Price"] - d["LSM_Price"]
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=y, mode="lines", name="BS - LSM"))

    fig.update_layout(
        title=f"{ticker} – Delta BS vs LSM ($)",
        xaxis_title="Date",
        yaxis_title="Δ ($)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ==========================================================
# SENSITIVITIES (TESI): mispricing % vs T + shares (se presenti)
# ==========================================================
def fig_sens_mispricing_pct(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Sensitivities – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["T"] = _numeric(d.get("T"))
    d["Rab_minus_BS_pct"] = _numeric(d.get("Rab_minus_BS_pct")) * 100.0
    d = d.sort_values("T")

    fig.add_trace(go.Scatter(x=d["T"], y=d["Rab_minus_BS_pct"], mode="lines+markers", name="(Rab-BS)/BS %"))
    fig.update_layout(
        title=f"Rabinovitch vs Black–Scholes (Mispricing %) – {ticker}",
        xaxis_title="Time to Maturity (anni)",
        yaxis_title="Mispricing (%)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_sens_shares(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Sensitivity shares – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["T"] = _numeric(d.get("T"))
    for c in ["share_v", "share_rho"]:
        if c in d.columns:
            d[c] = _numeric(d[c])
    d = d.sort_values("T")

    if not all(c in d.columns for c in ["share_v", "share_rho"]):
        fig.update_layout(title=f"Sensitivity shares – {ticker} (colonne share_* mancanti)")
        return fig

    fig.add_trace(go.Scatter(x=d["T"], y=d["share_v"], mode="lines+markers", name="share_v"))
    fig.add_trace(go.Scatter(x=d["T"], y=d["share_rho"], mode="lines+markers", name="share_rho"))

    fig.update_layout(
        title=f"Sensitivity decomposition shares – {ticker}",
        xaxis_title="Time to Maturity (anni)",
        yaxis_title="share",
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
        ["IV", "Crash Prob", "RND", "MND", "Option Pricing", "Sensitivities"],
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

    st.plotly_chart(fig_iv_heatmap(df_surf, ticker), use_container_width=True)
    st.plotly_chart(fig_iv_smile(df_smile, ticker), use_container_width=True)

elif tab == "Crash Prob":
    df = by_ticker(data["crash"], ticker)
    st.plotly_chart(fig_crash(df, ticker), use_container_width=True)

elif tab == "RND":
    df = by_ticker(data["rnd"], ticker)
    st.plotly_chart(fig_mode(df, ticker, "RND"), use_container_width=True)

elif tab == "MND":
    df = by_ticker(data["mnd"], ticker)
    st.plotly_chart(fig_mode(df, ticker, "MND"), use_container_width=True)

elif tab == "Option Pricing":
    df = by_ticker(data["opt"], ticker)

    st.plotly_chart(fig_opt_market_vs_models(df, ticker), use_container_width=True)
    st.plotly_chart(fig_opt_mispricing_pct(df, ticker), use_container_width=True)
    st.plotly_chart(fig_opt_mispricing_vs_return(df, ticker), use_container_width=True)
    st.plotly_chart(fig_opt_rho_sensitivity(df, ticker), use_container_width=True)

    st.plotly_chart(fig_opt_bs_vs_lsm(df, ticker), use_container_width=True)
    st.plotly_chart(fig_opt_delta_bs_lsm_pct(df, ticker), use_container_width=True)
    st.plotly_chart(fig_opt_delta_bs_lsm_abs(df, ticker), use_container_width=True)

elif tab == "Sensitivities":
    df = by_ticker(data["sens"], ticker)
    st.plotly_chart(fig_sens_mispricing_pct(df, ticker), use_container_width=True)
    st.plotly_chart(fig_sens_shares(df, ticker), use_container_width=True)

st.divider()
st.caption("Se qualche grafico è vuoto: controlla i nomi colonne nei CSV (Date/Price/LSM_Price/DiffRel_* ecc.).")

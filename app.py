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
    "dens": "rnd_mnd_density_all.csv",      # <--- NEW: curve RND/MND (moneyness)
    "opt": "option_pricing_all.csv",
    "sens": "sensitivities_all.csv",
}

SECTIONS_ORDER = ["Pricing", "Sensitivities", "IV", "RND", "MND", "Crash Prob"]  # come tesi (ordine)


# ==========================================================
# HELPERS (LOAD / CLEAN)
# ==========================================================
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


def _has_cols(df: pd.DataFrame, cols) -> bool:
    if df is None or df.empty:
        return False
    return all(c in df.columns for c in cols)


# ==========================================================
# CHARTS – IV (surface + smile)
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
        scene=dict(xaxis_title="Moneyness (K/S)", yaxis_title="T (anni)", zaxis_title="IV"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_iv_surface_heatmap(df_surface: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df_surface = _prep_iv_surface(df_surface)
    if df_surface is None or df_surface.empty:
        fig.update_layout(title=f"IV Surface Heatmap – {ticker} (nessun dato)")
        return fig

    piv = (
        df_surface.pivot_table(index="T_years", columns="Moneyness", values="IV", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    if piv.empty:
        fig.update_layout(title=f"IV Surface Heatmap – {ticker} (dati non pivotabili)")
        return fig

    fig.add_trace(go.Heatmap(x=piv.columns.values, y=piv.index.values, z=piv.values, colorbar=dict(title="IV")))
    fig.update_layout(
        title=f"IV Surface (Heatmap) – {ticker}",
        xaxis_title="Moneyness (K/S)",
        yaxis_title="T (anni)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def _prep_iv_smile_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Atteso: una colonna per orizzonte (es 1M/3M/6M/1Y) e righe = moneyness (o strike).
    Se esistono 'Moneyness'/'moneyness' come colonna, la usa; altrimenti cerca la prima colonna non-numerica.
    """
    df = _std_ticker(df)
    if df is None or df.empty:
        return df

    d = df.copy()
    # standardizza moneyness col
    if "Moneyness" in d.columns:
        d["Moneyness"] = _numeric(d["Moneyness"])
        xcol = "Moneyness"
    elif "moneyness" in d.columns:
        d["moneyness"] = _numeric(d["moneyness"])
        xcol = "moneyness"
    else:
        # prova: prima colonna che sembra asse x
        xcol = d.columns[0]
        d[xcol] = _numeric(d[xcol])

    # pulizia
    d = d.dropna(subset=[xcol])
    return d, xcol


def fig_iv_smile(df_smile_wide: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df_smile_wide = _std_ticker(df_smile_wide)
    if df_smile_wide is None or df_smile_wide.empty:
        fig.update_layout(title=f"IV Smile – {ticker} (nessun dato)")
        return fig

    d, xcol = _prep_iv_smile_wide(df_smile_wide)
    # colonne candidate = tutte tranne ticker e x
    cols = [c for c in d.columns if c not in ["ticker", "Ticker", xcol]]
    if not cols:
        fig.update_layout(title=f"IV Smile – {ticker} (nessuna colonna tenor)")
        return fig

    d = d.sort_values(xcol)

    for c in cols:
        y = _numeric(d[c])
        if y.notna().sum() == 0:
            continue
        fig.add_trace(go.Scatter(x=d[xcol], y=y, mode="lines", name=str(c)))

    fig.update_layout(
        title=f"IV Smile (by tenor) – {ticker}",
        xaxis_title=xcol,
        yaxis_title="IV",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ==========================================================
# CHARTS – PRICING
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


def fig_opt_error_timeseries(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Pricing Errors (time) – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Date_dt"] = _to_datetime_safe(d.get("Date"), dayfirst=False)
    d = d.sort_values("Date_dt")
    d["Diff_BS_Market"] = _numeric(d.get("Diff_BS_Market"))
    d["Diff_Rab_Market"] = _numeric(d.get("Diff_Rab_Market"))

    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["Diff_BS_Market"], mode="lines", name="BS - Market"))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["Diff_Rab_Market"], mode="lines", name="Rab - Market"))

    fig.update_layout(
        title=f"Pricing Errors over Time – {ticker}",
        xaxis_title="Date",
        yaxis_title="Model - Market",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_opt_error_hist(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Pricing Errors (dist) – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Diff_BS_Market"] = _numeric(d.get("Diff_BS_Market"))
    d["Diff_Rab_Market"] = _numeric(d.get("Diff_Rab_Market"))

    # nbins e overlay come “tesi style”
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


def fig_pricing_scatter_market_vs_models(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Market vs Models (scatter) – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Market_Price"] = _numeric(d.get("Market_Price"))
    d["BS_Price"] = _numeric(d.get("BS_Price"))
    d["Rab_Price"] = _numeric(d.get("Rab_Price"))

    m = d.dropna(subset=["Market_Price"])
    if m.empty:
        fig.update_layout(title=f"Market vs Models (scatter) – {ticker} (nessun dato)")
        return fig

    if m["BS_Price"].notna().sum() > 0:
        fig.add_trace(go.Scatter(x=m["Market_Price"], y=m["BS_Price"], mode="markers", name="BS"))
    if m["Rab_Price"].notna().sum() > 0:
        fig.add_trace(go.Scatter(x=m["Market_Price"], y=m["Rab_Price"], mode="markers", name="Rabinovitch"))

    # diagonale y=x
    xmin = np.nanmin(m["Market_Price"].values)
    xmax = np.nanmax(m["Market_Price"].values)
    fig.add_trace(go.Scatter(x=[xmin, xmax], y=[xmin, xmax], mode="lines", name="y=x"))

    fig.update_layout(
        title=f"Market vs Models – Scatter – {ticker}",
        xaxis_title="Market Price",
        yaxis_title="Model Price",
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


def fig_sens_greek_compare(df: pd.DataFrame, ticker: str, greek: str) -> go.Figure:
    """
    Grafico aggiuntivo "mancante": confronto greca BS vs Rab se le colonne esistono.
    Prova nomi tipo:
      Delta_BS / Delta_Rab, Gamma_BS / Gamma_Rab, Vega_BS / Vega_Rab, Theta_BS / Theta_Rab
    """
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"{greek} – {ticker} (nessun dato)")
        return fig

    bs_col = f"{greek}_BS"
    rb_col = f"{greek}_Rab"
    if not _has_cols(df, ["days", bs_col, rb_col]):
        fig.update_layout(title=f"{greek} (BS vs Rab) – {ticker} (colonne mancanti)")
        return fig

    d = df.copy()
    d["days"] = _numeric(d["days"])
    d[bs_col] = _numeric(d[bs_col])
    d[rb_col] = _numeric(d[rb_col])
    d = d.sort_values("days")

    fig.add_trace(go.Scatter(x=d["days"], y=d[bs_col], mode="lines", name=bs_col))
    fig.add_trace(go.Scatter(x=d["days"], y=d[rb_col], mode="lines", name=rb_col))

    fig.update_layout(
        title=f"{greek} – BS vs Rabinovitch – {ticker}",
        xaxis_title="days",
        yaxis_title=greek,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ==========================================================
# CHARTS – RND/MND (MODE + DENSITY CURVES LIKE TESI)
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


def fig_mode_vs_horizon(df: pd.DataFrame, ticker: str, title_prefix: str) -> go.Figure:
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


def fig_density_moneyness(df_dens: pd.DataFrame, ticker: str, measure: str, show_models=("BS", "RABINOVITCH")) -> go.Figure:
    """
    Grafico "uguale alla tesi": densità in funzione del Moneyness (K/S).
    measure: "RND" oppure "MND"
    """
    fig = go.Figure()
    df_dens = _std_ticker(df_dens)
    if df_dens is None or df_dens.empty:
        fig.update_layout(title=f"{measure} Density (Moneyness) – {ticker} (nessun dato)")
        return fig

    d = by_ticker(df_dens, ticker)

    # standardizza
    if "Measure" in d.columns:
        d["Measure"] = d["Measure"].astype(str).str.upper().str.strip()
        d = d[d["Measure"] == str(measure).upper()]
    else:
        fig.update_layout(title=f"{measure} Density (Moneyness) – {ticker} (colonna Measure mancante)")
        return fig

    # colonne attese
    if not _has_cols(d, ["Moneyness", "Density", "DeltaT", "Modello"]):
        fig.update_layout(title=f"{measure} Density (Moneyness) – {ticker} (colonne mancanti)")
        return fig

    d["Moneyness"] = _numeric(d["Moneyness"])
    d["Density"] = _numeric(d["Density"])
    d = d.dropna(subset=["Moneyness", "Density"])

    if d.empty:
        fig.update_layout(title=f"{measure} Density (Moneyness) – {ticker} (nessun dato)")
        return fig

    # filtro modelli (se vuoi nella UI)
    d["Modello"] = d["Modello"].astype(str)
    keep = []
    for m in show_models:
        keep.append(m.lower())
    d = d[d["Modello"].str.lower().isin(keep)]

    # stile “tesi”: RND tratteggiata, MND piena
    dash = "dash" if str(measure).upper() == "RND" else "solid"

    # ordine tenori (se presenti)
    tenor_order = {"1M": 1, "3M": 2, "6M": 3, "1Y": 4}
    if "DeltaT" in d.columns:
        d["_tenor_rank"] = d["DeltaT"].astype(str).map(lambda x: tenor_order.get(x.strip().upper(), 99))
    else:
        d["_tenor_rank"] = 99

    # group
    for (modello, deltaT), g in d.groupby(["Modello", "DeltaT"], dropna=False):
        g = g.sort_values("Moneyness")
        name = f"{measure} {modello} {deltaT}"
        fig.add_trace(go.Scatter(
            x=g["Moneyness"],
            y=g["Density"],
            mode="lines",
            name=name,
            line=dict(dash=dash),
            hovertemplate="Moneyness=%{x:.4f}<br>Density=%{y:.6g}<extra></extra>",
        ))

    fig.update_layout(
        title=f"{ticker} – {measure} (Moneyness K/S)",
        xaxis_title="Moneyness (K / S)",
        yaxis_title="Density",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
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
# STREAMLIT UI (controls RIGHT)
# ==========================================================
st.set_page_config(page_title="Thesis Dashboard", layout="wide")

data = load_all_data()
tickers = available_tickers(data)

# Header
st.title("Paolo Gaudiano – Dal pricing teorico alla realtà di mercato")

# Layout: main left, controls right (come “elenco a destra”)
main_col, right_col = st.columns([4.6, 1.4], vertical_alignment="top")

with right_col:
    st.markdown("### Filtri")
    ticker = st.selectbox("Ticker", tickers, index=0)
    section = st.radio("Sezione", SECTIONS_ORDER, index=0)
    st.divider()

    # Toggle modelli per RND/MND density
    show_bs = st.checkbox("Mostra BS", value=True)
    show_rab = st.checkbox("Mostra Rabinovitch", value=True)
    chosen_models = []
    if show_bs:
        chosen_models.append("BS")
    if show_rab:
        chosen_models.append("Rabinovitch")
    if not chosen_models:
        chosen_models = ["BS", "Rabinovitch"]

    st.divider()
    st.caption(f"Data dir: {DATA_DIR}")

if ticker == "(nessun ticker)":
    with main_col:
        st.warning("Non ho trovato ticker nei CSV. Controlla che esista la colonna 'ticker' o 'Ticker'.")
    st.stop()


# ==========================================================
# RENDER (ordine come tesi)
# ==========================================================
with main_col:
    # ------------------- PRICING -------------------
    if section == "Pricing":
        df = by_ticker(data["opt"], ticker)

        st.plotly_chart(fig_opt_timeseries(df, ticker), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_opt_error_timeseries(df, ticker), use_container_width=True)
        with c2:
            st.plotly_chart(fig_pricing_scatter_market_vs_models(df, ticker), use_container_width=True)

        st.plotly_chart(fig_opt_error_hist(df, ticker), use_container_width=True)

    # ------------------- SENSITIVITIES -------------------
    elif section == "Sensitivities":
        df = by_ticker(data["sens"], ticker)
        st.plotly_chart(fig_sens_diff_pct(df, ticker), use_container_width=True)

        # “grafici mancanti” (se nel CSV ci sono le colonne)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_sens_greek_compare(df, ticker, "Delta"), use_container_width=True)
        with c2:
            st.plotly_chart(fig_sens_greek_compare(df, ticker, "Vega"), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(fig_sens_greek_compare(df, ticker, "Gamma"), use_container_width=True)
        with c4:
            st.plotly_chart(fig_sens_greek_compare(df, ticker, "Theta"), use_container_width=True)

        st.caption("Se i grafici Delta/Vega/Gamma/Theta risultano vuoti: nel CSV mancano le colonne tipo Delta_BS / Delta_Rab ecc.")

    # ------------------- IV -------------------
    elif section == "IV":
        df_surf = by_ticker(data["iv"], ticker)
        df_smile = by_ticker(data["iv_smile"], ticker)

        st.plotly_chart(fig_iv_surface_3d(df_surf, ticker), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_iv_surface_heatmap(df_surf, ticker), use_container_width=True)
        with c2:
            st.plotly_chart(fig_iv_smile(df_smile, ticker), use_container_width=True)

    # ------------------- RND -------------------
    elif section == "RND":
        # 1) densità “come tesi” (curve vs moneyness)
        df_dens = data.get("dens", pd.DataFrame())
        st.plotly_chart(fig_density_moneyness(df_dens, ticker, measure="RND", show_models=tuple(chosen_models)),
                        use_container_width=True)

        # 2) mode vs horizon (se vuoi tenerlo)
        df_mode = by_ticker(data["rnd"], ticker)
        st.plotly_chart(fig_mode_vs_horizon(df_mode, ticker, title_prefix="RND"), use_container_width=True)

    # ------------------- MND -------------------
    elif section == "MND":
        df_dens = data.get("dens", pd.DataFrame())
        st.plotly_chart(fig_density_moneyness(df_dens, ticker, measure="MND", show_models=tuple(chosen_models)),
                        use_container_width=True)

        df_mode = by_ticker(data["mnd"], ticker)
        st.plotly_chart(fig_mode_vs_horizon(df_mode, ticker, title_prefix="MND"), use_container_width=True)

    # ------------------- CRASH PROB -------------------
    elif section == "Crash Prob":
        df = by_ticker(data["crash"], ticker)
        st.plotly_chart(fig_crash(df, ticker), use_container_width=True)

    st.divider()
    st.caption("Se qualche grafico è vuoto: controlla i nomi delle colonne nei CSV e che il file rnd_mnd_density_all.csv sia in /data.")

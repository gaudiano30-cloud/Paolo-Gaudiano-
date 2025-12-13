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
    "iv_smile": "iv_smile_wide_all.csv",          # (non usato ora, ma lo lasciamo)
    "crash": "crash_probabilities_all.csv",
    "dens": "rnd_mnd_density_all.csv",            # <-- deve stare in /data
    "opt": "option_pricing_all.csv",
    "sens": "sensitivities_all.csv",
}

SECTIONS = ["Option pricing", "IV", "RND", "MND", "Crash Prob"]


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
        return df.rename(columns={"Ticker": "ticker"})
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
# CHARTS – IV SURFACE (ONLY 3D)
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
# CHARTS – OPTION PRICING (core + mispricing% + sensitivities + rho)
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
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["BS_Price"], mode="lines", name="Black–Scholes"))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["Rab_Price"], mode="lines", name="Rabinovitch"))

    fig.update_layout(
        title=f"{ticker} – Option Price (Market vs Models)",
        xaxis_title="Date",
        yaxis_title="Option Price",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_mispricing_pct_timeseries(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Mispricing Model to Market in % nel tempo:
    (BS-Mkt)/Mkt e (Rab-Mkt)/Mkt
    """
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Mispricing Model to Market – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Date_dt"] = _to_datetime_safe(d.get("Date"), dayfirst=False)
    d = d.sort_values("Date_dt")

    d["mkt"] = _numeric(d.get("Market_Price"))
    d["bs"] = _numeric(d.get("BS_Price"))
    d["rab"] = _numeric(d.get("Rab_Price"))

    d = d.dropna(subset=["Date_dt", "mkt", "bs", "rab"])
    d = d[d["mkt"] != 0]

    if d.empty:
        fig.update_layout(title=f"Mispricing Model to Market – {ticker} (nessun dato)")
        return fig

    d["mis_bs_pct"] = (d["bs"] - d["mkt"]) / d["mkt"] * 100.0
    d["mis_rab_pct"] = (d["rab"] - d["mkt"]) / d["mkt"] * 100.0

    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["mis_bs_pct"], mode="lines", name="(BS-Mkt)/Mkt"))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["mis_rab_pct"], mode="lines", name="(Rab-Mkt)/Mkt"))

    fig.add_hline(y=0, line_dash="dash")

    fig.update_layout(
        title=f"Mispricing Model to Market ({ticker})",
        xaxis_title="Date",
        yaxis_title="Δ (%)",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_sens_diff_pct(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Grafico sensitivities "iniziale": (Rab - BS)% vs days (dal sensitivities_all.csv)
    """
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Sensitivities (Rab-BS)% – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["days"] = _numeric(d.get("days"))
    d["Rab_minus_BS_pct"] = _numeric(d.get("Rab_minus_BS_pct"))
    d = d.dropna(subset=["days", "Rab_minus_BS_pct"]).sort_values("days")

    if d.empty:
        fig.update_layout(title=f"Sensitivities (Rab-BS)% – {ticker} (nessun dato)")
        return fig

    fig.add_trace(go.Scatter(x=d["days"], y=d["Rab_minus_BS_pct"], mode="lines+markers", name="(Rab-BS)%"))
    fig.update_layout(
        title=f"Sensitivities: (Rabinovitch - BS)% vs days – {ticker}",
        xaxis_title="days",
        yaxis_title="%",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_sensitivity_rho(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Grafico 'Pricing Sensitivity to rho' (come tesi).
    Cerca colonne rho estreme in option_pricing_all.csv.
    """
    fig = go.Figure()
    df = _std_ticker(df)
    if df is None or df.empty:
        fig.update_layout(title=f"Pricing Sensitivity to rho – {ticker} (nessun dato)")
        return fig

    d = df.copy()
    d["Date_dt"] = _to_datetime_safe(d.get("Date"), dayfirst=False)
    d = d.sort_values("Date_dt")

    d["Market_Price"] = _numeric(d.get("Market_Price"))
    d["BS_Price"] = _numeric(d.get("BS_Price"))
    d["Rab_Price"] = _numeric(d.get("Rab_Price"))

    cols = {str(c).lower().replace(" ", ""): c for c in d.columns}
    rho_minus = None
    rho_plus = None

    # pattern comuni
    cand_minus = ["rab_price_rho_-1", "rab_price_rho=-1", "rab_price_rho_m1", "rab_price_rho_-1.0"]
    cand_plus = ["rab_price_rho_+1", "rab_price_rho=+1", "rab_price_rho_p1", "rab_price_rho_+1.0", "rab_price_rho_1"]

    for key in cand_minus:
        if key in cols:
            rho_minus = cols[key]
            break
    for key in cand_plus:
        if key in cols:
            rho_plus = cols[key]
            break

    # fallback fuzzy
    if rho_minus is None:
        for c in d.columns:
            lc = str(c).lower().replace(" ", "")
            if "rho" in lc and ("-1" in lc or "m1" in lc) and "rab" in lc:
                rho_minus = c
                break
    if rho_plus is None:
        for c in d.columns:
            lc = str(c).lower().replace(" ", "")
            if "rho" in lc and ("+1" in lc or "p1" in lc or lc.endswith("rho1")) and "rab" in lc:
                rho_plus = c
                break

    fig.add_trace(go.Scatter(
        x=d["Date_dt"], y=d["Market_Price"],
        mode="lines", name="Market Price",
        line=dict(dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=d["Date_dt"], y=d["BS_Price"],
        mode="lines", name="Black–Scholes"
    ))
    fig.add_trace(go.Scatter(
        x=d["Date_dt"], y=d["Rab_Price"],
        mode="lines", name="Rabinovitch (rho est.)"
    ))

    if rho_minus is not None:
        fig.add_trace(go.Scatter(
            x=d["Date_dt"], y=_numeric(d[rho_minus]),
            mode="lines", name="Rabinovitch (rho = -1)"
        ))
    if rho_plus is not None:
        fig.add_trace(go.Scatter(
            x=d["Date_dt"], y=_numeric(d[rho_plus]),
            mode="lines", name="Rabinovitch (rho = +1)"
        ))

    fig.update_layout(
        title=f"{ticker} – Pricing Sensitivity to rho",
        xaxis_title="Date",
        yaxis_title="Option Price",
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
    return fig_opt_error_hist


# ==========================================================
# CHARTS – RND / MND DENSITIES (BELL CURVES: moneyness + strike)
#   >>> MODIFICATO SOLO QUI per supportare il tuo CSV merged
# ==========================================================

def _colmap_case_insensitive(df: pd.DataFrame) -> dict:
    """mappa lower(col)->col originale"""
    return {str(c).strip().lower(): c for c in (df.columns if df is not None else [])}


def _get_col(df: pd.DataFrame, *candidates: str):
    """ritorna la prima colonna presente (case-insensitive) tra candidates"""
    cmap = _colmap_case_insensitive(df)
    for c in candidates:
        key = str(c).strip().lower()
        if key in cmap:
            return cmap[key]
    return None


def _normalize_model_name(x: str) -> str:
    s = str(x).strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    return s


def _is_bs(model_str: str) -> bool:
    s = _normalize_model_name(model_str)
    return ("bs" in s) or ("black" in s and "scholes" in s)


def _is_rab(model_str: str) -> bool:
    s = _normalize_model_name(model_str)
    return ("rab" in s) or ("rabin" in s)


def _pick_latest_date(d: pd.DataFrame) -> pd.DataFrame:
    """Se c'è una colonna date, prende l'ultima data disponibile (come nei grafici tesi)."""
    if d is None or d.empty:
        return d
    c_date = _get_col(d, "date", "Date", "Data")
    if c_date is None:
        return d
    dd = d.copy()
    # prova sia dayfirst True che False perché nel tuo file può essere 1/23/2024
    dt = pd.to_datetime(dd[c_date], errors="coerce", dayfirst=False)
    dt2 = pd.to_datetime(dd[c_date], errors="coerce", dayfirst=True)
    dd["_date_dt"] = dt
    dd.loc[dd["_date_dt"].isna(), "_date_dt"] = dt2[dd["_date_dt"].isna()]
    dd = dd.dropna(subset=["_date_dt"])
    if dd.empty:
        return d
    last = dd["_date_dt"].max()
    return dd[dd["_date_dt"] == last].copy()


def _spot_series_from_opt(df_opt: pd.DataFrame, ticker: str):
    """
    Ritorna una Series indicizzata per data con lo spot/underlying per ricostruire Strike = Moneyness * S.
    Cerca colonne comuni in option_pricing_all.csv.
    """
    if df_opt is None or df_opt.empty:
        return None

    d = by_ticker(_std_ticker(df_opt), ticker)
    if d is None or d.empty:
        return None

    c_date = _get_col(d, "date", "Date", "Data")
    if c_date is None:
        return None

    # candidati spot (in ordine)
    c_spot = _get_col(
        d,
        "S", "Spot", "spot", "Underlying", "Underlying_Price", "UnderlyingPrice",
        "Stock_Price", "StockPrice", "Price", "Underlying price"
    )
    if c_spot is None:
        return None

    dd = d.copy()
    dd["_date_dt"] = pd.to_datetime(dd[c_date], errors="coerce", dayfirst=False)
    dd2 = pd.to_datetime(dd[c_date], errors="coerce", dayfirst=True)
    dd.loc[dd["_date_dt"].isna(), "_date_dt"] = dd2[dd["_date_dt"].isna()]
    dd["_spot"] = _numeric(dd[c_spot])
    dd = dd.dropna(subset=["_date_dt", "_spot"])
    if dd.empty:
        return None

    # se ci sono duplicati sulla stessa data, prendo il primo
    dd = dd.sort_values("_date_dt").drop_duplicates("_date_dt", keep="last")
    s = dd.set_index("_date_dt")["_spot"]
    return s


def _prep_density(df_dens: pd.DataFrame, df_opt: pd.DataFrame, ticker: str, measure: str) -> pd.DataFrame:
    """
    Supporta file merged come il tuo:
      ticker, date, DeltaT, Modello, Measure, Moneyness, Density
    e ricostruisce Strike se manca (Strike = Moneyness * Spot preso da option_pricing_all.csv).
    """
    df_dens = _std_ticker(df_dens)
    if df_dens is None or df_dens.empty:
        return pd.DataFrame()

    d = by_ticker(df_dens, ticker)
    if d is None or d.empty:
        return pd.DataFrame()

    c_measure = _get_col(d, "Measure", "measure")
    if c_measure is None:
        return pd.DataFrame()

    d = d.copy()
    d[c_measure] = d[c_measure].astype(str).str.upper().str.strip()
    d = d[d[c_measure] == str(measure).upper()]
    if d.empty:
        return pd.DataFrame()

    # normalizza colonne principali
    c_model = _get_col(d, "Modello", "modello", "Model", "model")
    c_deltaT = _get_col(d, "DeltaT", "deltat", "tenor", "Tenor", "T")
    c_mny = _get_col(d, "Moneyness", "moneyness")
    c_density = _get_col(d, "Density", "density")

    if c_model is None or c_deltaT is None or c_mny is None or c_density is None:
        return pd.DataFrame()

    d["Modello"] = d[c_model].astype(str).str.strip()
    d["DeltaT"] = d[c_deltaT].astype(str).str.strip()
    d["Moneyness"] = _numeric(d[c_mny])
    d["Density"] = _numeric(d[c_density])

    # date (serve per “ultima data” + ricostruzione strike)
    c_date = _get_col(d, "date", "Date", "Data")
    if c_date is not None:
        d["_date_dt"] = pd.to_datetime(d[c_date], errors="coerce", dayfirst=False)
        d2 = pd.to_datetime(d[c_date], errors="coerce", dayfirst=True)
        d.loc[d["_date_dt"].isna(), "_date_dt"] = d2[d["_date_dt"].isna()]
    else:
        d["_date_dt"] = pd.NaT

    # strike: se manca, prova a costruirlo da option_pricing (Strike = Moneyness * Spot)
    c_strike = _get_col(d, "Strike", "strike", "K", "k")
    if c_strike is not None:
        d["Strike"] = _numeric(d[c_strike])
    else:
        d["Strike"] = np.nan
        spot_series = _spot_series_from_opt(df_opt, ticker)
        if spot_series is not None and c_date is not None:
            # allinea spot per data
            tmp = d.dropna(subset=["_date_dt", "Moneyness"]).copy()
            if not tmp.empty:
                # merge spot
                tmp["_spot"] = tmp["_date_dt"].map(spot_series)
                tmp.loc[tmp["_spot"].isna(), "_spot"] = np.nan
                d.loc[tmp.index, "Strike"] = tmp["Moneyness"] * tmp["_spot"]

    # pulizia finale
    d = d.dropna(subset=["Moneyness", "Density"])
    return d


def fig_density_curve(df_dens: pd.DataFrame, df_opt: pd.DataFrame, ticker: str, measure: str, xmode: str, model_choice: str) -> go.Figure:
    """
    xmode: "Moneyness" oppure "Strike"
    model_choice: "Rabinovitch" oppure "BS"
    """
    fig = go.Figure()
    d = _prep_density(df_dens, df_opt, ticker, measure)

    if d.empty:
        fig.update_layout(title=f"{ticker} – {measure} Density ({xmode}) (nessun dato)")
        return fig

    # Prendi l'ULTIMA data disponibile (replica i grafici “exp …” che hai in tesi)
    d = _pick_latest_date(d)

    # filtra modello in modo robusto
    choice = str(model_choice).strip().lower()
    if choice == "bs":
        d = d[d["Modello"].map(_is_bs)]
        title_model = "BS"
    else:
        d = d[d["Modello"].map(_is_rab)]
        title_model = "Rabinovitch"

    if d.empty:
        fig.update_layout(title=f"{ticker} – {measure} Density ({xmode}) – {title_model} (nessun dato)")
        return fig

    # asse X
    xcol = "Moneyness" if xmode.lower().startswith("m") else "Strike"
    if xcol not in d.columns:
        fig.update_layout(title=f"{ticker} – {measure} Density ({xmode}) (colonna {xcol} mancante)")
        return fig

    # se Strike è tutto NaN, avvisa chiaramente
    if xcol == "Strike" and d["Strike"].isna().all():
        fig.update_layout(
            title=f"{ticker} – {measure} Density (Strike) – {title_model} (Strike non ricostruibile)",
            xaxis_title="Strike (K)",
            yaxis_title="Densità",
            margin=dict(l=0, r=0, t=50, b=0),
        )
        return fig

    d = d.dropna(subset=[xcol, "Density"])
    if d.empty:
        fig.update_layout(title=f"{ticker} – {measure} Density ({xmode}) – {title_model} (nessun dato)")
        return fig

    # ordine tenor “tesi”
    tenor_order = {"1M": 1, "3M": 2, "6M": 3, "1Y": 4}
    d["_ten"] = d["DeltaT"].astype(str).str.upper().map(lambda x: tenor_order.get(x, 99))

    # tracce per DeltaT (campane)
    for tenor, g in d.sort_values(["_ten", xcol]).groupby("DeltaT", dropna=False):
        g = g.sort_values(xcol)
        fig.add_trace(go.Scatter(
            x=g[xcol],
            y=g["Density"],
            mode="lines",
            name=str(tenor)
        ))

    # titolo con data (se disponibile)
    if "_date_dt" in d.columns and pd.notna(d["_date_dt"]).any():
        dt_show = pd.to_datetime(d["_date_dt"].dropna().iloc[0]).date()
        title_date = f" – {dt_show}"
    else:
        title_date = ""

    fig.update_layout(
        title=f"{ticker} – {measure} Density ({xmode}) – {title_model}{title_date}",
        xaxis_title="Moneyness (K/S)" if xcol == "Moneyness" else "Strike (K)",
        yaxis_title="Densità",
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ==========================================================
# STREAMLIT UI (MENU A SINISTRA)
# ==========================================================
st.set_page_config(page_title="Thesis Dashboard", layout="wide")
st.title("Paolo Gaudiano – Dal pricing teorico alla realtà di mercato")

data = load_all_data()
tickers = available_tickers(data)

with st.sidebar:
    st.header("Filtri")
    ticker = st.selectbox("Ticker", tickers, index=0)
    section = st.radio("Sezione", SECTIONS, index=0)

    # Mostra la scelta modello SOLO dentro RND/MND
    model_choice = "Rabinovitch"
    xmode = "Moneyness"
    if section in ["RND", "MND"]:
        st.divider()
        st.subheader("RND/MND options")
        model_choice = st.radio("Modello", ["Rabinovitch", "BS"], index=0, horizontal=True)
        xmode = st.radio("Asse X", ["Moneyness", "Strike"], index=0, horizontal=True)

    st.divider()
    st.caption(f"Data directory: {DATA_DIR}")

if ticker == "(nessun ticker)":
    st.warning("Non ho trovato ticker nei CSV. Controlla che esista la colonna 'ticker' o 'Ticker'.")
    st.stop()


# ==========================================================
# RENDER (TUTTO UNO SOTTO L'ALTRO)
# ==========================================================
if section == "Option pricing":
    df_opt = by_ticker(data["opt"], ticker)
    df_sens = by_ticker(data["sens"], ticker)

    # 1) Market vs BS vs Rab (time series)
    st.plotly_chart(fig_opt_timeseries(df_opt, ticker), use_container_width=True)

    # 2) Mispricing % nel tempo (come immagine)
    st.plotly_chart(fig_mispricing_pct_timeseries(df_opt, ticker), use_container_width=True)

    # 3) Sensitivities (Rab-BS)% vs days
    st.plotly_chart(fig_sens_diff_pct(df_sens, ticker), use_container_width=True)

    # 4) Sensitivity to rho
    st.plotly_chart(fig_sensitivity_rho(df_opt, ticker), use_container_width=True)

    # 5) Error distribution (istogramma)
    st.plotly_chart(fig_opt_error_hist(df_opt, ticker), use_container_width=True)

elif section == "IV":
    df_surf = by_ticker(data["iv"], ticker)
    st.plotly_chart(fig_iv_surface_3d(df_surf, ticker), use_container_width=True)

elif section == "RND":
    df_dens = data.get("dens", pd.DataFrame())
    df_opt_all = data.get("opt", pd.DataFrame())
    st.plotly_chart(
        fig_density_curve(df_dens, df_opt_all, ticker, measure="RND", xmode=xmode, model_choice=model_choice),
        use_container_width=True
    )

elif section == "MND":
    df_dens = data.get("dens", pd.DataFrame())
    df_opt_all = data.get("opt", pd.DataFrame())
    st.plotly_chart(
        fig_density_curve(df_dens, df_opt_all, ticker, measure="MND", xmode=xmode, model_choice=model_choice),
        use_container_width=True
    )

elif section == "Crash Prob":
    df = by_ticker(data["crash"], ticker)
    st.plotly_chart(fig_crash(df, ticker), use_container_width=True)

st.caption(
    "RND/MND: usa rnd_mnd_density_all.csv (ticker, date, DeltaT, Modello, Measure, Moneyness, Density). "
    "Se scegli 'Strike' e nel CSV non c’è Strike/K, il grafico prova a ricostruirlo da option_pricing_all.csv "
    "con Strike = Moneyness * Spot (serve una colonna spot/underlying nel pricing)."
)

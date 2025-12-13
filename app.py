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


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip + rimuove BOM dai nomi colonna."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def _std_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Rende sempre disponibile la colonna 'ticker' (t piccola)."""
    if df is None or df.empty:
        return df
    df = _clean_columns(df)
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


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# MODIFICA RICHIESTA: anni float -> label standard (1Y/6M/3M/1M)
def _years_to_tenor_label(y: float) -> str:
    """
    Converte anni (float) in label standard:
    1Y, 6M, 3M, 1M
    """
    if pd.isna(y):
        return ""
    if y >= 0.9:
        return "1Y"
    if 0.45 <= y < 0.9:
        return "6M"
    if 0.20 <= y < 0.45:
        return "3M"
    if 0.05 <= y < 0.20:
        return "1M"
    return ""
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def _normalize_density_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza le colonne del CSV densità a questi nomi standard:
    ticker, date, DeltaT, Modello, Measure, Moneyness, Density
    """
    if df is None or df.empty:
        return df
    df = _clean_columns(df).copy()

    cmap = {str(c).strip().lower().replace(" ", ""): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            k = str(c).strip().lower().replace(" ", "")
            if k in cmap:
                return cmap[k]
        return None

    ren = {}

    c_ticker = pick("ticker", "Ticker")
    if c_ticker and c_ticker != "ticker":
        ren[c_ticker] = "ticker"

    c_date = pick("date", "Date", "Data")
    if c_date and c_date != "date":
        ren[c_date] = "date"

    c_dt = pick("deltat", "DeltaT", "tenor", "Tenor")
    if c_dt and c_dt != "DeltaT":
        ren[c_dt] = "DeltaT"

    c_model = pick("modello", "Modello", "model", "Model")
    if c_model and c_model != "Modello":
        ren[c_model] = "Modello"

    c_meas = pick("measure", "Measure")
    if c_meas and c_meas != "Measure":
        ren[c_meas] = "Measure"

    c_mny = pick("moneyness", "Moneyness")
    if c_mny and c_mny != "Moneyness":
        ren[c_mny] = "Moneyness"

    c_den = pick("density", "Density")
    if c_den and c_den != "Density":
        ren[c_den] = "Density"

    if ren:
        df = df.rename(columns=ren)

    return _std_ticker(df)


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
        df = _clean_columns(df)
        df = _std_ticker(df)

        if k == "dens":
            df = _normalize_density_columns(df)

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
# >>> AGGIUNTA: TABELLE "STRIKE PIÙ PROBABILE" (DA TESI PDF)
# ==========================================================
RND_MODE_TABLES = {
    "AAPL": pd.DataFrame([
        {"Modello-ΔT": "Rab-1Y", "Strike più probabile (K)": 209.40, "Moneyness": 1.0729, "S a scadenza": 229.98, "Δ(S–K)": 20.58, "Δ(%)": "9.83%",  "(S0)": 195.18},
        {"Modello-ΔT": "BS-1Y",  "Strike più probabile (K)": 211.36, "Moneyness": 1.0829, "S a scadenza": 229.98, "Δ(S–K)": 18.62, "Δ(%)": "8.81%",  "(S0)": 195.18},
        {"Modello-ΔT": "Rab-6M", "Strike più probabile (K)": 239.58, "Moneyness": 0.9589, "S a scadenza": 229.98, "Δ(S–K)": -9.55, "Δ(%)": "-3.99%", "(S0)": 224.31},
        {"Modello-ΔT": "BS-6M",  "Strike più probabile (K)": 239.50, "Moneyness": 0.9593, "S a scadenza": 229.98, "Δ(S–K)": -9.52, "Δ(%)": "-3.98%", "(S0)": 224.31},
        {"Modello-ΔT": "Rab-3M", "Strike più probabile (K)": 244.39, "Moneyness": 0.9414, "S a scadenza": 229.98, "Δ(S–K)": -14.41, "Δ(%)": "-6.26%", "(S0)": 235.00},
        {"Modello-ΔT": "BS-3M",  "Strike più probabile (K)": 244.73, "Moneyness": 0.9400, "S a scadenza": 229.98, "Δ(S–K)": -14.75, "Δ(%)": "-6.41%", "(S0)": 235.00},
        {"Modello-ΔT": "Rab-1M", "Strike più probabile (K)": 244.93, "Moneyness": 0.9376, "S a scadenza": 229.98, "Δ(S–K)": -14.95, "Δ(%)": "-6.51%", "(S0)": 248.05},
        {"Modello-ΔT": "BS-1M",  "Strike più probabile (K)": 246.18, "Moneyness": 0.9925, "S a scadenza": 229.98, "Δ(S–K)": -16.20, "Δ(%)": "-6.58%", "(S0)": 248.05},
    ]),
    "NVDA": pd.DataFrame([
        {"Modello-ΔT": "Rab-1Y", "Strike più probabile (K)": 46.79,  "Moneyness": 0.7814, "S a scadenza": 137.71, "Δ(S–K)": 90.92, "Δ(%)": "194.35%", "(S0)": 59.87},
        {"Modello-ΔT": "BS-1Y",  "Strike più probabile (K)": 51.59,  "Moneyness": 0.8586, "S a scadenza": 137.71, "Δ(S–K)": 86.41, "Δ(%)": "168.45%", "(S0)": 59.87},
        {"Modello-ΔT": "Rab-6M", "Strike più probabile (K)": 105.78, "Moneyness": 0.8970, "S a scadenza": 137.71, "Δ(S–K)": 31.93, "Δ(%)": "30.18%",  "(S0)": 117.93},
        {"Modello-ΔT": "BS-6M",  "Strike più probabile (K)": 105.95, "Moneyness": 0.8979, "S a scadenza": 137.71, "Δ(S–K)": 31.93, "Δ(%)": "30.18%",  "(S0)": 117.93},
        {"Modello-ΔT": "Rab-3M", "Strike più probabile (K)": 121.70, "Moneyness": 0.8819, "S a scadenza": 137.71, "Δ(S–K)": 16.01, "Δ(%)": "13.15%",  "(S0)": 138.00},
        {"Modello-ΔT": "BS-3M",  "Strike più probabile (K)": 121.78, "Moneyness": 0.8819, "S a scadenza": 137.71, "Δ(S–K)": 15.93, "Δ(%)": "13.15%",  "(S0)": 138.00},
        {"Modello-ΔT": "Rab-1M", "Strike più probabile (K)": 125.35, "Moneyness": 0.9724, "S a scadenza": 137.71, "Δ(S–K)": 12.36, "Δ(%)": "9.86%",   "(S0)": 128.91},
        {"Modello-ΔT": "BS-1M",  "Strike più probabile (K)": 125.35, "Moneyness": 0.9724, "S a scadenza": 137.71, "Δ(S–K)": 12.36, "Δ(%)": "9.86%",   "(S0)": 128.91},
    ]),
    "ISP": pd.DataFrame([
        {"Modello-ΔT": "Rab-1Y", "Strike più probabile (K)": 3.42, "Moneyness": 0.9724, "S a scadenza": 4.81, "Δ(S–K)": 1.39, "Δ(%)": "40.73%", "(S0)": 3.52},
        {"Modello-ΔT": "BS-1Y",  "Strike più probabile (K)": 3.44, "Moneyness": 0.9774, "S a scadenza": 4.81, "Δ(S–K)": 1.38, "Δ(%)": "40.01%", "(S0)": 3.52},
        {"Modello-ΔT": "Rab-6M", "Strike più probabile (K)": 3.84, "Moneyness": 1.0025, "S a scadenza": 4.81, "Δ(S–K)": 0.97, "Δ(%)": "25.19%", "(S0)": 3.83},
        {"Modello-ΔT": "BS-6M",  "Strike più probabile (K)": 3.84, "Moneyness": 1.0025, "S a scadenza": 4.81, "Δ(S–K)": 0.97, "Δ(%)": "25.19%", "(S0)": 3.83},
        {"Modello-ΔT": "Rab-3M", "Strike più probabile (K)": 4.67, "Moneyness": 0.9673, "S a scadenza": 4.81, "Δ(S–K)": 0.14, "Δ(%)": "3.04%",  "(S0)": 4.83},
        {"Modello-ΔT": "BS-3M",  "Strike più probabile (K)": 4.69, "Moneyness": 0.9724, "S a scadenza": 4.81, "Δ(S–K)": 0.12, "Δ(%)": "2.51%",  "(S0)": 4.83},
        {"Modello-ΔT": "Rab-1M", "Strike più probabile (K)": 4.86, "Moneyness": 0.9824, "S a scadenza": 4.81, "Δ(S–K)": -0.05, "Δ(%)": "-1.07%", "(S0)": 4.95},
        {"Modello-ΔT": "BS-1M",  "Strike più probabile (K)": 4.86, "Moneyness": 0.9824, "S a scadenza": 4.81, "Δ(S–K)": -0.05, "Δ(%)": "-1.07%", "(S0)": 4.95},
    ])
}

MND_MODE_TABLES = {
    "AAPL": pd.DataFrame([
        {"Data": "23/01/2024", "Modello": "MND Rab", "ΔT": "1Y", "Strike più prob. (K)": 211.36, "Moneyness": 1.0829, "S a scadenza": 229.98, "Δ(S−K)": 18.62, "Δ (%)": "8.81%",  "S0": 195.18},
        {"Data": "23/01/2024", "Modello": "MND BS",  "ΔT": "1Y", "Strike più prob. (K)": 212.34, "Moneyness": 1.0879, "S a scadenza": 229.98, "Δ(S−K)": 17.64, "Δ (%)": "8.31%",  "S0": 195.18},
        {"Data": "19/07/2024", "Modello": "MND Rab", "ΔT": "6M", "Strike più prob. (K)": 239.53, "Moneyness": 1.0678, "S a scadenza": 229.98, "Δ(S−K)": -9.55, "Δ (%)": "-3.99%", "S0": 224.31},
        {"Data": "19/07/2024", "Modello": "MND BS",  "ΔT": "6M", "Strike più prob. (K)": 240.65, "Moneyness": 1.0729, "S a scadenza": 229.98, "Δ(S−K)": -10.67, "Δ (%)": "-4.44%", "S0": 224.31},
        {"Data": "18/10/2024", "Modello": "MND Rab", "ΔT": "3M", "Strike più prob. (K)": 249.76, "Moneyness": 1.0628, "S a scadenza": 229.98, "Δ(S−K)": -19.78, "Δ (%)": "-7.92%", "S0": 235.00},
        {"Data": "18/10/2024", "Modello": "MND BS",  "ΔT": "3M", "Strike più prob. (K)": 249.76, "Moneyness": 1.0628, "S a scadenza": 229.98, "Δ(S−K)": -19.78, "Δ (%)": "-7.92%", "S0": 235.00},
        {"Data": "18/12/2024", "Modello": "MND Rab", "ΔT": "1M", "Strike più prob. (K)": 246.18, "Moneyness": 0.9925, "S a scadenza": 229.98, "Δ(S−K)": -16.20, "Δ (%)": "-6.58%", "S0": 248.05},
        {"Data": "18/12/2024", "Modello": "MND BS",  "ΔT": "1M", "Strike più prob. (K)": 246.18, "Moneyness": 0.9925, "S a scadenza": 229.98, "Δ(S−K)": -16.20, "Δ (%)": "-6.58%", "S0": 248.05},
    ]),
    "NVDA": pd.DataFrame([
        {"Data": "23/01/2024", "Modello": "MND Rab", "ΔT": "1Y", "Strike più prob. (K)": 70.55, "Moneyness": 1.1784, "S a scadenza": 137.71, "Δ(S−K)": 67.16, "Δ (%)": "95.18%", "S0": 59.87},
        {"Data": "23/01/2024", "Modello": "MND BS",  "ΔT": "1Y", "Strike più prob. (K)": 81.99, "Moneyness": 1.3693, "S a scadenza": 137.71, "Δ(S−K)": 55.72, "Δ (%)": "67.97%", "S0": 59.87},
        {"Data": "19/07/2024", "Modello": "MND Rab", "ΔT": "6M", "Strike più prob. (K)": 106.37, "Moneyness": 0.9020, "S a scadenza": 137.71, "Δ(S−K)": 31.34, "Δ (%)": "29.46%", "S0": 117.93},
        {"Data": "19/07/2024", "Modello": "MND BS",  "ΔT": "6M", "Strike più prob. (K)": 106.37, "Moneyness": 0.9020, "S a scadenza": 137.71, "Δ(S−K)": 31.34, "Δ (%)": "29.46%", "S0": 117.93},
        {"Data": "18/10/2024", "Modello": "MND Rab", "ΔT": "3M", "Strike più prob. (K)": 150.14, "Moneyness": 1.0879, "S a scadenza": 137.71, "Δ(S−K)": -12.43, "Δ (%)": "-8.28%", "S0": 138.00},
        {"Data": "18/10/2024", "Modello": "MND BS",  "ΔT": "3M", "Strike più prob. (K)": 150.14, "Moneyness": 1.0879, "S a scadenza": 137.71, "Δ(S−K)": -12.43, "Δ (%)": "-8.28%", "S0": 138.00},
        {"Data": "18/12/2024", "Modello": "MND Rab", "ΔT": "1M", "Strike più prob. (K)": 126.64, "Moneyness": 0.9824, "S a scadenza": 137.71, "Δ(S−K)": 11.07, "Δ (%)": "8.74%", "S0": 128.91},
        {"Data": "18/12/2024", "Modello": "MND BS",  "ΔT": "1M", "Strike più prob. (K)": 126.64, "Moneyness": 0.9824, "S a scadenza": 137.71, "Δ(S−K)": 11.07, "Δ (%)": "8.74%", "S0": 128.91},
    ]),
    "ISP": pd.DataFrame([
        {"Data": "25/06/2024", "Modello": "MND Rab", "ΔT": "1Y", "Strike più prob. (K)": 3.44, "Moneyness": 0.9774, "S a scadenza": 4.81, "Δ(S−K)": 1.38, "Δ (%)": "40.01%", "S0": 3.52},
        {"Data": "25/06/2024", "Modello": "MND BS",  "ΔT": "1Y", "Strike più prob. (K)": 3.45, "Moneyness": 0.9824, "S a scadenza": 4.81, "Δ(S−K)": 1.36, "Δ (%)": "39.29%", "S0": 3.52},
        {"Data": "20/12/2024", "Modello": "MND Rab", "ΔT": "6M", "Strike più prob. (K)": 3.86, "Moneyness": 1.0075, "S a scadenza": 4.81, "Δ(S−K)": 0.95, "Δ (%)": "24.57%", "S0": 3.83},
        {"Data": "20/12/2024", "Modello": "MND BS",  "ΔT": "6M", "Strike più prob. (K)": 3.86, "Moneyness": 1.0075, "S a scadenza": 4.81, "Δ(S−K)": 0.95, "Δ (%)": "24.57%", "S0": 3.83},
        {"Data": "21/03/2025", "Modello": "MND Rab", "ΔT": "3M", "Strike più prob. (K)": 4.74, "Moneyness": 0.9824, "S a scadenza": 4.81, "Δ(S−K)": 0.07, "Δ (%)": "1.46%",  "S0": 4.83},
        {"Data": "21/03/2025", "Modello": "MND BS",  "ΔT": "3M", "Strike più prob. (K)": 5.13, "Moneyness": 1.0628, "S a scadenza": 4.81, "Δ(S−K)": -0.32, "Δ (%)": "-6.21%", "S0": 4.83},
        {"Data": "21/05/2025", "Modello": "MND Rab", "ΔT": "1M", "Strike più prob. (K)": 4.89, "Moneyness": 0.9874, "S a scadenza": 4.81, "Δ(S−K)": -0.08, "Δ (%)": "-1.57%", "S0": 4.95},
        {"Data": "21/05/2025", "Modello": "MND BS",  "ΔT": "1M", "Strike più prob. (K)": 4.89, "Moneyness": 0.9874, "S a scadenza": 4.81, "Δ(S−K)": -0.08, "Δ (%)": "-1.57%", "S0": 4.95},
    ]),
}

def _mode_table_from_pdf(ticker: str, measure: str) -> pd.DataFrame:
    t = str(ticker).upper().strip()
    m = str(measure).upper().strip()
    if m == "RND":
        return RND_MODE_TABLES.get(t, pd.DataFrame())
    if m == "MND":
        return MND_MODE_TABLES.get(t, pd.DataFrame())
    return pd.DataFrame()
# ==========================================================
# <<< FINE AGGIUNTA
# ==========================================================


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
    d["_tenor_lbl"] = d["DeltaT_years"].map(_years_to_tenor_label)  # <-- MODIFICA
    d["P_Q"] = _numeric(d.get("P_crash_Q (RND)"))
    d["P_P"] = _numeric(d.get("P_crash_P (MND)"))
    d = d.sort_values("DeltaT_years")

    fig.add_trace(go.Bar(x=d["_tenor_lbl"], y=d["P_Q"], name="P_crash_Q (RND)"))  # <-- MODIFICA
    fig.add_trace(go.Bar(x=d["_tenor_lbl"], y=d["P_P"], name="P_crash_P (MND)"))  # <-- MODIFICA

    fig.update_layout(
        title=f"Crash Probabilities – {ticker}",
        barmode="group",
        xaxis_title="DeltaT",
        yaxis_title="Probabilità",
        margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(categoryorder="array", categoryarray=["1M", "3M", "6M", "1Y"]),  # <-- MODIFICA
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

    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["Market_Price"], mode="lines", name="Market Price", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["BS_Price"], mode="lines", name="Black–Scholes"))
    fig.add_trace(go.Scatter(x=d["Date_dt"], y=d["Rab_Price"], mode="lines", name="Rabinovitch (rho est.)"))

    if rho_minus is not None:
        fig.add_trace(go.Scatter(x=d["Date_dt"], y=_numeric(d[rho_minus]), mode="lines", name="Rabinovitch (rho = -1)"))
    if rho_plus is not None:
        fig.add_trace(go.Scatter(x=d["Date_dt"], y=_numeric(d[rho_plus]), mode="lines", name="Rabinovitch (rho = +1)"))

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
    return fig


# ==========================================================
# CHARTS – RND / MND DENSITIES (ONLY MONEyness, ALL DATES, BS/RAB/BOTH)
# ==========================================================
def _normalize_model(s: str) -> str:
    return str(s).strip().lower()


def fig_density_curve(df_dens: pd.DataFrame, ticker: str, measure: str, model_choice: str) -> go.Figure:
    """
    Mostra SOLO Moneyness.
    Mostra TUTTE le date sovrapposte.
    model_choice: "Rabinovitch" / "BS" / "Entrambi"
    """
    fig = go.Figure()
    df_dens = _normalize_density_columns(df_dens)
    d = by_ticker(df_dens, ticker)

    if d is None or d.empty:
        fig.update_layout(title=f"{ticker} – {measure} Density (Moneyness) (nessun dato)")
        return fig

    # filtra misura
    d["Measure"] = d["Measure"].astype(str).str.upper().str.strip()
    d = d[d["Measure"] == str(measure).upper().strip()]
    if d.empty:
        fig.update_layout(title=f"{ticker} – {measure} Density (Moneyness) (nessun dato)")
        return fig

    # numerici
    d["Moneyness"] = _numeric(d["Moneyness"])
    d["Density"] = _numeric(d["Density"])
    d = d.dropna(subset=["Moneyness", "Density"])
    if d.empty:
        fig.update_layout(title=f"{ticker} – {measure} Density (Moneyness) (nessun dato)")
        return fig

    # date (non obbligatoria, ma serve per distinguere in legenda)
    if "date" in d.columns:
        dt = pd.to_datetime(d["date"], errors="coerce", dayfirst=False)
        dt2 = pd.to_datetime(d["date"], errors="coerce", dayfirst=True)
        d["_date_dt"] = dt
        d.loc[d["_date_dt"].isna(), "_date_dt"] = dt2[d["_date_dt"].isna()]
    else:
        d["_date_dt"] = pd.NaT

    # modello filter
    d["Modello_norm"] = d["Modello"].map(_normalize_model)
    choice = str(model_choice).strip().lower()

    def keep_row(model_norm: str) -> bool:
        if choice == "entrambi":
            return True
        if choice == "bs":
            return ("bs" in model_norm) or ("black" in model_norm and "scholes" in model_norm)
        # rabinovitch
        return ("rab" in model_norm) or ("rabin" in model_norm)

    d = d[d["Modello_norm"].map(keep_row)]
    if d.empty:
        fig.update_layout(title=f"{ticker} – {measure} Density (Moneyness) ({model_choice}) (nessun dato)")
        return fig

    # ordine tenori
    d["_T"] = _parse_deltaT_to_years(d["DeltaT"]).fillna(999.0)
    d["_tenor_lbl"] = d["_T"].map(_years_to_tenor_label)  # <-- MODIFICA

    # label date per legenda
    if d["_date_dt"].notna().any():
        d["_date_lbl"] = d["_date_dt"].dt.date.astype(str)
    elif "date" in d.columns:
        d["_date_lbl"] = d["date"].astype(str)
    else:
        d["_date_lbl"] = ""

    # gruppa per (Modello, TenorLabel, date) così vedi tutte sovrapposte
    group_cols = ["Modello", "_tenor_lbl", "_date_lbl"]  # <-- MODIFICA
    for (model, tenor_lbl, dtlbl), g in d.sort_values(["_T", "Moneyness"]).groupby(group_cols, dropna=False):
        g = g.sort_values("Moneyness")
        name = f"{tenor_lbl} | {dtlbl} | {model}"  # <-- MODIFICA
        fig.add_trace(go.Scatter(
            x=g["Moneyness"],
            y=g["Density"],
            mode="lines",
            name=name
        ))

    fig.update_layout(
        title=f"{ticker} – {measure} Density (Moneyness) – {model_choice}",
        xaxis_title="Moneyness (K/S)",
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

    model_choice = "Rabinovitch"
    if section in ["RND", "MND"]:
        st.divider()
        st.subheader("RND/MND options")
        model_choice = st.radio("Modello", ["Rabinovitch", "BS", "Entrambi"], index=0, horizontal=True)

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

    st.plotly_chart(fig_opt_timeseries(df_opt, ticker), use_container_width=True)
    st.plotly_chart(fig_mispricing_pct_timeseries(df_opt, ticker), use_container_width=True)
    st.plotly_chart(fig_sens_diff_pct(df_sens, ticker), use_container_width=True)
    st.plotly_chart(fig_sensitivity_rho(df_opt, ticker), use_container_width=True)
    st.plotly_chart(fig_opt_error_hist(df_opt, ticker), use_container_width=True)

elif section == "IV":
    df_surf = by_ticker(data["iv"], ticker)
    st.plotly_chart(fig_iv_surface_3d(df_surf, ticker), use_container_width=True)

elif section == "RND":
    df_dens = data.get("dens", pd.DataFrame())
    st.plotly_chart(
        fig_density_curve(df_dens, ticker, measure="RND", model_choice=model_choice),
        use_container_width=True
    )

    # >>> AGGIUNTA TABELLA RND (strike più probabile)
    tbl = _mode_table_from_pdf(ticker, "RND")
    if tbl is not None and not tbl.empty:
        st.subheader("Strike più probabile (RND) – tabella")
        st.dataframe(tbl, use_container_width=True, hide_index=True)
    # <<<

elif section == "MND":
    df_dens = data.get("dens", pd.DataFrame())
    st.plotly_chart(
        fig_density_curve(df_dens, ticker, measure="MND", model_choice=model_choice),
        use_container_width=True
    )

    # >>> AGGIUNTA TABELLA MND (strike più probabile)
    tbl = _mode_table_from_pdf(ticker, "MND")
    if tbl is not None and not tbl.empty:
        st.subheader("Strike più probabile (MND) – tabella")
        st.dataframe(tbl, use_container_width=True, hide_index=True)
    # <<<

elif section == "Crash Prob":
    df = by_ticker(data["crash"], ticker)
    st.plotly_chart(fig_crash(df, ticker), use_container_width=True)

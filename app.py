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

    bs_vals = d["Diff_BS_Market"].dropna()
    rab_vals = d["Diff_Rab_Market"].dropna()

    if bs_vals.empty and rab_vals.empty:
        fig.update_layout(title=f"Pricing Error Distribution – {ticker} (nessun dato)")
        return fig

    if not bs_vals.empty:
        fig.add_trace(go.Histogram(x=bs_vals, nbinsx=60, name="BS - Market", opacity=0.6))
    if not rab_vals.empty:
        fig.add_trace(go.Histogram(x=rab_vals, nbinsx=60, name="Rab - Market", opacity=0.6))

    fig.update_layout(
        title=f"Pricing Error Distribution – {ticker}",
        barmode="overlay",
        xaxis_title="Model - Market",
        yaxis_title="Count",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ==========================================================
# CHARTS – RND / MND (ONLY MONEYNES) + HISTOGRAMS + TABLES
# ==========================================================
MONEYNESS_BINS = [
    (-np.inf, 0.80, "≤ 0.80"),
    (0.80, 0.90, "0.80–0.90"),
    (0.90, 1.00, "0.90–1.00"),
    (1.00, 1.10, "1.00–1.10"),
    (1.10, 1.20, "1.10–1.20"),
    (1.20, np.inf, "≥ 1.20"),
]


def _normalize_model_label(x: str) -> str:
    s = str(x).strip()
    sl = s.lower()
    if "bs" in sl:
        return "BS"
    if "rab" in sl or "rabin" in sl:
        return "Rabinovitch"
    return s


def _prep_density(df_dens: pd.DataFrame, ticker: str, measure: str) -> pd.DataFrame:
    """
    CSV merge (tuo): ticker, date, DeltaT, Modello, Measure, Moneyness, Density
    """
    df_dens = _std_ticker(df_dens)
    if df_dens is None or df_dens.empty:
        return pd.DataFrame()

    d = by_ticker(df_dens, ticker)

    if "Measure" not in d.columns:
        return pd.DataFrame()

    d["Measure"] = d["Measure"].astype(str).str.upper().str.strip()
    d = d[d["Measure"] == str(measure).upper()].copy()
    if d.empty:
        return pd.DataFrame()

    # date (nel tuo file è 'date' in formato m/d/Y)
    if "date" in d.columns:
        d["date_dt"] = _to_datetime_safe(d["date"], dayfirst=False)
    elif "Date" in d.columns:
        d["date_dt"] = _to_datetime_safe(d["Date"], dayfirst=False)
    else:
        d["date_dt"] = pd.NaT

    d["date_str"] = d["date_dt"].dt.strftime("%Y-%m-%d")
    d.loc[d["date_dt"].isna(), "date_str"] = "NA"

    # numerici
    if "Moneyness" in d.columns:
        d["Moneyness"] = _numeric(d["Moneyness"])
    if "Density" in d.columns:
        d["Density"] = _numeric(d["Density"])

    # pulizia labels
    d["Modello"] = d.get("Modello", "").astype(str).map(_normalize_model_label)
    d["DeltaT"] = d.get("DeltaT", "").astype(str).str.strip()

    d = d.dropna(subset=["Moneyness", "Density"])
    return d


def fig_density_curve_moneyness(df_dens: pd.DataFrame, ticker: str, measure: str, model_choice: str) -> go.Figure:
    """
    Curve a campana su Moneyness, con tutte le date sovrapposte.
    """
    fig = go.Figure()
    d = _prep_density(df_dens, ticker, measure)

    if d.empty:
        fig.update_layout(title=f"{ticker} – {measure} Density (Moneyness) (nessun dato)")
        return fig

    # filtro modello: Rab / BS / Entrambi
    mc = str(model_choice).strip().lower()
    if mc == "bs":
        d = d[d["Modello"] == "BS"]
    elif mc.startswith("rab"):
        d = d[d["Modello"] == "Rabinovitch"]
    else:
        d = d[d["Modello"].isin(["BS", "Rabinovitch"])]

    if d.empty:
        fig.update_layout(title=f"{ticker} – {measure} Density (Moneyness) ({model_choice}) (nessun dato)")
        return fig

    # ordine tenori per legenda più leggibile
    tenor_order = {"1M": 1, "3M": 2, "6M": 3, "1Y": 4}
    d["_ten"] = d["DeltaT"].astype(str).str.upper().map(lambda x: tenor_order.get(x, 99))

    # gruppo: (date, tenor, modello)
    d = d.sort_values(["date_dt", "_ten", "Modello", "Moneyness"])

    for (date_str, tenor, modello), g in d.groupby(["date_str", "DeltaT", "Modello"], dropna=False):
        g = g.sort_values("Moneyness")
        name = f"{tenor} | {date_str} | {modello}"
        fig.add_trace(go.Scatter(
            x=g["Moneyness"],
            y=g["Density"],
            mode="lines",
            name=name
        ))

    # Layout: legenda orizzontale sopra + margine alto per non sovrapporre il titolo
    title_txt = f"{ticker} – {measure} Density (Moneyness) – {model_choice}"
    fig.update_layout(
        title=dict(text=title_txt, y=0.92),
        xaxis_title="Moneyness (K/S)",
        yaxis_title="Densità",
        margin=dict(l=0, r=0, t=135, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,          # sopra al titolo
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)"
        ),
    )
    return fig


def _integrate_prob_in_bin(x: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    """
    Integra y dx su [a,b] con interpolazione ai bordi.
    Restituisce massa di probabilità (non %).
    """
    if len(x) < 2:
        return 0.0

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # se bin infinito, limitiamo al range osservato
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    aa = lo if not np.isfinite(a) else a
    bb = hi if not np.isfinite(b) else b

    if bb <= lo or aa >= hi or aa >= bb:
        return 0.0

    # punti interni
    mask = (x > aa) & (x < bb)
    xi = x[mask]
    yi = y[mask]

    # interp ai bordi (se dentro range)
    ya = np.interp(aa, x, y)
    yb = np.interp(bb, x, y)

    xx = np.concatenate([[aa], xi, [bb]])
    yy = np.concatenate([[ya], yi, [yb]])

    # assicura ordinamento
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]

    return float(np.trapz(yy, xx))


def fig_moneyness_histogram(df_dens: pd.DataFrame, ticker: str, measure: str, model_choice: str) -> go.Figure:
    """
    Istogrammi (probabilità per intervalli di moneyness), calcolati integrando la densità.
    Stile: barre sovrapposte/semi-trasparenti come nella tua figura.
    """
    fig = go.Figure()
    d = _prep_density(df_dens, ticker, measure)

    if d.empty:
        fig.update_layout(title=f"{ticker} – Probabilità per intervalli di Moneyness ({measure}) (nessun dato)")
        return fig

    mc = str(model_choice).strip().lower()
    if mc == "bs":
        d = d[d["Modello"] == "BS"]
    elif mc.startswith("rab"):
        d = d[d["Modello"] == "Rabinovitch"]
    else:
        d = d[d["Modello"].isin(["BS", "Rabinovitch"])]

    if d.empty:
        fig.update_layout(title=f"{ticker} – Probabilità per intervalli di Moneyness ({measure}) ({model_choice}) (nessun dato)")
        return fig

    # integrazione per ogni (date, tenor, modello)
    rows = []
    for (date_str, tenor, modello), g in d.groupby(["date_str", "DeltaT", "Modello"], dropna=False):
        g = g.sort_values("Moneyness")
        x = g["Moneyness"].values
        y = g["Density"].values

        # normalizzazione (difensivo): garantisce area ~ 1
        area = float(np.trapz(y, x)) if len(x) > 1 else np.nan
        if np.isfinite(area) and area > 0:
            y = y / area

        for a, b, lab in MONEYNESS_BINS:
            p = _integrate_prob_in_bin(x, y, a, b) * 100.0
            rows.append({
                "bin": lab,
                "prob_pct": p,
                "DeltaT": tenor,
                "date": date_str,
                "Modello": modello
            })

    hist = pd.DataFrame(rows)
    if hist.empty:
        fig.update_layout(title=f"{ticker} – Probabilità per intervalli di Moneyness ({measure}) (nessun dato)")
        return fig

    # ordine bin
    bin_order = [lab for _, _, lab in MONEYNESS_BINS]
    hist["bin"] = pd.Categorical(hist["bin"], categories=bin_order, ordered=True)
    hist = hist.sort_values(["bin", "date", "DeltaT", "Modello"])

    # barre: una traccia per (tenor, modello, date) -> molte, ma coerente con “tutte le date”
    for (date_str, tenor, modello), g in hist.groupby(["date", "DeltaT", "Modello"], dropna=False):
        name = f"{tenor} | {date_str} | {modello}"
        fig.add_trace(go.Bar(
            x=g["bin"].astype(str),
            y=g["prob_pct"],
            name=name,
            opacity=0.55
        ))

    fig.update_layout(
        title=dict(text=f"{ticker} – Probabilità per intervalli di Moneyness ({measure}) – {model_choice}", y=0.92),
        xaxis_title="Moneyness (K/S)",
        yaxis_title="Probabilità (%)",
        barmode="overlay",
        margin=dict(l=0, r=0, t=135, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)"
        ),
    )
    return fig


# -------------------------
# TABELLE "DA TESI" (hardcoded dal PDF)
# -------------------------
def _df_table(data, columns):
    return pd.DataFrame(data, columns=columns)


RND_TABLES = {
    "AAPL": {
        "prob_moneyness": _df_table(
            [
                ["≤ 0.80", 22.46, 16.56, 19.14, 15.37, 6.63, 5.44, 1.14, 0.99],
                ["0.80–0.90", 11.45, 16.10, 16.46, 18.91, 15.51, 15.32, 13.42, 13.62],
                ["0.90–1.00", 16.04, 18.47, 19.41, 19.97, 20.82, 22.97, 41.12, 43.62],
                ["1.00–1.10", 18.16, 17.06, 18.97, 18.37, 27.38, 27.31, 43.62, 41.12],
                ["1.10–1.20", 15.15, 15.46, 15.12, 13.57, 17.06, 16.02, 7.62, 7.44],
                ["≥ 1.20", 16.74, 16.35, 10.90, 14.81, 12.60, 10.93, 3.08, 3.21],
            ],
            ["Moneyness", "Rab 1Y", "BS 1Y", "Rab 6M", "BS 6M", "Rab 3M", "BS 3M", "Rab 1M", "BS 1M"]
        ),
        "mode_strike": _df_table(
            [
                ["Rab-1Y", 209.40, 1.0729, 229.98, 20.58, "9.83%", 195.18],
                ["BS-1Y", 211.36, 1.0829, 229.98, 18.62, "8.81%", 195.18],
                ["Rab-6M", 239.58, 0.9589, 229.98, -9.55, "-3.99%", 224.31],
                ["BS-6M", 239.50, 0.9593, 229.98, -9.52, "-3.98%", 224.31],
                ["Rab-3M", 244.39, 0.9414, 229.98, -14.41, "-6.26%", 235.00],
                ["BS-3M", 244.73, 0.9400, 229.98, -14.75, "-6.41%", 235.00],
                ["Rab-1M", 244.93, 0.9376, 229.98, -14.95, "-6.51%", 248.05],
                ["BS-1M", 246.18, 0.9925, 229.98, -16.20, "-6.58%", 248.05],
            ],
            ["Modello-ΔT", "Strike più probabile (K)", "Moneyness", "S a scadenza", "Δ(S–K)", "Δ(%)", "S0"]
        ),
    },
    "NVDA": {
        "prob_moneyness": _df_table(
            [
                ["≤ 0.80", 34.30, 30.98, 36.84, 34.77, 24.98, 23.64, 10.07, 9.63],
                ["0.80–0.90", 19.53, 15.57, 15.17, 13.76, 18.36, 17.96, 11.70, 11.70],
                ["0.90–1.00", 18.13, 18.14, 15.57, 15.26, 25.36, 25.77, 25.97, 25.77],
                ["1.00–1.10", 14.06, 17.14, 16.36, 17.58, 17.78, 17.21, 18.31, 21.31],
                ["1.10–1.20", 13.98, 18.17, 16.06, 18.63, 11.52, 11.65, 14.40, 10.78],
            ],
            ["Moneyness", "Rab 1Y", "BS 1Y", "Rab 6M", "BS 6M", "Rab 3M", "BS 3M", "Rab 1M", "BS 1M"]
        ),
        "mode_strike": _df_table(
            [
                ["Rab-1Y", 46.79, 0.7814, 137.71, 90.92, "194.35%", 59.87],
                ["BS-1Y", 51.59, 0.8586, 137.71, 86.41, "168.45%", 59.87],
                ["Rab-6M", 105.78, 0.8970, 137.71, 31.93, "30.18%", 117.93],
                ["BS-6M", 105.95, 0.8979, 137.71, 31.93, "30.18%", 117.93],
                ["Rab-3M", 121.70, 0.8819, 137.71, 16.01, "13.15%", 138.00],
                ["BS-3M", 121.78, 0.8819, 137.71, 15.93, "13.15%", 138.00],
                ["Rab-1M", 125.35, 0.9724, 137.71, 12.36, "9.86%", 128.91],
                ["BS-1M", 125.35, 0.9724, 137.71, 12.36, "9.86%", 128.91],
            ],
            ["Modello-ΔT", "Strike più probabile (K)", "Moneyness", "S a scadenza", "Δ(S–K)", "Δ(%)", "S0"]
        ),
    },
    "ISP": {
        "prob_moneyness": _df_table(
            [
                ["≤ 0.80", 29.43, 24.69, 17.56, 15.96, 13.16, 12.39, 8.09, 7.90],
                ["0.80–0.90", 19.05, 17.63, 19.45, 17.96, 16.02, 15.19, 10.34, 10.05],
                ["0.90–1.00", 20.01, 20.19, 27.35, 27.06, 27.73, 27.25, 41.71, 41.06],
                ["1.00–1.10", 12.73, 13.99, 15.31, 16.28, 24.89, 25.51, 32.26, 32.95],
                ["1.10–1.20", 8.31, 9.72, 7.58, 8.26, 12.87, 13.79, 6.55, 6.91],
                ["≥ 1.20", 10.47, 13.79, 12.75, 14.48, 5.33, 5.88, 1.05, 1.13],
            ],
            ["Moneyness", "Rab 1Y", "BS 1Y", "Rab 6M", "BS 6M", "Rab 3M", "BS 3M", "Rab 1M", "BS 1M"]
        ),
        "mode_strike": _df_table(
            [
                ["Rab-1Y", 3.42, 0.9724, 4.81, 1.39, "40.73%", 3.52],
                ["BS-1Y", 3.44, 0.9774, 4.81, 1.38, "40.01%", 3.52],
                ["Rab-6M", 3.84, 1.0025, 4.81, 0.97, "25.19%", 3.83],
                ["BS-6M", 3.84, 1.0025, 4.81, 0.97, "25.19%", 3.83],
                ["Rab-3M", 4.67, 0.9673, 4.81, 0.14, "3.04%", 4.83],
                ["BS-3M", 4.69, 0.9724, 4.81, 0.12, "2.51%", 4.83],
                ["Rab-1M", 4.86, 0.9824, 4.81, -0.05, "-1.07%", 4.95],
                ["BS-1M", 4.86, 0.9824, 4.81, -0.05, "-1.07%", 4.95],
            ],
            ["Modello-ΔT", "Strike più probabile (K)", "Moneyness", "S a scadenza", "Δ(S–K)", "Δ(%)", "S0"]
        ),
    }
}

MND_MODE_TABLES = {
    "AAPL": _df_table(
        [
            ["2024-01-23", "MND", "Rab", "1Y", 211.36, 1.0829, 229.98, 18.62, "8.81%", 195.18],
            ["2024-01-23", "MND", "BS",  "1Y", 212.34, 1.0879, 229.98, 17.64, "8.31%", 195.18],
            ["2024-07-19", "MND", "Rab", "6M", 239.53, 1.0678, 229.98, -9.55, "-3.99%", 224.31],
            ["2024-07-19", "MND", "BS",  "6M", 240.65, 1.0729, 229.98, -10.67, "-4.44%", 224.31],
            ["2024-10-18", "MND", "Rab", "3M", 249.76, 1.0628, 229.98, -19.78, "-7.92%", 235.00],
            ["2024-10-18", "MND", "BS",  "3M", 249.76, 1.0628, 229.98, -19.78, "-7.92%", 235.00],
            ["2024-12-18", "MND", "Rab", "1M", 246.18, 0.9925, 229.98, -16.20, "-6.58%", 248.05],
            ["2024-12-18", "MND", "BS",  "1M", 246.18, 0.9925, 229.98, -16.20, "-6.58%", 248.05],
        ],
        ["Data", "Modello", "Tipo", "ΔT", "Strike più prob. (K)", "Moneyness", "S a scadenza", "Δ(S−K)", "Δ(%)", "S0"]
    ),
    "NVDA": _df_table(
        [
            ["2024-01-23", "MND", "Rab", "1Y", 70.55, 1.1784, 137.71, 67.16, "95.18%", 59.87],
            ["2024-01-23", "MND", "BS",  "1Y", 81.99, 1.3693, 137.71, 55.72, "67.97%", 59.87],
            ["2024-07-19", "MND", "Rab", "6M", 106.37, 0.9020, 137.71, 31.34, "29.46%", 117.93],
            ["2024-07-19", "MND", "BS",  "6M", 106.37, 0.9020, 137.71, 31.34, "29.46%", 117.93],
            ["2024-10-18", "MND", "Rab", "3M", 150.14, 1.0879, 137.71, -12.43, "-8.28%", 138.00],
            ["2024-10-18", "MND", "BS",  "3M", 150.14, 1.0879, 137.71, -12.43, "-8.28%", 138.00],
            ["2024-12-18", "MND", "Rab", "1M", 126.64, 0.9824, 137.71, 11.07, "8.74%", 128.91],
            ["2024-12-18", "MND", "BS",  "1M", 126.64, 0.9824, 137.71, 11.07, "8.74%", 128.91],
        ],
        ["Data", "Modello", "Tipo", "ΔT", "Strike più prob. (K)", "Moneyness", "S a scadenza", "Δ(S−K)", "Δ(%)", "S0"]
    ),
    "ISP": _df_table(
        [
            ["2024-06-25", "MND", "Rab", "1Y", 3.44, 0.9774, 4.81, 1.38, "40.01%", 3.52],
            ["2024-06-25", "MND", "BS",  "1Y", 3.45, 0.9824, 4.81, 1.36, "39.29%", 3.52],
            ["2024-12-20", "MND", "Rab", "6M", 3.86, 1.0075, 4.81, 0.95, "24.57%", 3.83],
            ["2024-12-20", "MND", "BS",  "6M", 3.86, 1.0075, 4.81, 0.95, "24.57%", 3.83],
            ["2025-03-21", "MND", "Rab", "3M", 4.74, 0.9824, 4.81, 0.07, "1.46%", 4.83],
            ["2025-03-21", "MND", "BS",  "3M", 5.13, 1.0628, 4.81, -0.32, "-6.21%", 4.83],
            ["2025-05-21", "MND", "Rab", "1M", 4.89, 0.9874, 4.81, -0.08, "-1.57%", 4.95],
            ["2025-05-21", "MND", "BS",  "1M", 4.89, 0.9874, 4.81, -0.08, "-1.57%", 4.95],
        ],
        ["Data", "Modello", "Tipo", "ΔT", "Strike più prob. (K)", "Moneyness", "S a scadenza", "Δ(S−K)", "Δ(%)", "S0"]
    ),
}


def render_thesis_tables(ticker: str, measure: str):
    if measure == "RND":
        pack = RND_TABLES.get(ticker)
        if not pack:
            st.info("Tabelle RND da tesi non disponibili per questo ticker (hardcoded).")
            return
        st.subheader("Tabelle (da tesi) – RND")
        st.markdown("**Probabilità per gruppi di moneyness**")
        st.dataframe(pack["prob_moneyness"], use_container_width=True, hide_index=True)
        st.markdown("**Strike più probabile e delta a scadenza**")
        st.dataframe(pack["mode_strike"], use_container_width=True, hide_index=True)

    if measure == "MND":
        tab = MND_MODE_TABLES.get(ticker)
        if tab is None:
            st.info("Tabella MND (strike più probabile) non disponibile per questo ticker (hardcoded).")
            return
        st.subheader("Tabelle (da tesi) – MND")
        st.markdown("**Strike più probabile e delta a scadenza (MND)**")
        st.dataframe(tab, use_container_width=True, hide_index=True)


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
    if section in ["RND", "MND"]:
        st.divider()
        st.subheader("RND/MND options")
        model_choice = st.radio("Modello", ["Rabinovitch", "BS", "Entrambi"], index=2, horizontal=True)

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

    # 1) curve
    st.plotly_chart(
        fig_density_curve_moneyness(df_dens, ticker, measure="RND", model_choice=model_choice),
        use_container_width=True
    )

    # 2) istogrammi moneyness
    st.plotly_chart(
        fig_moneyness_histogram(df_dens, ticker, measure="RND", model_choice=model_choice),
        use_container_width=True
    )

    # 3) tabelle da tesi
    render_thesis_tables(ticker, "RND")

elif section == "MND":
    df_dens = data.get("dens", pd.DataFrame())

    st.plotly_chart(
        fig_density_curve_moneyness(df_dens, ticker, measure="MND", model_choice=model_choice),
        use_container_width=True
    )

    st.plotly_chart(
        fig_moneyness_histogram(df_dens, ticker, measure="MND", model_choice=model_choice),
        use_container_width=True
    )

    render_thesis_tables(ticker, "MND")

elif section == "Crash Prob":
    df = by_ticker(data["crash"], ticker)
    st.plotly_chart(fig_crash(df, ticker), use_container_width=True)

st.caption(
    "RND/MND: il CSV /data/rnd_mnd_density_all.csv deve contenere colonne: "
    "ticker, date, DeltaT, Modello, Measure, Moneyness, Density. "
    "Gli istogrammi sono calcolati integrando la densità in bin di moneyness."
)

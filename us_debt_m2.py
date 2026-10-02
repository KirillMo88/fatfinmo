from __future__ import annotations

import numpy as np
import pandas as pd


DEBT_M2_SERIES_START = "1959-01-01"
DEBT_M2_CHANNEL_START = pd.Timestamp("1990-01-01")


def build_us_debt_m2_history(fred_data: pd.DataFrame) -> pd.DataFrame:
    """Align quarterly end-period federal debt with monthly M2 and calculate the ratio."""
    columns = [
        "Date",
        "FederalDebtUSD_Bn",
        "M2USD_Bn",
        "DebtM2Ratio",
        "SMA20",
        "SMA50",
        "SMA100",
        "SMA200",
        "Support",
        "Resistance",
    ]
    if fred_data.empty or not {"Series_ID", "Date", "Value"}.issubset(fred_data.columns):
        return pd.DataFrame(columns=columns)

    data = fred_data.copy()
    data["Series_ID"] = data["Series_ID"].astype(str).str.upper()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
    data = data.dropna(subset=["Date", "Value"])

    debt = data.loc[data["Series_ID"].eq("GFDEBTN"), ["Date", "Value"]].copy()
    m2 = data.loc[data["Series_ID"].eq("M2SL"), ["Date", "Value"]].copy()
    if debt.empty or m2.empty:
        return pd.DataFrame(columns=columns)

    # FRED timestamps quarterly observations at quarter start even when the
    # measure is end-of-period; align it to quarter-end before carrying forward.
    debt["Date"] = debt["Date"].dt.to_period("Q").dt.end_time.dt.normalize()
    debt = debt.rename(columns={"Value": "FederalDebtUSD_Mn"}).sort_values("Date")
    debt["FederalDebtUSD_Bn"] = debt["FederalDebtUSD_Mn"] / 1000.0
    debt = debt[["Date", "FederalDebtUSD_Bn"]].drop_duplicates("Date", keep="last")

    # M2 is a monthly observation; move period labels to month-end so both
    # series align on the same economic observation dates.
    m2["Date"] = m2["Date"].dt.to_period("M").dt.end_time.dt.normalize()
    m2 = m2.rename(columns={"Value": "M2USD_Bn"}).sort_values("Date")
    m2 = m2[["Date", "M2USD_Bn"]].drop_duplicates("Date", keep="last")

    out = pd.merge_asof(m2, debt, on="Date", direction="backward")
    out["DebtM2Ratio"] = out["FederalDebtUSD_Bn"] / out["M2USD_Bn"].where(out["M2USD_Bn"].gt(0))
    out = out.dropna(subset=["DebtM2Ratio"]).sort_values("Date").reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=columns)

    for window in (20, 50, 100, 200):
        out[f"SMA{window}"] = out["DebtM2Ratio"].rolling(window, min_periods=window).mean()

    out["Support"] = np.nan
    out["Resistance"] = np.nan
    channel = out.loc[out["Date"].ge(DEBT_M2_CHANNEL_START)]
    if len(channel) >= 36:
        x = (channel["Date"] - channel["Date"].min()).dt.days.to_numpy(dtype=float) / 365.25
        y = channel["DebtM2Ratio"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        center = slope * x + intercept
        out.loc[channel.index, "Support"] = center + float(np.nanquantile(residual, 0.10))
        out.loc[channel.index, "Resistance"] = center + float(np.nanquantile(residual, 0.90))
    return out[columns]


def build_us_debt_m2_figure(frame: pd.DataFrame):
    import plotly.graph_objects as go

    fig = go.Figure()
    if frame.empty:
        return fig

    colors = {
        "SMA20": "#3b82f6",
        "SMA50": "#2dd4bf",
        "SMA100": "#f59e0b",
        "SMA200": "#a78bfa",
    }
    for name, color in colors.items():
        fig.add_trace(
            go.Scatter(
                x=frame["Date"],
                y=frame[name],
                mode="lines",
                name=name.replace("SMA", "SMA "),
                line={"color": color, "width": 1.15},
                hovertemplate=f"%{{x|%Y-%m}}<br>{name.replace('SMA', 'SMA ')}: %{{y:.3f}}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["DebtM2Ratio"],
            mode="lines",
            name="(GFDEBTN / 1,000) / M2SL",
            line={"color": "#ff4655", "width": 2.0},
            customdata=np.column_stack([frame["FederalDebtUSD_Bn"], frame["M2USD_Bn"]]),
            hovertemplate=(
                "Date: %{x|%Y-%m}<br>Debt / M2: %{y:.3f}<br>"
                "Federal debt: $%{customdata[0]:,.0f}B<br>M2: $%{customdata[1]:,.0f}B<extra></extra>"
            ),
        )
    )

    for name, color in (("Support", "#94a3b8"), ("Resistance", "#cbd5e1")):
        fig.add_trace(
            go.Scatter(
                x=frame["Date"],
                y=frame[name],
                mode="lines",
                name=f"Long-term {name.lower()}",
                line={"color": color, "width": 1.2, "dash": "dash"},
                hovertemplate=f"%{{x|%Y-%m}}<br>{name}: %{{y:.3f}}<extra></extra>",
            )
        )

    annotations = [
        ("1997-12-31", "ASIAN CRISIS", 28),
        ("2001-06-30", "DOT-COM BUBBLE", -28),
        ("2011-09-30", "GFC + EURO CRISIS", 28),
        ("2022-03-31", "EVERYTHING BUBBLE", -30),
    ]
    for date_text, label, yshift in annotations:
        target = pd.Timestamp(date_text)
        index = (frame["Date"] - target).abs().idxmin()
        row = frame.loc[index]
        fig.add_annotation(
            x=row["Date"],
            y=row["DebtM2Ratio"],
            text=label,
            showarrow=False,
            yshift=yshift,
            font={"color": "#60a5fa", "size": 10},
        )

    fig.update_layout(
        title="U.S. Federal Debt / M2 — (GFDEBTN / 1,000) / M2SL",
        height=470,
        paper_bgcolor="#0f131a",
        plot_bgcolor="#0f131a",
        font={"color": "#e5e7eb", "size": 11},
        margin={"l": 58, "r": 28, "t": 58, "b": 44},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "top", "y": -0.15, "xanchor": "left", "x": 0},
    )
    fig.update_xaxes(tickformat="%b'%y", showgrid=False, zeroline=False, color="#cbd5e1", linecolor="#475569", ticks="outside")
    fig.update_yaxes(title="Ratio", showgrid=True, gridcolor="#263241", zeroline=False, color="#cbd5e1", linecolor="#475569", ticks="outside")
    return fig

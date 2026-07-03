"""Plotly figures for live feedback: a tuner needle gauge and an FFT spectrum.

Both are pure functions returning `go.Figure` so they can be dropped into a
Streamlit placeholder and redrawn every frame.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

# Cents window shown on the needle dial.
CENTS_RANGE = 50.0
IN_TUNE_CENTS = 5.0


def needle_figure(
    cents: Optional[float],
    note_label: str = "--",
    detail: str = "",
    in_tune_cents: float = IN_TUNE_CENTS,
) -> go.Figure:
    """
    A tuner dial: needle points to the cents deviation (-50..+50).
    Green center band = in tune; red flanks = flat/sharp.
    """
    value = 0.0 if cents is None else float(np.clip(cents, -CENTS_RANGE, CENTS_RANGE))
    if cents is None:
        bar_color = "#888888"
    elif abs(cents) <= in_tune_cents:
        bar_color = "#2ecc71"
    elif abs(cents) <= 20:
        bar_color = "#f1c40f"
    else:
        bar_color = "#e74c3c"

    title_text = f"<b>{note_label}</b>"
    if detail:
        title_text += f"<br><span style='font-size:0.6em;color:#aaa'>{detail}</span>"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " ¢", "font": {"size": 34}},
            title={"text": title_text, "font": {"size": 40}},
            gauge={
                "axis": {
                    "range": [-CENTS_RANGE, CENTS_RANGE],
                    "tickmode": "array",
                    "tickvals": [-50, -25, -in_tune_cents, 0, in_tune_cents, 25, 50],
                    "ticktext": ["-50", "-25", "", "0", "", "+25", "+50"],
                },
                "bar": {"color": bar_color, "thickness": 0.25},
                "steps": [
                    {"range": [-CENTS_RANGE, -20], "color": "#3a1f1f"},
                    {"range": [-20, -in_tune_cents], "color": "#3a331f"},
                    {"range": [-in_tune_cents, in_tune_cents], "color": "#1f3a24"},
                    {"range": [in_tune_cents, 20], "color": "#3a331f"},
                    {"range": [20, CENTS_RANGE], "color": "#3a1f1f"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.85,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=30, r=30, t=90, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
    )
    return fig


def tuner_meter_figure(
    cents: Optional[float],
    note_label: str = "--",
    target_hz: Optional[float] = None,
    detected_hz: Optional[float] = None,
    in_tune_cents: float = IN_TUNE_CENTS,
) -> go.Figure:
    """
    Clean horizontal strobe-style tuner meter (GuitarTuna-like).

    A center green zone marks "in tune"; the marker slides left (flat) or right
    (sharp). Large note name in the middle, with FLAT / SHARP direction hints.
    """
    has = cents is not None
    value = 0.0 if not has else float(np.clip(cents, -CENTS_RANGE, CENTS_RANGE))
    if not has:
        accent = "#8a8f98"
    elif abs(cents) <= in_tune_cents:
        accent = "#2ecc71"
    elif abs(cents) <= 20:
        accent = "#f1c40f"
    else:
        accent = "#e74c3c"

    fig = go.Figure()

    # Colored background zones (flat red | warn | in-tune green | warn | sharp red).
    zones = [
        (-CENTS_RANGE, -20, "#40232a"),
        (-20, -in_tune_cents, "#40391f"),
        (-in_tune_cents, in_tune_cents, "#1f4029"),
        (in_tune_cents, 20, "#40391f"),
        (20, CENTS_RANGE, "#40232a"),
    ]
    for x0, x1, color in zones:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1,
                      fillcolor=color, line_width=0, layer="below")

    # Center reference line and tick marks.
    for tick in (-50, -25, 0, 25, 50):
        fig.add_shape(type="line", x0=tick, x1=tick, y0=0, y1=1,
                      line={"color": "#5a606b", "width": 2 if tick == 0 else 1})

    # The moving marker.
    if has:
        fig.add_shape(type="line", x0=value, x1=value, y0=-0.05, y1=1.05,
                      line={"color": accent, "width": 6})
        fig.add_trace(go.Scatter(
            x=[value], y=[1.16], mode="markers",
            marker={"symbol": "triangle-down", "size": 20, "color": accent},
            hoverinfo="skip", showlegend=False,
        ))

    # Big note label.
    fig.add_annotation(x=0, y=0.5, text=f"<b>{note_label}</b>", showarrow=False,
                       font={"size": 54, "color": accent})

    # Cents readout + direction hint.
    if has:
        sign = "+" if value > 0 else ""
        direction = "IN TUNE" if abs(cents) <= in_tune_cents else ("TUNE DOWN ▼" if cents > 0 else "▲ TUNE UP")
        sub = f"{sign}{value:.1f}¢"
        if detected_hz and target_hz:
            sub += f"   ·   {detected_hz:.1f} Hz → {target_hz:.1f} Hz"
        fig.add_annotation(x=0, y=-0.32, text=sub, showarrow=False,
                           font={"size": 16, "color": "#c8ccd2"})
        fig.add_annotation(x=0, y=1.42, text=direction, showarrow=False,
                           font={"size": 18, "color": accent})

    # Flat / sharp edge labels.
    fig.add_annotation(x=-CENTS_RANGE, y=1.42, text="♭ flat", showarrow=False,
                       font={"size": 14, "color": "#8a8f98"}, xanchor="left")
    fig.add_annotation(x=CENTS_RANGE, y=1.42, text="sharp ♯", showarrow=False,
                       font={"size": 14, "color": "#8a8f98"}, xanchor="right")

    fig.update_xaxes(range=[-CENTS_RANGE - 2, CENTS_RANGE + 2], showgrid=False,
                     zeroline=False, showticklabels=True,
                     tickvals=[-50, -25, 0, 25, 50], ticktext=["-50", "-25", "0", "+25", "+50"],
                     tickfont={"color": "#8a8f98", "size": 11})
    fig.update_yaxes(range=[-0.5, 1.6], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(
        height=240, margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"}, showlegend=False,
    )
    return fig


def string_status_figure(targets, active_string: Optional[int], cents_by_string: dict) -> go.Figure:
    """
    Row of string "pills" showing each open string of the tuning; the active
    string is highlighted and tinted by how in-tune it is.
    """
    fig = go.Figure()
    n = len(targets)
    for idx, tgt in enumerate(targets):
        x = idx
        cents = cents_by_string.get(tgt.string_number)
        if tgt.string_number == active_string and cents is not None:
            if abs(cents) <= IN_TUNE_CENTS:
                color = "#2ecc71"
            elif abs(cents) <= 20:
                color = "#f1c40f"
            else:
                color = "#e74c3c"
            border = "white"
        else:
            color = "#2b3038"
            border = "#4a505a"
        fig.add_shape(type="circle", x0=x - 0.42, x1=x + 0.42, y0=-0.42, y1=0.42,
                      fillcolor=color, line={"color": border, "width": 2})
        fig.add_annotation(x=x, y=0, text=f"<b>{tgt.label}</b>", showarrow=False,
                           font={"size": 16, "color": "#f0f0f0"})
        fig.add_annotation(x=x, y=-0.62, text=f"{tgt.string_number}", showarrow=False,
                           font={"size": 11, "color": "#8a8f98"})

    fig.update_xaxes(range=[-0.7, n - 0.3], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-0.9, 0.7], showgrid=False, zeroline=False, showticklabels=False,
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(
        height=110, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"}, showlegend=False,
    )
    return fig


def spectrum_figure(
    freqs: Sequence[float],
    mags: Sequence[float],
    highlight_hz: Optional[float] = None,
    fmax: float = 2000.0,
    title: str = "Live FFT Spectrum",
) -> go.Figure:
    """
    Frequency-domain view. Optionally marks the frequency that drives the
    detected note with a vertical line, so you can see what leads to what.
    """
    freqs = np.asarray(freqs, dtype=float)
    mags = np.asarray(mags, dtype=float)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=freqs,
            y=mags,
            mode="lines",
            fill="tozeroy",
            line={"color": "#4aa3ff", "width": 1.5},
            name="Magnitude",
            hovertemplate="%{x:.1f} Hz<br>%{y:.2f}<extra></extra>",
        )
    )

    if highlight_hz and highlight_hz > 0:
        fig.add_vline(
            x=highlight_hz,
            line={"color": "#2ecc71", "width": 2, "dash": "dash"},
            annotation_text=f"f0 ≈ {highlight_hz:.1f} Hz",
            annotation_position="top",
            annotation_font_color="#2ecc71",
        )

    fig.update_layout(
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
        title={"text": title, "font": {"size": 15}},
        xaxis_title="Frequency (Hz)",
        yaxis_title="Normalized magnitude",
        xaxis={"range": [0, fmax]},
        yaxis={"range": [0, 1.05]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
        showlegend=False,
    )
    return fig


def fretboard_figure(frets, open_midis, title: str = "", n_frets: int = 5) -> go.Figure:
    """
    Draw a vertical guitar chord diagram. `frets` is thickest->thinnest with
    None = muted, 0 = open. Dots mark fretted notes; O/X above the nut.
    """
    n_strings = len(frets)
    fretted = [f for f in frets if f and f > 0]
    start_fret = max(min(fretted), 1) if fretted else 1
    if start_fret > 1 and max(fretted, default=0) - start_fret < n_frets - 1:
        start_fret = max(1, max(fretted, default=1) - (n_frets - 1))

    fig = go.Figure()

    # Fret lines (horizontal) and string lines (vertical).
    for r in range(n_frets + 1):
        fig.add_shape(type="line", x0=0, x1=n_strings - 1, y0=r, y1=r,
                      line={"color": "#888", "width": 3 if (start_fret == 1 and r == 0) else 1})
    for s in range(n_strings):
        fig.add_shape(type="line", x0=s, x1=s, y0=0, y1=n_frets,
                      line={"color": "#888", "width": 1})

    # String index 0 is the thickest (6th). Draw thickest on the left.
    dot_x, dot_y, markers, mx = [], [], [], []
    for s, f in enumerate(frets):
        if f is None:
            markers.append("✕")
            mx.append(s)
        elif f == 0:
            markers.append("○")
            mx.append(s)
        else:
            rel = f - start_fret + 0.5
            dot_x.append(s)
            dot_y.append(rel)

    if dot_x:
        fig.add_trace(go.Scatter(
            x=dot_x, y=dot_y, mode="markers",
            marker={"size": 22, "color": "#2ecc71", "line": {"color": "white", "width": 1}},
            hoverinfo="skip", showlegend=False,
        ))
    for s, m in zip(mx, markers):
        fig.add_annotation(x=s, y=-0.4, text=m, showarrow=False,
                           font={"color": "#e0e0e0", "size": 16})

    fig.update_yaxes(range=[n_frets, -0.8], showticklabels=(start_fret > 1),
                     tickvals=[i + 0.5 for i in range(n_frets)],
                     ticktext=[str(start_fret + i) for i in range(n_frets)])
    fig.update_xaxes(range=[-0.6, n_strings - 0.4], showticklabels=False)
    fig.update_layout(
        height=260, width=200, title={"text": title, "font": {"size": 14}},
        margin=dict(l=30, r=10, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"}, showlegend=False,
    )
    return fig


def range_progress_figure(low_hz: float, high_hz: float, current_hz: float) -> go.Figure:
    """Horizontal bar showing running vocal range with a live pitch marker."""
    fig = go.Figure()
    if low_hz > 0 and high_hz > low_hz:
        fig.add_trace(
            go.Bar(
                x=[high_hz - low_hz],
                y=["Range"],
                base=[low_hz],
                orientation="h",
                marker={"color": "#4aa3ff"},
                hovertemplate="%{base:.1f}–%{x:.1f} Hz<extra></extra>",
            )
        )
    if current_hz and current_hz > 0:
        fig.add_vline(
            x=current_hz,
            line={"color": "#2ecc71", "width": 3},
            annotation_text=f"{current_hz:.0f} Hz",
            annotation_position="top",
        )
    fig.update_layout(
        height=140,
        margin=dict(l=20, r=20, t=30, b=30),
        xaxis_title="Frequency (Hz)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
        showlegend=False,
    )
    return fig

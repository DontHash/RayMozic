"""Plotly figures — minimal, readable, tool-like (not dashboard-flashy)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

CENTS_RANGE = 50.0
IN_TUNE_CENTS = 5.0

# Restrained palette
C_TEXT = "#d0d0d0"
C_MUTED = "#7a7a7a"
C_GRID = "#2e2e2e"
C_LINE = "#6a8fb5"
C_FILL = "rgba(106, 143, 181, 0.18)"
C_OK = "#5d8a6a"
C_WARN = "#a68b4b"
C_BAD = "#9e5a5a"
C_MARK = "#c4a35a"
C_BG = "rgba(0,0,0,0)"


def _layout(fig: go.Figure, *, height: int, margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=margin or dict(l=48, r=16, t=12, b=36),
        paper_bgcolor=C_BG,
        plot_bgcolor=C_BG,
        font={"family": "Inter, system-ui, sans-serif", "color": C_TEXT, "size": 12},
        showlegend=False,
    )
    return fig


def _axis(fig: go.Figure, x_title: str = "", y_title: str = "") -> go.Figure:
    fig.update_xaxes(
        title=x_title,
        title_font={"size": 11, "color": C_MUTED},
        tickfont={"size": 10, "color": C_MUTED},
        gridcolor=C_GRID,
        gridwidth=1,
        zeroline=False,
        linecolor=C_GRID,
        mirror=False,
    )
    fig.update_yaxes(
        title=y_title,
        title_font={"size": 11, "color": C_MUTED},
        tickfont={"size": 10, "color": C_MUTED},
        gridcolor=C_GRID,
        gridwidth=1,
        zeroline=False,
        linecolor=C_GRID,
        mirror=False,
    )
    return fig


def _tune_color(cents: Optional[float]) -> str:
    if cents is None:
        return C_MUTED
    a = abs(cents)
    if a <= IN_TUNE_CENTS:
        return C_OK
    if a <= 20:
        return C_WARN
    return C_BAD


def needle_figure(
    cents: Optional[float],
    note_label: str = "--",
    detail: str = "",
    in_tune_cents: float = IN_TUNE_CENTS,
) -> go.Figure:
    value = 0.0 if cents is None else float(np.clip(cents, -CENTS_RANGE, CENTS_RANGE))
    color = _tune_color(cents)
    title = f"{note_label}"
    if detail:
        title += f"<br><span style='font-size:11px;color:{C_MUTED}'>{detail}</span>"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": " ¢", "font": {"size": 22, "color": C_TEXT}},
            title={"text": title, "font": {"size": 26, "color": color}},
            gauge={
                "axis": {
                    "range": [-CENTS_RANGE, CENTS_RANGE],
                    "tickmode": "array",
                    "tickvals": [-50, -25, 0, 25, 50],
                    "ticktext": ["-50", "-25", "0", "+25", "+50"],
                    "tickfont": {"size": 9, "color": C_MUTED},
                },
                "bar": {"color": color, "thickness": 0.18},
                "bgcolor": C_BG,
                "borderwidth": 0,
                "steps": [
                    {"range": [-CENTS_RANGE, -in_tune_cents], "color": "rgba(158,90,90,0.12)"},
                    {"range": [-in_tune_cents, in_tune_cents], "color": "rgba(93,138,106,0.15)"},
                    {"range": [in_tune_cents, CENTS_RANGE], "color": "rgba(158,90,90,0.12)"},
                ],
            },
        )
    )
    return _layout(fig, height=220, margin=dict(l=24, r=24, t=48, b=8))


def tuner_meter_figure(
    cents: Optional[float],
    note_label: str = "--",
    target_hz: Optional[float] = None,
    detected_hz: Optional[float] = None,
    in_tune_cents: float = IN_TUNE_CENTS,
) -> go.Figure:
    has = cents is not None
    value = 0.0 if not has else float(np.clip(cents, -CENTS_RANGE, CENTS_RANGE))
    accent = _tune_color(cents)

    fig = go.Figure()
    fig.add_shape(type="rect", x0=-CENTS_RANGE, x1=CENTS_RANGE, y0=0.35, y1=0.65,
                  fillcolor="rgba(255,255,255,0.04)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=-in_tune_cents, x1=in_tune_cents, y0=0.35, y1=0.65,
                  fillcolor="rgba(93,138,106,0.2)", line_width=0, layer="below")
    fig.add_shape(type="line", x0=0, x1=0, y0=0.2, y1=0.8,
                  line={"color": C_MUTED, "width": 1, "dash": "dot"})

    if has:
        fig.add_shape(type="line", x0=value, x1=value, y0=0.15, y1=0.85,
                      line={"color": accent, "width": 3})

    note_y = 0.92
    fig.add_annotation(x=0, y=note_y, text=f"<b>{note_label}</b>", showarrow=False,
                       font={"size": 36, "color": accent if has else C_MUTED})

    if has:
        sign = "+" if value > 0 else ""
        hint = "in tune" if abs(cents) <= in_tune_cents else ("flat" if cents > 0 else "sharp")
        sub = f"{sign}{value:.1f} ¢ · {hint}"
        if detected_hz and target_hz:
            sub += f" · {detected_hz:.0f} / {target_hz:.0f} Hz"
        fig.add_annotation(x=0, y=0.08, text=sub, showarrow=False,
                           font={"size": 12, "color": C_MUTED})

    fig.update_xaxes(
        range=[-CENTS_RANGE - 4, CENTS_RANGE + 4],
        tickvals=[-50, -25, 0, 25, 50],
        ticktext=["♭50", "♭25", "0", "♯25", "♯50"],
        showgrid=False, zeroline=False,
    )
    fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False)
    return _layout(fig, height=160, margin=dict(l=12, r=12, t=8, b=8))


def string_status_figure(targets, active_string: Optional[int], cents_by_string: dict) -> go.Figure:
    labels = [t.label for t in targets]
    nums = [str(t.string_number) for t in targets]
    colors = []
    for t in targets:
        c = cents_by_string.get(t.string_number)
        if t.string_number == active_string and c is not None:
            colors.append(_tune_color(c))
        else:
            colors.append("#3a3a3a")

    fig = go.Figure(go.Bar(
        x=labels, y=[1] * len(labels),
        marker={"color": colors, "line": {"width": 0}},
        hoverinfo="skip",
        width=0.55,
    ))
    for i, (lab, num) in enumerate(zip(labels, nums)):
        fig.add_annotation(x=lab, y=0.5, text=lab, showarrow=False,
                           font={"size": 13, "color": C_TEXT})
        fig.add_annotation(x=lab, y=-0.15, text=num, showarrow=False,
                           font={"size": 9, "color": C_MUTED})

    fig.update_yaxes(visible=False, range=[-0.3, 1.2])
    fig.update_xaxes(showgrid=False, showticklabels=False)
    return _layout(fig, height=72, margin=dict(l=8, r=8, t=4, b=28))


def spectrum_figure(
    freqs: Sequence[float],
    mags: Sequence[float],
    highlight_hz: Optional[float] = None,
    fmax: float = 2000.0,
    title: str = "",
    extra_markers_hz: Optional[Sequence[float]] = None,
) -> go.Figure:
    freqs = np.asarray(freqs, dtype=float)
    mags = np.asarray(mags, dtype=float)
    mask = freqs <= fmax
    freqs, mags = freqs[mask], mags[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=freqs, y=mags, mode="lines",
        line={"color": C_LINE, "width": 1.2},
        fill="tozeroy", fillcolor=C_FILL,
        hovertemplate="%{x:.0f} Hz<extra></extra>",
    ))

    if highlight_hz and highlight_hz > 0 and highlight_hz <= fmax:
        fig.add_vline(x=highlight_hz, line={"color": C_OK, "width": 1.5})
        fig.add_annotation(
            x=highlight_hz, y=1.02, yref="paper", text=f"{highlight_hz:.0f} Hz",
            showarrow=False, font={"size": 10, "color": C_OK}, xanchor="center",
        )

    for hz in extra_markers_hz or []:
        if 0 < hz <= fmax:
            fig.add_vline(x=hz, line={"color": C_MARK, "width": 1, "dash": "dot"})

    fig.update_xaxes(range=[0, fmax])
    fig.update_yaxes(range=[0, max(1.05, float(mags.max()) * 1.1) if len(mags) else 1.05], showticklabels=False)
    _axis(fig, x_title="Hz")
    if title:
        fig.update_layout(title={"text": title, "font": {"size": 12, "color": C_MUTED}, "x": 0})
    return _layout(fig, height=200)


def capo_match_figure(rows: list[dict], highlight_fret: Optional[int] = None) -> go.Figure:
    """Bar chart: voice-match score per capo fret."""
    by_fret = sorted(rows, key=lambda r: r["capo_fret"])
    frets = [r["capo_fret"] for r in by_fret]
    scores = [r["match_score"] * 100 for r in by_fret]
    colors = []
    for r in by_fret:
        if r["capo_fret"] == highlight_fret:
            colors.append(C_OK)
        elif r.get("matches_voice"):
            colors.append(C_LINE)
        else:
            colors.append("#404040")

    fig = go.Figure(go.Bar(x=frets, y=scores, marker_color=colors, width=0.7,
                             hovertemplate="Capo %{x}<br>%{y:.0f}%<extra></extra>"))
    fig.update_xaxes(dtick=1, title="Capo fret")
    fig.update_yaxes(range=[0, 105], ticksuffix="%", title="Voice match")
    _axis(fig)
    return _layout(fig, height=200)


def fretboard_figure(frets, open_midis, title: str = "", n_frets: int = 5) -> go.Figure:
    n_strings = len(frets)
    fretted = [f for f in frets if f and f > 0]
    start_fret = max(min(fretted), 1) if fretted else 1
    if start_fret > 1 and max(fretted, default=0) - start_fret < n_frets - 1:
        start_fret = max(1, max(fretted, default=1) - (n_frets - 1))

    fig = go.Figure()
    nut_w = 2 if start_fret == 1 else 1
    for r in range(n_frets + 1):
        fig.add_shape(type="line", x0=-0.15, x1=n_strings - 0.85, y0=r, y1=r,
                      line={"color": "#555", "width": nut_w if r == 0 else 1})
    for s in range(n_strings):
        w = 2.5 if s == 0 else (1.5 if s == n_strings - 1 else 1)
        fig.add_shape(type="line", x0=s, x1=s, y0=0, y1=n_frets,
                      line={"color": "#555", "width": w})

    dot_x, dot_y, mx, markers = [], [], [], []
    for s, f in enumerate(frets):
        if f is None:
            markers.append("×")
            mx.append(s)
        elif f == 0:
            markers.append("○")
            mx.append(s)
        else:
            dot_x.append(s)
            dot_y.append(f - start_fret + 0.5)

    if dot_x:
        fig.add_trace(go.Scatter(
            x=dot_x, y=dot_y, mode="markers",
            marker={"size": 16, "color": C_TEXT, "line": {"width": 0}},
            hoverinfo="skip",
        ))
    for s, m in zip(mx, markers):
        fig.add_annotation(x=s, y=-0.35, text=m, showarrow=False,
                           font={"size": 12, "color": C_MUTED})

    if title:
        fig.add_annotation(x=(n_strings - 1) / 2, y=n_frets + 0.35, text=title, showarrow=False,
                           font={"size": 12, "color": C_TEXT})

    fig.update_yaxes(range=[n_frets + 0.5, -0.55], showticklabels=start_fret > 1,
                     tickvals=[i + 0.5 for i in range(n_frets)],
                     ticktext=[str(start_fret + i) for i in range(n_frets)],
                     tickfont={"size": 9, "color": C_MUTED})
    fig.update_xaxes(range=[-0.35, n_strings - 0.65], showticklabels=False, showgrid=False)
    fig.update_yaxes(showgrid=False)
    return _layout(fig, height=200, margin=dict(l=16, r=8, t=8, b=16))


def range_progress_figure(low_hz: float, high_hz: float, current_hz: float) -> go.Figure:
    fig = go.Figure()
    xmin = max(50, low_hz * 0.9) if low_hz > 0 else 80
    xmax = high_hz * 1.1 if high_hz > 0 else 500

    if low_hz > 0 and high_hz > low_hz:
        fig.add_shape(type="rect", x0=low_hz, x1=high_hz, y0=0.25, y1=0.75,
                      fillcolor=C_FILL, line={"width": 0})

    if current_hz and current_hz > 0:
        fig.add_vline(x=current_hz, line={"color": C_OK, "width": 2})
        fig.add_annotation(x=current_hz, y=0.9, text=f"{current_hz:.0f}", showarrow=False,
                           font={"size": 10, "color": C_OK})

    fig.update_xaxes(range=[xmin, xmax], title="Hz")
    fig.update_yaxes(visible=False, range=[0, 1])
    _axis(fig)
    return _layout(fig, height=72, margin=dict(l=48, r=16, t=4, b=28))

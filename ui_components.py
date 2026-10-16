# ui_components.py
"""Styling and HTML building blocks for the Streamlit dashboard."""
import html
import re

SEVERITY_ORDER = ["Minor", "Serious", "Fatal"]

SEVERITY_STYLE = {
    "Minor": {
        "color": "#22C55E",
        "headline": "Low risk",
        "desc": "Expected damage limited to property or minor abrasions.",
    },
    "Serious": {
        "color": "#F59E0B",
        "headline": "Elevated risk",
        "desc": "Probable hospitalisation or severe structural compromise.",
    },
    "Fatal": {
        "color": "#EF4444",
        "headline": "Critical risk",
        "desc": "High probability of fatal trauma. Emergency mitigation required.",
    },
}

ADVISORY_TONES = {
    "speed": "warn",
    "slippery": "info",
    "hazard": "danger",
    "critical": "danger",
    "clear": "ok",
}

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --bg: #0A0F1C;
    --surface: rgba(17, 26, 46, 0.72);
    --surface-solid: #111A2E;
    --border: rgba(148, 163, 184, 0.14);
    --border-strong: rgba(148, 163, 184, 0.28);
    --text: #E2E8F0;
    --muted: #94A3B8;
    --faint: #64748B;
    --accent: #22D3EE;
    --accent-2: #818CF8;
    --ok: #22C55E;
    --warn: #F59E0B;
    --danger: #EF4444;
    --info: #38BDF8;
}

.stApp {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(34, 211, 238, 0.10), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(129, 140, 248, 0.10), transparent 55%),
        var(--bg);
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stAppDeployButton"], .stDeployButton { display: none !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1480px; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1526 0%, #0A0F1C 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
.side-brand { display: flex; align-items: center; gap: 12px; margin: 4px 0 22px; }
.side-brand .logo {
    width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 8px 24px rgba(34, 211, 238, 0.25); font-size: 20px;
}
.side-brand .name { font-weight: 700; font-size: 15px; color: var(--text); line-height: 1.2; }
.side-brand .tag { font-size: 12px; color: var(--muted); }
.side-label {
    font-size: 11px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--faint); margin: 18px 0 6px; display: flex; align-items: center; gap: 8px;
}
.side-label::after { content: ""; flex: 1; height: 1px; background: var(--border); }
.side-hint { font-size: 12px; color: var(--muted); margin: -6px 0 10px; }
.side-hint b { color: var(--text); font-weight: 600; }

/* ---------- Hero ---------- */
.hero {
    display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; flex-wrap: wrap;
    padding: 4px 0 20px; border-bottom: 1px solid var(--border); margin-bottom: 20px;
}
.hero .eyebrow {
    display: inline-flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 600;
    color: var(--accent); letter-spacing: 0.08em; text-transform: uppercase;
}
.hero .eyebrow .live {
    width: 8px; height: 8px; border-radius: 50%; background: var(--ok);
    box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.6); animation: pulse 2s infinite;
}
.hero h1 {
    font-size: 34px; font-weight: 800; letter-spacing: -0.02em; margin: 6px 0 6px; padding: 0;
    background: linear-gradient(90deg, #F8FAFC 0%, #A5F3FC 60%, #C7D2FE 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { color: var(--muted); font-size: 15px; margin: 0; max-width: 720px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.chip {
    font-size: 12px; color: var(--muted); padding: 5px 10px; border-radius: 999px;
    border: 1px solid var(--border); background: rgba(15, 23, 42, 0.6);
}
.route-pill {
    display: flex; align-items: center; gap: 14px; padding: 12px 18px; border-radius: 16px;
    background: var(--surface); border: 1px solid var(--border-strong); backdrop-filter: blur(8px);
}
.route-pill .stop { display: flex; flex-direction: column; }
.route-pill .stop small { font-size: 11px; color: var(--faint); text-transform: uppercase; letter-spacing: 0.08em; }
.route-pill .stop span { font-weight: 600; color: var(--text); font-size: 15px; }
.route-pill .arrow { color: var(--accent); font-size: 18px; }
.route-pill .dist {
    font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--accent);
    padding-left: 14px; border-left: 1px solid var(--border);
}

/* ---------- KPI tiles ---------- */
.kpi {
    position: relative; overflow: hidden; padding: 16px 18px; border-radius: 16px;
    background: var(--surface); border: 1px solid var(--border); height: 100%;
}
.kpi::before {
    content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
    background: var(--kpi-accent, var(--accent)); opacity: 0.9;
}
.kpi .label { font-size: 12px; color: var(--muted); font-weight: 500; display: flex; align-items: center; gap: 8px; }
.kpi .value {
    font-size: clamp(18px, 1.9vw, 26px); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 700; color: var(--text); margin-top: 6px; letter-spacing: -0.01em;
    font-family: 'JetBrains Mono', monospace;
}
.kpi .value.text { font-family: 'Inter', sans-serif; }
.kpi .sub { font-size: 12px; color: var(--faint); margin-top: 2px; }

/* ---------- Panels ---------- */
.panel-title {
    display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 2px 12px;
    font-size: 13px; font-weight: 700; color: var(--text); letter-spacing: 0.02em; margin: 6px 0 10px;
}
.panel-title small { font-weight: 500; color: var(--faint); font-size: 12px; }
.card {
    padding: 18px; border-radius: 18px; background: var(--surface); border: 1px solid var(--border);
    margin-bottom: 14px;
}
.progress-meta {
    display: flex; justify-content: space-between; font-size: 12px; color: var(--muted);
    margin: -6px 0 8px; font-family: 'JetBrains Mono', monospace;
}
.progress-meta b { color: var(--text); }
iframe[title="streamlit_folium.st_folium"] {
    border-radius: 18px; border: 1px solid var(--border-strong);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.45);
}

/* ---------- Risk card ---------- */
.risk-card {
    padding: 20px; border-radius: 20px; margin-bottom: 14px;
    background:
        radial-gradient(400px 200px at 100% 0%, color-mix(in srgb, var(--sev) 22%, transparent), transparent 70%),
        var(--surface);
    border: 1px solid color-mix(in srgb, var(--sev) 45%, transparent);
    box-shadow: 0 0 0 1px rgba(0,0,0,0.2), 0 18px 50px color-mix(in srgb, var(--sev) 14%, transparent);
}
.risk-top { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; }
.risk-card .eyebrow { font-size: 11px; font-weight: 700; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; }
.risk-card .level { font-size: 34px; font-weight: 800; color: var(--sev); letter-spacing: -0.01em; line-height: 1.1; margin-top: 4px; }
.risk-card .headline { font-size: 14px; font-weight: 600; color: var(--text); margin-top: 6px; }
.risk-card .desc { font-size: 13px; color: var(--muted); margin-top: 2px; max-width: 280px; }
.ring { position: relative; width: 108px; height: 108px; flex-shrink: 0; }
.ring svg { transform: rotate(-90deg); }
.ring .center { position: absolute; inset: 0; display: grid; place-items: center; text-align: center; }
.ring .pct { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 700; color: var(--text); }
.ring .cap { font-size: 10px; color: var(--faint); text-transform: uppercase; letter-spacing: 0.1em; margin-top: -4px; }
.probs { margin-top: 18px; display: flex; flex-direction: column; gap: 10px; }
.prob-row { display: grid; grid-template-columns: 52px minmax(32px, 1fr) 46px; align-items: center; gap: 8px; font-size: 13px; }
.prob-row .name { color: var(--muted); }
.prob-row .name.active { color: var(--text); font-weight: 600; }
.prob-row .track { height: 8px; border-radius: 999px; background: rgba(148, 163, 184, 0.12); overflow: hidden; }
.prob-row .fill { height: 100%; border-radius: 999px; transition: width .4s ease; }
.prob-row .val { text-align: right; font-family: 'JetBrains Mono', monospace; color: var(--text); }

/* ---------- Advisories ---------- */
.adv {
    display: flex; gap: 12px; padding: 12px 14px; border-radius: 14px; margin-bottom: 8px;
    background: color-mix(in srgb, var(--tone) 9%, rgba(15, 23, 42, 0.7));
    border: 1px solid color-mix(in srgb, var(--tone) 30%, transparent);
    border-left: 3px solid var(--tone);
}
.adv .icon { font-size: 18px; line-height: 1.3; }
.adv .title { font-weight: 700; font-size: 13px; color: var(--text); }
.adv .body { font-size: 13px; color: var(--muted); margin-top: 2px; line-height: 1.45; }
.tone-ok { --tone: var(--ok); }
.tone-warn { --tone: var(--warn); }
.tone-danger { --tone: var(--danger); }
.tone-info { --tone: var(--info); }

/* ---------- Hazard timeline ---------- */
.timeline { display: flex; flex-direction: column; gap: 0; }
.tl-item { display: grid; grid-template-columns: 64px 18px 1fr auto; gap: 12px; align-items: start; padding: 10px 0; }
.tl-item + .tl-item { border-top: 1px dashed var(--border); }
.tl-km { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--muted); padding-top: 2px; }
.tl-dot { width: 12px; height: 12px; border-radius: 50%; margin-top: 4px; background: var(--dot); box-shadow: 0 0 0 4px color-mix(in srgb, var(--dot) 20%, transparent); }
.tl-name { font-weight: 600; font-size: 14px; color: var(--text); }
.tl-desc { font-size: 12.5px; color: var(--muted); margin-top: 2px; }
.badge {
    font-size: 11px; font-weight: 700; padding: 4px 9px; border-radius: 999px; white-space: nowrap;
    color: var(--b); background: color-mix(in srgb, var(--b) 14%, transparent);
    border: 1px solid color-mix(in srgb, var(--b) 35%, transparent);
}
.tl-item.passed { opacity: 0.45; }
.empty { text-align: center; padding: 22px 10px; color: var(--muted); font-size: 13px; }
.empty .big { font-size: 26px; margin-bottom: 6px; }

/* ---------- Model inputs ---------- */
.inputs-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
.input-cell { padding: 12px; border-radius: 12px; background: rgba(15, 23, 42, 0.6); border: 1px solid var(--border); }
.input-cell .k { font-size: 11px; color: var(--faint); text-transform: uppercase; letter-spacing: 0.08em; }
.input-cell .v { font-size: 14px; font-weight: 600; color: var(--text); margin-top: 3px; }
.footnote { font-size: 12px; color: var(--faint); margin-top: 10px; line-height: 1.5; }

/* ---------- Model performance ---------- */
.cm { display: grid; grid-template-columns: 92px repeat(var(--n), minmax(0, 1fr)); gap: 6px; align-items: stretch; }
.cm-corner { font-size: 10.5px; color: var(--faint); text-transform: uppercase; letter-spacing: 0.08em; display: flex; align-items: flex-end; line-height: 1.3; }
.cm-head { text-align: center; font-size: 12px; font-weight: 700; color: var(--muted); padding-bottom: 4px; }
.cm-rowlabel { display: flex; align-items: center; font-size: 12px; font-weight: 700; color: var(--muted); }
.cm-cell {
    display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
    padding: 14px 6px; border-radius: 12px; min-height: 74px;
    background: color-mix(in srgb, var(--tone) calc(6% + var(--s) * 64%), rgba(15, 23, 42, 0.6));
    border: 1px solid color-mix(in srgb, var(--tone) calc(10% + var(--s) * 40%), transparent);
}
.cm-cell b { font-family: 'JetBrains Mono', monospace; font-size: 17px; color: var(--text); }
.cm-cell span { font-size: 11px; color: var(--muted); margin-top: 2px; }
.cm-cell.ok { --tone: var(--ok); }
.cm-cell.bad { --tone: var(--danger); }
.cm-legend { display: flex; gap: 16px; font-size: 12px; color: var(--muted); margin-top: 12px; }
.cm-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.perf-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.perf-table th, .perf-table td { border-left: 0; border-right: 0; border-bottom: 0; background: transparent; }
.perf-table th { text-align: right; font-size: 11px; font-weight: 600; color: var(--faint); text-transform: uppercase; letter-spacing: 0.08em; padding: 0 0 8px; }
.perf-table th:first-child, .perf-table td:first-child { text-align: left; }
.perf-table td { text-align: right; padding: 9px 0; border-top: 1px dashed var(--border); color: var(--text); font-family: 'JetBrains Mono', monospace; }
.perf-table td:first-child { font-family: 'Inter', sans-serif; font-weight: 600; }
.perf-table .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; }
.note { padding: 12px 14px; border-radius: 14px; font-size: 13px; line-height: 1.5; color: var(--muted);
    background: rgba(56, 189, 248, 0.07); border: 1px solid rgba(56, 189, 248, 0.25); border-left: 3px solid var(--info); }
.note b { color: var(--text); }
.stat-line { display: flex; justify-content: space-between; gap: 12px; font-size: 13px; padding: 8px 0; border-top: 1px dashed var(--border); color: var(--muted); }
.stat-line b { color: var(--text); font-family: 'JetBrains Mono', monospace; font-weight: 600; }

/* ---------- Error / empty state ---------- */
.state-card {
    max-width: 640px; margin: 40px auto; text-align: center; padding: 36px 28px; border-radius: 22px;
    background: var(--surface); border: 1px solid color-mix(in srgb, var(--danger) 35%, transparent);
}
.state-card .big { font-size: 40px; }
.state-card h3 { margin: 10px 0 6px; color: var(--text); }
.state-card p { color: var(--muted); margin: 0 0 6px; }
.state-card code { color: var(--accent); }

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.6); }
    70% { box-shadow: 0 0 0 10px rgba(34, 197, 94, 0); }
    100% { box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
}

@media (max-width: 900px) {
    .hero h1 { font-size: 26px; }
    .route-pill { width: 100%; justify-content: space-between; }
}
</style>
"""

# Styles injected into the Folium map document (it renders in its own iframe).
MAP_CSS = """
<style>
.leaflet-container { background: #0A0F1C; font-family: Inter, system-ui, sans-serif; }
.pin {
    width: 30px; height: 30px; border-radius: 50% 50% 50% 0; transform: rotate(-45deg);
    display: grid; place-items: center; border: 2px solid #fff; box-shadow: 0 6px 16px rgba(0,0,0,.45);
}
.pin span { transform: rotate(45deg); color: #fff; font: 700 13px Inter, sans-serif; }
.pin-start { background: #22C55E; }
.pin-end { background: #818CF8; }
.vehicle {
    width: 18px; height: 18px; border-radius: 50%; background: #22D3EE; border: 3px solid #fff;
    box-shadow: 0 0 0 0 rgba(34,211,238,.7); animation: vpulse 1.8s infinite;
}
@keyframes vpulse {
    0% { box-shadow: 0 0 0 0 rgba(34,211,238,.7); }
    70% { box-shadow: 0 0 0 18px rgba(34,211,238,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,211,238,0); }
}
.leaflet-tooltip {
    background: #111A2E; color: #E2E8F0; border: 1px solid rgba(148,163,184,.3);
    border-radius: 8px; box-shadow: 0 8px 20px rgba(0,0,0,.4); font-size: 12px;
}
.leaflet-tooltip-top:before { border-top-color: #111A2E; }
.leaflet-control-layers { border-radius: 10px !important; }
.leaflet-container .leaflet-control-attribution {
    background: rgba(10, 15, 28, 0.75) !important; color: #64748B !important;
    font-size: 10px !important; border-top-left-radius: 8px;
}
.leaflet-control-attribution a { color: #94A3B8 !important; }
</style>
"""

esc = html.escape


def severity_color(severity):
    return SEVERITY_STYLE.get(severity, {}).get("color", "#94A3B8")


def side_brand():
    return """
    <div class="side-brand">
        <div class="logo">🛡️</div>
        <div><div class="name">Road Safety Co-Pilot</div><div class="tag">AI risk navigation · India</div></div>
    </div>"""


def side_label(text):
    return f'<div class="side-label">{esc(text)}</div>'


def hero(start, end, distance_km=None):
    dist = f'<div class="dist">{distance_km:,.1f} km</div>' if distance_km is not None else ""
    return f"""
    <div class="hero">
        <div>
            <div class="eyebrow"><span class="live"></span>Live risk co-pilot</div>
            <h1>AI Road Safety Navigation</h1>
            <p>Route-level accident severity forecasting from historical crash records,
            live weather telemetry and a decision-tree classifier.</p>
            <div class="chips">
                <span class="chip">🧠 Decision Tree model</span>
                <span class="chip">🌦️ Open-Meteo live weather</span>
                <span class="chip">🛣️ OSRM routing</span>
                <span class="chip">📍 Blackspot proximity scan</span>
            </div>
        </div>
        <div class="route-pill">
            <div class="stop"><small>From</small><span>{esc(start)}</span></div>
            <div class="arrow">➜</div>
            <div class="stop"><small>To</small><span>{esc(end)}</span></div>
            {dist}
        </div>
    </div>"""


def kpi(label, value, sub="", accent=None, text_value=False):
    style = f' style="--kpi-accent:{accent}"' if accent else ""
    value_cls = "value text" if text_value else "value"
    return f"""
    <div class="kpi"{style}>
        <div class="label">{label}</div>
        <div class="{value_cls}">{esc(str(value))}</div>
        <div class="sub">{esc(sub)}</div>
    </div>"""


def panel_title(title, note=""):
    note_html = f"<small>{esc(note)}</small>" if note else ""
    return f'<div class="panel-title"><span>{title}</span>{note_html}</div>'


def risk_card(severity, prob_by_class):
    style = SEVERITY_STYLE.get(severity, {"color": "#94A3B8", "headline": "", "desc": ""})
    confidence = prob_by_class.get(severity, 0.0)
    radius = 46
    circumference = 2 * 3.14159265 * radius
    dash = circumference * confidence

    ordered = [c for c in SEVERITY_ORDER if c in prob_by_class]
    ordered += [c for c in prob_by_class if c not in ordered]
    rows = "".join(
        f"""
        <div class="prob-row">
            <div class="name{' active' if cls == severity else ''}">{esc(cls)}</div>
            <div class="track"><div class="fill" style="width:{prob_by_class[cls] * 100:.1f}%;background:{severity_color(cls)}"></div></div>
            <div class="val">{prob_by_class[cls] * 100:.1f}%</div>
        </div>"""
        for cls in ordered
    )

    return f"""
    <div class="risk-card" style="--sev:{style['color']}">
        <div class="risk-top">
            <div>
                <div class="eyebrow">Predicted severity</div>
                <div class="level">{esc(severity.upper())}</div>
                <div class="headline">{esc(style['headline'])}</div>
                <div class="desc">{esc(style['desc'])}</div>
            </div>
            <div class="ring">
                <svg width="108" height="108" viewBox="0 0 108 108">
                    <circle cx="54" cy="54" r="{radius}" fill="none" stroke="rgba(148,163,184,0.15)" stroke-width="10"/>
                    <circle cx="54" cy="54" r="{radius}" fill="none" stroke="{style['color']}" stroke-width="10"
                        stroke-linecap="round" stroke-dasharray="{dash:.1f} {circumference:.1f}"/>
                </svg>
                <div class="center"><div><div class="pct">{confidence * 100:.0f}%</div><div class="cap">confidence</div></div></div>
            </div>
        </div>
        <div class="probs">{rows}</div>
    </div>"""


_ADVISORY_RE = re.compile(r"^\s*(\S+)\s+\*\*(.+?):?\*\*:?\s*(.*)$", re.DOTALL)


def advisory_card(text):
    """Renders a navigation_engine advisory string ("<emoji> **Title:** body") as a card."""
    match = _ADVISORY_RE.match(text)
    if match:
        icon, title, body = match.groups()
    else:
        icon, title, body = "ℹ️", "Advisory", text
    tone = next((t for key, t in ADVISORY_TONES.items() if key in title.lower()), "info")
    return f"""
    <div class="adv tone-{tone}">
        <div class="icon">{esc(icon)}</div>
        <div><div class="title">{esc(title)}</div><div class="body">{esc(body.replace('**', ''))}</div></div>
    </div>"""


def hazard_timeline(route_hazards, covered_km):
    if not route_hazards:
        return """
        <div class="empty"><div class="big">✅</div>
        No known accident blackspots intersect this route.</div>"""

    items = []
    for spot in route_hazards:
        color = severity_color(spot.get("historical_severity"))
        start_km, end_km = spot["enter_km"], spot["exit_km"]
        if covered_km > end_km:
            cls, badge, badge_color = " passed", "Passed", "#64748B"
        elif covered_km >= start_km:
            cls, badge, badge_color = "", "You are here", color
        else:
            cls, badge, badge_color = "", f"in {start_km - covered_km:.1f} km", color
        items.append(f"""
        <div class="tl-item{cls}">
            <div class="tl-km">{spot['km']:.1f} km</div>
            <div class="tl-dot" style="--dot:{color}"></div>
            <div>
                <div class="tl-name">{esc(spot['name'])}</div>
                <div class="tl-desc">{esc(spot.get('historical_severity', 'Risk'))} zone · {esc(spot.get('primary_hazard', 'Historical accident spot'))}</div>
            </div>
            <span class="badge" style="--b:{badge_color}">{esc(badge)}</span>
        </div>""")
    return f'<div class="timeline">{"".join(items)}</div>'


def inputs_grid(cells):
    body = "".join(
        f'<div class="input-cell"><div class="k">{esc(k)}</div><div class="v">{esc(str(v))}</div></div>'
        for k, v in cells
    )
    return f'<div class="inputs-grid">{body}</div>'


CM_MODES = ("Counts", "% of actual (recall)", "% of predicted (precision)")


def confusion_matrix(labels, counts, recall, precision, mode):
    """HTML confusion matrix. Rows = actual severity, columns = predicted; diagonal = correct."""
    head = "".join(f'<div class="cm-head">{esc(l)}</div>' for l in labels)
    rows = []
    for i, actual in enumerate(labels):
        cells = [f'<div class="cm-rowlabel">{esc(actual)}</div>']
        for j in range(len(labels)):
            n = f"{counts[i][j]:,}"
            if mode == CM_MODES[1]:
                primary, secondary, strength = f"{recall[i][j] * 100:.1f}%", f"{n} cases", recall[i][j]
            elif mode == CM_MODES[2]:
                primary, secondary, strength = f"{precision[i][j] * 100:.1f}%", f"{n} cases", precision[i][j]
            else:
                primary, secondary, strength = n, f"{recall[i][j] * 100:.0f}% of actual", recall[i][j]
            tone = "ok" if i == j else "bad"
            cells.append(f'<div class="cm-cell {tone}" style="--s:{strength:.3f}"><b>{primary}</b><span>{secondary}</span></div>')
        rows.append("".join(cells))
    return (f'<div class="cm" style="--n:{len(labels)}"><div class="cm-corner">Actual ↓<br>Predicted →</div>{head}{"".join(rows)}</div>'
            '<div class="cm-legend"><span><i style="background:var(--ok)"></i>Correct (diagonal)</span>'
            '<span><i style="background:var(--danger)"></i>Misclassified</span></div>')


def per_class_table(labels, per_class):
    rows = "".join(
        f'<tr><td><span class="dot" style="background:{severity_color(lab)}"></span>{esc(lab)}</td>'
        f'<td>{per_class[lab]["precision"] * 100:.1f}%</td><td>{per_class[lab]["recall"] * 100:.1f}%</td>'
        f'<td>{per_class[lab]["f1"] * 100:.1f}%</td><td>{per_class[lab]["support"]:,}</td></tr>'
        for lab in labels
    )
    return ('<table class="perf-table"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Cases</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def stat_line(label, value):
    return f'<div class="stat-line"><span>{esc(label)}</span><b>{esc(value)}</b></div>'


def note(markup):
    return f'<div class="note">{markup}</div>'


def error_state(title, message):
    return f"""
    <div class="state-card">
        <div class="big">🧭</div>
        <h3>{esc(title)}</h3>
        <p>{esc(message)}</p>
        <p style="font-size:13px">Try a more specific place name, e.g. <code>Kochi, Kerala</code> or <code>Connaught Place, Delhi</code>.</p>
    </div>"""


def hour_label(hour):
    suffix = "AM" if hour < 12 else "PM"
    h12 = hour % 12 or 12
    if 5 <= hour < 12:
        phase = "🌅 Morning"
    elif 12 <= hour < 17:
        phase = "☀️ Afternoon"
    elif 17 <= hour < 20:
        phase = "🌇 Evening"
    else:
        phase = "🌙 Night"
    return f"{h12}:00 {suffix}", phase


def pretty_vehicle(name):
    return str(name).replace("?", "–")

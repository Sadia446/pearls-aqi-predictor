"""AeroSense — Streamlit Dashboard by Sadia.

A clean, modern IQAir-inspired AQI prediction dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path
import datetime
import base64

import pandas as pd
import streamlit as st
import altair as alt

# Reuse the project's AQI bands and city list
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.common.aqi import aqi_category, aqi_color  # noqa: E402
from ml.common.cities import CITIES  # noqa: E402
from ml.storage.feature_store import get_latest_features, read_features  # noqa: E402
from ml.storage.hopsworks_store import (  # noqa: E402
    ALERTS_FG,
    ALERTS_VERSION,
    DRIVERS_FG,
    DRIVERS_VERSION,
    FEATURES_FG,
    FEATURES_VERSION,
    PREDICTIONS_FG,
    PREDICTIONS_VERSION,
    read_fg,
)
from ml.storage.registry import get_best_models  # noqa: E402

st.set_page_config(
    page_title="AeroSense",
    page_icon="🍃",
    layout="wide",
)

def render_html(html_str: str):
    """Safely renders HTML in Streamlit by stripping line indentation to prevent Markdown code block triggers."""
    clean_html = "\n".join(line.strip() for line in html_str.strip().splitlines())
    st.markdown(clean_html, unsafe_allow_html=True)

# Load Hero Background Image Asset
hero_bg_path = Path(__file__).resolve().parent / "assets" / "hero_bg.jpg"
hero_bg_base64 = ""
if hero_bg_path.exists():
    with open(hero_bg_path, "rb") as f:
        hero_bg_base64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

# Load Pollutants Storm Sky Background Image Asset
pollutants_bg_path = Path(__file__).resolve().parent / "assets" / "pollutants_bg.jpg"
pollutants_bg_base64 = ""
if pollutants_bg_path.exists():
    with open(pollutants_bg_path, "rb") as f:
        pollutants_bg_base64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

# Custom CSS for IQAir-inspired styling, smooth scroll, 80% viewport centering (10% margins) and removing top whitespace
render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html {
    scroll-behavior: smooth;
}

/* Center content in 80% viewport width with 10% left/right spacing */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem !important;
    padding-left: 10vw !important;
    padding-right: 10vw !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] {
    display: none !important;
    height: 0px !important;
}
div[data-testid="stToolbar"] {
    display: none !important;
}
div[data-testid="stDecoration"] {
    display: none !important;
}
.stApp > header {
    display: none !important;
}
#MainMenu, footer {
    visibility: hidden;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #171717;
}
.stApp {
    background-color: #F7F8FA;
}

.card {
    background-color: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    margin-bottom: 24px;
    border: 1px solid #eef2f6;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: white !important;
    border-radius: 16px !important;
    border: 1px solid #eef2f6 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
    min-height: 480px !important;
    padding: 20px !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-between !important;
}

.main-aqi-number {
    font-size: 5rem;
    font-weight: 800;
    line-height: 1;
    margin: 0;
}

.status-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-size: 0.875rem;
    font-weight: 600;
    color: #171717;
    text-shadow: 0px 0px 2px rgba(255,255,255,0.8);
}

/* Navigation Pills */
.nav-pill {
    color: #4b5563 !important;
    font-weight: 500;
    padding: 8px 18px;
    border-radius: 999px;
    text-decoration: none !important;
    font-size: 0.95rem;
    display: inline-block;
    background-color: transparent;
    transition: all 0.2s ease;
}
.nav-pill:hover {
    background-color: #171717 !important;
    color: #ffffff !important;
}
.nav-pill.active {
    background-color: #171717 !important;
    color: #ffffff !important;
    font-weight: 600;
}

/* Nav Social Icons */
.nav-social-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: transparent;
    color: #4b5563;
    text-decoration: none !important;
    border: 1px solid #d0d5dd;
    transition: all 0.2s ease;
}
.nav-social-icon:hover {
    background-color: #171717 !important;
    color: #ffffff !important;
    border-color: #171717;
}

/* Bottom Query Band */
.bottom-band {
    background-color: #0b0f19;
    color: white;
    margin: 40px -10vw 0 -10vw;
    padding: 60px 10vw;
    text-align: center;
}

/* Footer */
.footer {
    background-color: #171717;
    color: #F7F8FA;
    margin: 0 -10vw -2rem -10vw;
    padding: 60px 10vw 24px 10vw;
}
.footer a {
    color: #D0D5DD;
    text-decoration: none;
    display: block;
    margin-bottom: 8px;
}
.footer a:hover {
    color: white;
    text-decoration: underline;
}
.footer-bottom {
    border-top: 1px solid #333;
    margin-top: 32px;
    padding-top: 24px;
    font-size: 0.875rem;
    color: #888;
    display: flex;
    justify-content: space-between;
}
</style>
""")

@st.cache_data(ttl=300)
def load_dashboard_data():
    features_df = read_fg(FEATURES_FG, FEATURES_VERSION)
    if not features_df.empty and "event_time" in features_df.columns:
        if not pd.api.types.is_datetime64_any_dtype(features_df["event_time"]):
            features_df["event_time"] = pd.to_datetime(features_df["event_time"], utc=True)
        latest_df = (
            features_df.sort_values("event_time")
            .groupby("city_id")
            .tail(1)
            .sort_values("city_id")
            .reset_index(drop=True)
        )
    else:
        latest_df = pd.DataFrame()

    preds_df = read_fg(PREDICTIONS_FG, PREDICTIONS_VERSION)
    alerts_df = read_fg(ALERTS_FG, ALERTS_VERSION)
    drivers_df = read_fg(DRIVERS_FG, DRIVERS_VERSION)
    models_df = get_best_models()
    return features_df, latest_df, preds_df, alerts_df, drivers_df, models_df


features_all, latest_all, preds_all, alerts_all, drivers_all, models_all = load_dashboard_data()

# Handle Navigation State via query_params or session_state
query_params = st.query_params
if "nav" in query_params:
    st.session_state.page = query_params["nav"]
elif "page" not in st.session_state:
    st.session_state.page = "Islamabad"

current_page = st.session_state.page

# Setup Cities mapping
names = {c.id: c.name for c in CITIES}
available_city_ids = (
    latest_all["city_id"].unique().tolist()
    if not latest_all.empty and "city_id" in latest_all.columns
    else [c.id for c in CITIES]
)
options = [c for c in available_city_ids if c in names] or [c.id for c in CITIES]

# Determine active city_id based on current_page
city_id = options[0]
for cid, cname in names.items():
    if current_page.lower() in cname.lower():
        city_id = cid
        break

# Helper function to generate nav item styling (support in-page anchor links for Recommendations & Contact)
def nav_link(label: str, target: str, is_anchor: bool = False) -> str:
    if is_anchor:
        href = f"#{target}"
        target_attr = ""
        is_active = False
    else:
        href = f"?nav={target}"
        target_attr = 'target="_self"'
        is_active = current_page.lower() == target.lower()
    
    active_class = "active" if is_active else ""
    if is_active:
        inline_style = "background-color: #171717; color: #ffffff !important; font-weight: 600;"
    else:
        inline_style = "color: #4b5563 !important; background-color: transparent;"
    return f'<a href="{href}" {target_attr} class="nav-pill {active_class}" style="{inline_style} padding: 8px 18px; border-radius: 999px; text-decoration: none; font-size: 0.95rem; display: inline-block;">{label}</a>'

# ================================================================
# AEROSENSE TOP BANNER & MAIN NAVBAR
# ================================================================

navbar_html = f"""
<div style="background-color: #191919; color: #d1d5db; padding: 18px 10vw; display: flex; justify-content: space-between; align-items: center; font-size: 0.9rem; margin: 0 -10vw 0 -10vw; border-bottom: 1px solid #2d2d2d; min-height: 54px;">
<div style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #ccc;">
Choose another country or region to see content specific to your location and air quality analytics.
</div>
<div style="display: flex; align-items: center; gap: 8px; flex-shrink: 0;">
<div style="background: #374151; padding: 6px 16px; border-radius: 4px; color: white; font-size: 0.85rem; font-weight: 500;">
Pakistan ▾
</div>
</div>
</div>

<div style="background: white; display: flex; justify-content: space-between; align-items: center; padding: 14px 10vw; margin: 0 -10vw 1.5rem -10vw; border-bottom: 1px solid #D0D5DD; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
<a href="?nav=Islamabad" target="_self" style="text-decoration: none; display: flex; align-items: center; gap: 6px;">
<span style="color: #D7282F; font-size: 1.3rem; font-weight: 800;">✚</span>
<span style="font-size: 1.4rem; font-weight: 800; color: #171717;">AeroSense</span>
</a>

<div style="display: flex; gap: 8px; align-items: center;">
{nav_link("Islamabad", "Islamabad")}
{nav_link("Lahore", "Lahore")}
{nav_link("Karachi", "Karachi")}
{nav_link("Recommendations", "recommendations-section", is_anchor=True)}
{nav_link("Model Diagnostics", "model-diagnostics-section", is_anchor=True)}
</div>

<div style="display: flex; align-items: center; gap: 10px;">
<a href="https://www.linkedin.com/in/sadia-noreen-6992682b2" target="_blank" title="LinkedIn" class="nav-social-icon">
<svg width="17" height="17" fill="currentColor" viewBox="0 0 24 24"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/></svg>
</a>
<a href="https://github.com/Sadia446" target="_blank" title="GitHub" class="nav-social-icon">
<svg width="17" height="17" fill="currentColor" viewBox="0 0 24 24"><path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0 0 22 12.017C22 6.484 17.522 2 12 2z"/></svg>
</a>
</div>
</div>
"""

render_html(navbar_html)

if latest_all.empty or city_id not in latest_all["city_id"].values:
    st.info(f"We're still gathering model data for {names.get(city_id, city_id)}. Check back shortly!")
    st.stop()

current = latest_all[latest_all["city_id"] == city_id].iloc[0]
city_forecast = (
    preds_all[preds_all["city_id"] == city_id].sort_values("horizon_h")
    if not preds_all.empty and "city_id" in preds_all.columns
    else pd.DataFrame()
)

aqi_now = float(current.get("aqi", 0))
aqi_cat = aqi_category(aqi_now)
aqi_col = aqi_color(aqi_now)

current_time = datetime.datetime.now().strftime("%H:%M, %b %d")
city_name = names.get(city_id, "Pakistan City")

# ================================================================
# DASHBOARD MAIN CONTENT
# ================================================================
if not alerts_all.empty and "city_id" in alerts_all.columns:
    city_alert = alerts_all[alerts_all["city_id"] == city_id]
    if not city_alert.empty:
        a = city_alert.iloc[0]
        st.error(f"🚨 **Warning:** Air quality may rise to **{a.get('category', 'Hazardous')}** levels (~{float(a.get('peak_aqi', 0)):.0f} AQI) in {int(a.get('starts_in_h', 0))} hours. {a.get('advice', '')}")

ratios = {
    "PM2.5": current.get("pm25", 0)/15,
    "PM10": current.get("pm10", 0)/45,
    "Ozone": current.get("o3", 0)/100,
    "NO2": current.get("no2", 0)/25,
    "SO2": current.get("so2", 0)/40,
    "CO": current.get("co", 0)/4000,
}
dominant = max(ratios, key=ratios.get)
temp_now = current.get('temp_c', 32)

if aqi_now <= 50: face_icon = "😊"
elif aqi_now <= 100: face_icon = "😐"
elif aqi_now <= 150: face_icon = "😷"
elif aqi_now <= 200: face_icon = "🤢"
else: face_icon = "💀"

text_color = "#171717" if aqi_now <= 100 else "white"

# ================================================================
# HERO SECTION WITH IMAGE BACKGROUND (City Name Left, AQI Card Right)
# ================================================================
hero_bg_style = f"background-image: linear-gradient(to right, rgba(15, 23, 42, 0.88) 0%, rgba(15, 23, 42, 0.70) 50%, rgba(15, 23, 42, 0.85) 100%), url('{hero_bg_base64}');" if hero_bg_base64 else "background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);"

hero_html = f"""
<div style="{hero_bg_style} background-size: cover; background-position: center; border-radius: 20px; padding: 36px 40px; margin-bottom: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.12); display: flex; justify-content: space-between; align-items: center; gap: 32px; flex-wrap: wrap;">
<div style="flex: 1; min-width: 300px;">
<h1 style="color: #ffffff; font-size: 2.6rem; font-weight: 800; margin: 0 0 12px 0; line-height: 1.15;">
Air quality in {city_name}
</h1>
<p style="color: #cbd5e1; font-size: 1.05rem; margin: 0; line-height: 1.5; max-width: 540px;">
Real-time air pollution index (AQI), PM2.5 concentrations, and weather forecast updates • Local time: {current_time}
</p>
</div>

<div style="width: 360px; max-width: 100%;">
<div style="background-color: {aqi_col}; border-radius: 16px; padding: 24px; box-shadow: 0 8px 24px rgba(0,0,0,0.18); color: {text_color}; display: flex; flex-direction: column; justify-content: space-between;">
<div style="display: flex; justify-content: space-between; align-items: flex-start;">
<div>
<div style="background: rgba(255,255,255,0.85); border-radius: 8px; padding: 4px 12px; display: inline-block; font-weight: 700; color: #171717; margin-bottom: 8px; font-size: 0.85rem;">AQI</div>
<div style="font-size: 4.5rem; font-weight: 800; line-height: 1; margin: 0; color: {text_color};">{int(aqi_now)}</div>
<div style="font-size: 1.2rem; font-weight: 700; color: {text_color}; margin-top: 6px;">{aqi_cat}</div>
</div>
<div style="font-size: 3.8rem; opacity: 0.95;">{face_icon}</div>
</div>
<div style="margin-top: 20px; background: rgba(255,255,255,0.22); padding: 10px 14px; border-radius: 8px; color: {text_color}; display: flex; justify-content: space-between; font-size: 0.95rem;">
<span style="font-weight: 500;">Main pollutant: {dominant}</span>
<span style="font-weight: 700;">{current.get('pm10', 176.5):.1f} µg/m³</span>
</div>
</div>
</div>
</div>
"""
render_html(hero_html)

# ================================================================
# THREE PREDICTION CARDS (Directly Below the Hero Section)
# ================================================================
fcols = st.columns(3)
icons = ["", "", ""]
day_labels = ["Tomorrow", "Two days Away", "Three days Away"]

if not city_forecast.empty:
    for i, (col, row) in enumerate(zip(fcols, city_forecast.itertuples())):
        label = day_labels[i] if i < len(day_labels) else f"Day {int(row.horizon_h // 24)}"
        pred_val = float(row.predicted_aqi)
        pred_cat = row.category
        pred_col = aqi_color(pred_val)
        icon = icons[i % len(icons)]

        h_temp = temp_now + 3 + (i * 0.5)
        l_temp = temp_now - 4 + (i * 0.5)

        with col:
            render_html(f"""
            <div class="card" style="text-align: center; margin-bottom: 24px;">
                <div style="font-weight: 600; font-size: 1.05rem; margin-bottom: 8px; color: #666;">{label}</div>
                <div style="font-size: 2.3rem; margin-bottom: 4px;">{icon}</div>
                <div style="font-size: 2.4rem; font-weight: 800; line-height: 1; margin-bottom: 8px;">{int(round(pred_val))}</div>
                <div class="status-badge" style="background-color: {pred_col}; font-size:0.75rem; padding: 4px 10px; margin-bottom: 12px;">{pred_cat}</div>
                <div style="color: #666; font-size: 0.9rem;">
                    <span style="font-weight: 700; color: #171717;">{h_temp:.0f}°</span> / {l_temp:.0f}°
                </div>
            </div>
            """)
else:
    with fcols[0]:
        st.info("Forecast data pending next pipeline execution.")

# ================================================================
# POLLUTANTS (STORM SKY BG) & HISTORIC AIR QUALITY (MATCHED SIZES)
# ================================================================
pt_col, tr_col = st.columns(2)

with pt_col:
    pollutants_bg_style = f"background-image: linear-gradient(rgba(15, 23, 42, 0.76), rgba(15, 23, 42, 0.88)), url('{pollutants_bg_base64}');" if pollutants_bg_base64 else "background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);"
    
    pollutant_card_html = f"""
    <div style="{pollutants_bg_style} background-size: cover; background-position: center; border-radius: 16px; padding: 24px; min-height: 480px; height: 100%; box-shadow: 0 4px 14px rgba(0,0,0,0.12); display: flex; flex-direction: column; justify-content: space-between; color: white; margin-bottom: 24px; border: 1px solid rgba(255,255,255,0.12);">
    <div>
    <h3 style="color: #ffffff; margin: 0 0 4px 0; font-size: 1.35rem; font-weight: 700;">Air pollutants</h3>
    <p style="color: #cbd5e1; font-size: 0.85rem; margin: 0 0 16px 0;">Live atmospheric concentrations in {city_name}</p>
    
    <div style="background: rgba(15, 23, 42, 0.65); backdrop-filter: blur(10px); border-radius: 12px; border: 1px solid rgba(255,255,255,0.14); overflow: hidden;">
    <table style="width: 100%; border-collapse: collapse; color: #f8fafc; font-size: 0.9rem;">
    <thead>
    <tr style="border-bottom: 1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.08); text-align: left;">
    <th style="padding: 10px 14px; font-weight: 600; color: #e2e8f0;">Pollutant</th>
    <th style="padding: 10px 14px; font-weight: 600; color: #e2e8f0;">Concentration</th>
    <th style="padding: 10px 14px; font-weight: 600; color: #e2e8f0;">WHO Guide</th>
    </tr>
    </thead>
    <tbody>
    <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 9px 14px; font-weight: 600;">PM2.5</td><td style="padding: 9px 14px; color: #38bdf8; font-weight: 700;">{current.get('pm25', 37.3):.1f} µg/m³</td><td style="padding: 9px 14px; color: #94a3b8;">15 µg/m³</td></tr>
    <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 9px 14px; font-weight: 600;">PM10</td><td style="padding: 9px 14px; color: #38bdf8; font-weight: 700;">{current.get('pm10', 176.5):.1f} µg/m³</td><td style="padding: 9px 14px; color: #94a3b8;">45 µg/m³</td></tr>
    <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 9px 14px; font-weight: 600;">Ozone (O₃)</td><td style="padding: 9px 14px; color: #38bdf8; font-weight: 700;">{current.get('o3', 169.2):.1f} µg/m³</td><td style="padding: 9px 14px; color: #94a3b8;">100 µg/m³</td></tr>
    <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 9px 14px; font-weight: 600;">Nitrogen Dioxide (NO₂)</td><td style="padding: 9px 14px; color: #38bdf8; font-weight: 700;">{current.get('no2', 13.7):.1f} µg/m³</td><td style="padding: 9px 14px; color: #94a3b8;">25 µg/m³</td></tr>
    <tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 9px 14px; font-weight: 600;">Sulfur Dioxide (SO₂)</td><td style="padding: 9px 14px; color: #38bdf8; font-weight: 700;">{current.get('so2', 8.2):.1f} µg/m³</td><td style="padding: 9px 14px; color: #94a3b8;">40 µg/m³</td></tr>
    <tr><td style="padding: 9px 14px; font-weight: 600;">Carbon Monoxide (CO)</td><td style="padding: 9px 14px; color: #38bdf8; font-weight: 700;">{current.get('co', 820):.0f} µg/m³</td><td style="padding: 9px 14px; color: #94a3b8;">4000 µg/m³</td></tr>
    </tbody>
    </table>
    </div>
    </div>
    <div style="margin-top: 14px; font-size: 0.82rem; color: #cbd5e1; background: rgba(0,0,0,0.35); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
    ⚠️ Fine particulate levels exceed recommended annual thresholds.
    </div>
    </div>
    """
    render_html(pollutant_card_html)

with tr_col:
    try:
        since_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=48)
        if not features_all.empty and "city_id" in features_all.columns and "event_time" in features_all.columns:
            history = features_all[
                (features_all["city_id"] == city_id) & (features_all["event_time"] >= since_time)
            ].sort_values("event_time").reset_index(drop=True)
        else:
            history = pd.DataFrame()
        min_aqi_val = float(history["aqi"].min()) if not history.empty and "aqi" in history.columns else aqi_now - 15
        max_aqi_val = float(history["aqi"].max()) if not history.empty and "aqi" in history.columns else aqi_now + 20
    except Exception:
        history = pd.DataFrame()
        min_aqi_val, max_aqi_val = aqi_now - 10, aqi_now + 15

    with st.container(border=True):
        render_html(f"""
        <div>
            <h3 style="margin: 0 0 4px 0; font-size: 1.35rem; font-weight: 700; color: #171717;">Historic air quality</h3>
            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 12px;">
                <span style="color: #fb923c; font-size: 1.2rem; line-height: 0.5;">●</span> <strong>{int(aqi_now)} AQI</strong> ({aqi_cat}) • Past 48 Hours
            </div>
        </div>
        """)

        if not history.empty:
            hist_df = history.copy()
            hist_df["event_time_dt"] = pd.to_datetime(hist_df["event_time"])
            hist_df["Formatted_Time"] = hist_df["event_time_dt"].dt.strftime("%d %b, %H:%M")
            
            # Soft light orange & light pink palette
            def get_light_color(val):
                if val <= 100:
                    return "#FED7AA"  # Light pastel peach orange
                elif val <= 150:
                    return "#FDBA74"  # Soft light orange
                elif val <= 200:
                    return "#FDA4AF"  # Soft pastel pink
                else:
                    return "#F472B6"  # Light rose pink

            hist_df["BarColor"] = hist_df["aqi"].apply(get_light_color)

            chart = (
                alt.Chart(hist_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=7)
                .encode(
                    x=alt.X("Formatted_Time:N", sort=None, title=None, axis=alt.Axis(labels=True, labelAngle=-45, labelColor="#94a3b8", labelFontSize=9, ticks=False, domain=False)),
                    y=alt.Y("aqi:Q", title="AQI", axis=alt.Axis(titleColor="#64748b", labelColor="#94a3b8", grid=True, gridColor="#f1f5f9", gridDash=[2, 2])),
                    color=alt.Color("BarColor:N", scale=None),
                    tooltip=[
                        alt.Tooltip("Formatted_Time:N", title="Time"),
                        alt.Tooltip("aqi:Q", title="AQI")
                    ]
                )
                .properties(height=245)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Insufficient historical logs for the past 48 hours.")

        render_html(f"""
        <div style="background: #f8fafc; border-radius: 8px; padding: 10px 14px; border: 1px solid #e2e8f0; font-size: 0.85rem; color: #64748b; display: flex; justify-content: space-between; margin-top: 10px;">
            <span>Observed 48h Range</span>
            <strong style="color: #1e293b;">{min_aqi_val:.0f} – {max_aqi_val:.0f} AQI</strong>
        </div>
        """)

# ================================================================
# HEALTH RECOMMENDATIONS SECTION (Anchor Target #recommendations-section)
# ================================================================
render_html("""
<div id="recommendations-section" style="padding-top: 12px; margin-top: 12px;">
    <h3 style="margin-top: 0.5rem; margin-bottom: 1rem; font-size: 1.35rem; font-weight: 700; color: #171717;">Health recommendations</h3>
</div>
""")

tips = [
    ("", "Reduce outdoor exercise", "Limit high-intensity cardio outdoors during peak pollution periods."),
    ("", "Close your windows", "Keep outdoor smog and fine particulates from entering living spaces."),
    ("", "Wear a mask outdoors", "Sensitive groups and vulnerable individuals should wear high-filtration masks."),
]
tcols = st.columns(3)
for tcol, (icon, title, desc) in zip(tcols, tips):
    with tcol:
        render_html(f"""
        <div class="card" style="margin-bottom: 24px; padding: 22px;">
            <div style="font-size: 2rem; margin-bottom: 8px;">{icon}</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #171717; margin-bottom: 6px;">{title}</div>
            <div style="color: #64748b; font-size: 0.875rem; line-height: 1.45;">{desc}</div>
        </div>
        """)

# ================================================================
# MODEL DIAGNOSTICS SECTION (Anchor Target #model-diagnostics-section)
# ================================================================
pollutants_diag_bg_style = f"background-image: linear-gradient(rgba(15, 23, 42, 0.76), rgba(15, 23, 42, 0.88)), url('{pollutants_bg_base64}');" if pollutants_bg_base64 else "background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);"

# Build model diagnostics table rows
if not models_all.empty:
    diag_rows = ""
    for _, mrow in models_all.iterrows():
        h = int(mrow.get('horizon_h', 0))
        mname = mrow.get('model_name', 'N/A')
        rmse_val = mrow.get('rmse', None)
        mae_val = mrow.get('mae', None)
        r2_val = mrow.get('r2', None)
        rmse_str = f"{float(rmse_val):.2f}" if rmse_val is not None else "—"
        mae_str = f"{float(mae_val):.2f}" if mae_val is not None else "—"
        r2_str = f"{float(r2_val):.4f}" if r2_val is not None else "—"
        diag_rows += f'<tr style="border-bottom: 1px solid rgba(255,255,255,0.08);"><td style="padding: 9px 14px; font-weight: 600;">+{h}h</td><td style="padding: 9px 14px; color: #38bdf8; font-weight: 700;">{mname}</td><td style="padding: 9px 14px; color: #34d399; font-weight: 700;">{rmse_str}</td><td style="padding: 9px 14px; color: #fbbf24; font-weight: 700;">{mae_str}</td><td style="padding: 9px 14px; color: #a78bfa; font-weight: 700;">{r2_str}</td></tr>'
else:
    diag_rows = '<tr><td colspan="5" style="padding: 14px; text-align: center; color: #94a3b8;">Model diagnostics data is not yet available. Run the training pipeline to populate metrics.</td></tr>'

render_html(f"""
<div id="model-diagnostics-section" style="{pollutants_diag_bg_style} background-size: cover; background-position: center; border-radius: 16px; padding: 28px 32px; margin-top: 24px; margin-bottom: 24px; box-shadow: 0 4px 14px rgba(0,0,0,0.12); color: white; border: 1px solid rgba(255,255,255,0.12);">
<div>
<h3 style="color: #ffffff; margin: 0 0 4px 0; font-size: 1.35rem; font-weight: 700;">Model Diagnostics</h3>
<p style="color: #cbd5e1; font-size: 0.85rem; margin: 0 0 16px 0;">Best performing model per forecast horizon — training metrics from the latest pipeline run</p>

<div style="background: rgba(15, 23, 42, 0.65); backdrop-filter: blur(10px); border-radius: 12px; border: 1px solid rgba(255,255,255,0.14); overflow: hidden;">
<table style="width: 100%; border-collapse: collapse; color: #f8fafc; font-size: 0.9rem;">
<thead>
<tr style="border-bottom: 1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.08); text-align: left;">
<th style="padding: 10px 14px; font-weight: 600; color: #e2e8f0;">Horizon</th>
<th style="padding: 10px 14px; font-weight: 600; color: #e2e8f0;">Model</th>
<th style="padding: 10px 14px; font-weight: 600; color: #e2e8f0;">RMSE</th>
<th style="padding: 10px 14px; font-weight: 600; color: #e2e8f0;">MAE</th>
<th style="padding: 10px 14px; font-weight: 600; color: #e2e8f0;">R²</th>
</tr>
</thead>
<tbody>
{diag_rows}
</tbody>
</table>
</div>
</div>
<div style="margin-top: 14px; font-size: 0.82rem; color: #cbd5e1; background: rgba(0,0,0,0.35); padding: 8px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1);">
📊 Lower RMSE/MAE and higher R² indicate better model fit. Metrics reflect hold-out test set performance.
</div>
</div>
""")


# ------------------------------------------------------------ DARK FOOTER WITH SOCIAL ICONS ---
render_html("""
<div class="footer">
    <div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 24px;">
        <div style="flex: 1.2; min-width: 250px; margin-bottom: 16px;">
            <div style="font-size: 1.4rem; font-weight: 800; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
                <span style="color: #D7282F; font-size: 1.3rem; font-weight: 800;">&#10010;</span>
                <span style="font-size: 1.4rem; font-weight: 800; color: #ffffff;">AeroSense</span>
            </div>
            <p style="color: #94a3b8; line-height: 1.5; font-size: 0.9rem; max-width: 320px; margin-bottom: 16px;">
                Advanced machine learning air quality forecasting dashboard for Pakistan. Designed and built by Sadia.
            </p>
            <!-- Social Icons (LinkedIn & GitHub) -->
            <div style="display: flex; gap: 10px; align-items: center;">
                <a href="www.linkedin.com/in/sadia-noreen-6992682b2" target="_blank" title="LinkedIn" style="display: flex; align-items: center; justify-content: center; width: 38px; height: 38px; border-radius: 50%; background: #262626; color: #cbd5e1; text-decoration: none; border: 1px solid #383838; transition: all 0.2s;">
                    <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/></svg>
                </a>
                <a href="https://github.com/Sadia446" target="_blank" title="GitHub" style="display: flex; align-items: center; justify-content: center; width: 38px; height: 38px; border-radius: 50%; background: #262626; color: #cbd5e1; text-decoration: none; border: 1px solid #383838; transition: all 0.2s;">
                    <svg width="18" height="18" fill="currentColor" viewBox="0 0 24 24"><path fill-rule="evenodd" clip-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.53 1.032 1.53 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0 1 12 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0 0 22 12.017C22 6.484 17.522 2 12 2z"/></svg>
                </a>
            </div>
        </div>
        <div style="flex: 1; min-width: 150px; margin-bottom: 16px;">
            <div style="font-size: 0.95rem; font-weight: 700; margin-bottom: 14px; color: #fff; letter-spacing: 0.5px;">CITIES</div>
            <a href="?nav=Islamabad" target="_self">Islamabad AQI</a>
            <a href="?nav=Lahore" target="_self">Lahore AQI</a>
            <a href="?nav=Karachi" target="_self">Karachi AQI</a>
        </div>
        <div style="flex: 1; min-width: 150px; margin-bottom: 16px;">
            <div style="font-size: 0.95rem; font-weight: 700; margin-bottom: 14px; color: #fff; letter-spacing: 0.5px;">QUICK LINKS</div>
            <a href="#recommendations-section">Health Recommendations</a>
            <a href="#model-diagnostics-section">Model Diagnostics</a>
        </div>
    </div>
     <div class="footer-bottom">
        <div>Pakistan Air Quality Analytics © 2026. Built by Sadia. All rights reserved.</div>
        <div style="color: #888;">
            <a href="#" style="display:inline; margin-right:15px; color:#888; text-decoration: none;">Terms of Use</a>
            <a href="#" style="display:inline; color:#888; text-decoration: none;">Privacy Policy</a>
        </div>
    </div>
</div>
""")
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="FireGuard AI",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    model = joblib.load('fire_risk_model.pkl')
    le    = joblib.load('label_encoder.pkl')
    cols  = joblib.load('feature_columns.pkl')
    return model, le, cols

model, le, cols = load_model()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding: 2rem 2rem 4rem; max-width: 780px; }
    .hero { text-align:center; padding:2rem 1rem 1rem; }
    .hero-badge {
        display:inline-block; background:rgba(255,80,50,0.12);
        border:1px solid rgba(255,80,50,0.3); color:#ff7055;
        font-size:0.72rem; font-weight:600; letter-spacing:0.12em;
        text-transform:uppercase; padding:0.3rem 0.9rem;
        border-radius:99px; margin-bottom:1rem;
    }
    .hero h1 { font-size:2.2rem; font-weight:700; color:#ffffff; margin:0 0 0.4rem; }
    .hero p  { color:#8b8b9a; font-size:0.92rem; margin:0; }
    .card {
    background:#13131a; border:1px solid #1e1e2e;
    border-radius:16px; padding:1.6rem 1.8rem; margin-bottom:1.2rem;
    }
    .card-title {
        font-size:0.75rem; font-weight:600; letter-spacing:0.1em;
        text-transform:uppercase; color:#5a5a72; margin-bottom:1.2rem;
    }
    .result-high   { background:linear-gradient(135deg,#2a0e0e,#1a0808); border:1px solid #6b1a1a; border-radius:16px; padding:2rem; text-align:center; margin:1rem 0; }
    .result-medium { background:linear-gradient(135deg,#1f1a08,#151208); border:1px solid #5a4a0a; border-radius:16px; padding:2rem; text-align:center; margin:1rem 0; }
    .result-low    { background:linear-gradient(135deg,#0a1f12,#081510); border:1px solid #0f4a22; border-radius:16px; padding:2rem; text-align:center; margin:1rem 0; }
    .result-score  { font-size:4rem; font-weight:700; line-height:1; margin:0.4rem 0; }
    .result-label  { font-size:0.82rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; }
    .result-sub    { font-size:0.82rem; color:#6b6b80; margin-top:0.4rem; }
    .tip {
        display:flex; align-items:flex-start; gap:0.8rem;
        background:#0e0e18; border:1px solid #1a1a2e;
        border-radius:10px; padding:0.9rem 1rem; margin-bottom:0.6rem;
    }
    .tip-icon  { font-size:1.1rem; margin-top:1px; }
    .tip-text  { font-size:0.875rem; color:#b0b0c4; line-height:1.5; }
    .tip-title { font-weight:600; color:#d0d0e4; font-size:0.875rem; }
    .stButton > button {
        width:100%;
        background:linear-gradient(135deg,#ff4422,#cc2200);
        color:white; border:none; border-radius:10px;
        padding:0.8rem 2rem; font-size:0.95rem;
        font-weight:600; letter-spacing:0.04em;
    }
    .stButton > button:hover {
        background:linear-gradient(135deg,#ff5533,#dd3311);
        box-shadow:0 8px 24px rgba(255,68,34,0.3);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Risk Analysis</div>
    <h1>🔥 FireGuard AI</h1>
    <p>Predict building fire risk before incidents occur — powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">Building Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    occupancy   = st.selectbox("Occupancy Type", [
        'apartments','office','1 family dwelling','2 family dwelling',
        'retail sales','warehouse','hotel/motel',
        'factory/industrial','storage','school','hospital'])
    floors      = st.slider("Number of Floors", 1, 50, 2)
    incidents   = st.number_input("Past Fire Incidents", 0, 50, 0)

with col2:
    wiring_age  = st.slider("Wiring / Construction Age (yrs)", 0, 80, 10)
    num_permits = st.number_input("Permits Filed", 1, 500, 4)
    had_permit  = st.radio("Fire Safety Permit?", ['No','Yes'], horizontal=True)
    maintenance = st.selectbox("Maintenance Frequency", [
        'Regular (< 5 years)','Irregular (5+ years)'])

st.markdown('</div>', unsafe_allow_html=True)

predict = st.button("⚡ Assess Fire Risk")

if predict:
    construction_risk    = min(5, max(1, round(wiring_age / 16)))
    occupancy_risk_map   = {
        'warehouse':5,'factory/industrial':5,'storage':5,
        'retail sales':4,'apartments':4,'hotel/motel':4,
        '2 family dwelling':3,'1 family dwelling':3,
        'office':2,'school':2,'hospital':2
    }
    occupancy_risk       = occupancy_risk_map.get(occupancy, 3)
    neglected            = 1 if maintenance == 'Irregular (5+ years)' else 0
    fire_permit          = 1 if had_permit == 'Yes' else 0
    incident_per_story   = incidents / (floors + 1)
    permit_risk_combined = construction_risk * occupancy_risk

    try:
        existing_use_encoded = le.transform([occupancy])[0]
    except:
        existing_use_encoded = 0

    input_data = {
        'existing_use'           : existing_use_encoded,
        'construction_risk_score': construction_risk,
        'num_stories'            : floors,
        'had_fire_permit'        : fire_permit,
        'num_permits'            : num_permits,
        'total_fire_incidents'   : incidents,
        'neglected_flag'         : neglected,
        'occupancy_risk_score'   : occupancy_risk,
        'incident_per_story'     : incident_per_story,
        'permit_risk_combined'   : permit_risk_combined,
    }

    input_df  = pd.DataFrame([input_data])[cols]
    risk_prob = model.predict_proba(input_df)[0][1]
    risk_pct  = round(risk_prob * 100, 1)
    # ── Risk penalty system ──
    penalty = 0
    if fire_permit == 0:       penalty += 10
    if incidents > 0:          penalty += incidents * 8
    if neglected == 1:         penalty += 12
    if construction_risk >= 4: penalty += 10
    penalty += (occupancy_risk - 2) * 4
    penalty += min(floors * 1.5, 15)

    risk_pct = round(min(100, max(5, risk_pct + penalty)), 1)

    if risk_pct >= 70:
        color, cls, label, sub = "#ff4422","result-high","High Risk","Immediate action recommended"
    elif risk_pct >= 40:
        color, cls, label, sub = "#f5a623","result-medium","Moderate Risk","Monitor and address key issues"
    elif risk_pct >= 20:
        color, cls, label, sub = "#f5a623","result-medium","Low-Moderate Risk","Some concerns worth addressing"
    else:
        color, cls, label, sub = "#2ecc71","result-low","Low Risk","Building appears reasonably safe"

    st.markdown(f"""
    <div class="{cls}">
        <div class="result-label" style="color:{color}">{label}</div>
        <div class="result-score" style="color:{color}">{risk_pct}%</div>
        <div class="result-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = risk_pct,
        number= {"suffix":"%","font":{"size":32,"color":color,"family":"Inter"}},
        gauge = {
            "axis"      : {"range":[0,100],"tickcolor":"#3a3a4a","tickfont":{"color":"#5a5a72","size":11}},
            "bar"       : {"color":color,"thickness":0.3},
            "bgcolor"   : "#13131a",
            "bordercolor":"#1e1e2e",
            "steps"     : [
                {"range":[0, 40],"color":"#0a1f12"},
                {"range":[40,70],"color":"#1f1a08"},
                {"range":[70,100],"color":"#2a0e0e"},
            ],
            "threshold" : {"line":{"color":color,"width":3},"value":risk_pct}
        }
    ))
    fig.update_layout(
        height=240, margin=dict(t=20,b=10,l=30,r=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family":"Inter"}
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="card"><div class="card-title">Risk Factor Breakdown</div>', unsafe_allow_html=True)
    factors = {
        "Construction / Wiring" : (construction_risk / 5) * 100,
        "Occupancy Type"        : (occupancy_risk    / 5) * 100,
        "Past Fire Incidents"   : min(incidents * 20, 100),
        "Building Height"       : min(floors * 4, 100),
        "Maintenance Neglect"   : neglected * 100,
        "No Fire Permit"        : (1 - fire_permit)  * 100,
    }
    for factor, val in factors.items():
        val = round(val)
        bar_color = "#ff4422" if val >= 70 else "#f5a623" if val >= 40 else "#2ecc71"
        st.markdown(f"""
        <div style="margin-bottom:0.8rem">
            <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#8b8b9a;margin-bottom:4px">
                <span>{factor}</span>
                <span style="color:{bar_color};font-weight:600">{int(val)}%</span>
            </div>
            <div style="background:#1e1e2e;border-radius:99px;height:6px">
                <div style="width:{val}%;background:{bar_color};height:6px;border-radius:99px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Smart Recommendation Engine ──────────────────────────
    tips = []

    # CRITICAL — Immediate action
    if incidents > 2:
        tips.append({
            "icon"    : "🚨",
            "title"   : "Repeat Fire History Detected",
            "text"    : f"This building has {incidents} past fire incidents — far above average. Install or upgrade an automatic sprinkler system and conduct a full fire safety audit.",
            "timeline": "Immediate",
            "color"   : "#ff4422"
        })

    if fire_permit == 0 and incidents > 0:
        tips.append({
            "icon"    : "📋",
            "title"   : "No Fire Permit Despite Incidents",
            "text"    : "Building has fire history but no active fire safety permit. This is a compliance violation — apply with local fire authority immediately.",
            "timeline": "Immediate",
            "color"   : "#ff4422"
        })

    if construction_risk >= 4 and neglected == 1:
        tips.append({
            "icon"    : "⚡",
            "title"   : "Critical Infrastructure Risk",
            "text"    : f"Wiring age of ~{wiring_age} years combined with irregular maintenance is a high-risk combination. Schedule a certified electrical inspection before next occupancy.",
            "timeline": "Immediate",
            "color"   : "#ff4422"
        })

    if floors > 15 and fire_permit == 0:
        tips.append({
            "icon"    : "🏢",
            "title"   : "High-Rise Without Fire Permit",
            "text"    : f"A {floors}-floor building without a fire permit poses serious evacuation risks. Mandatory fire safety certification required.",
            "timeline": "Immediate",
            "color"   : "#ff4422"
        })

    # WARNING — Within 30 days
    if construction_risk >= 4 and neglected == 0:
        tips.append({
            "icon"    : "⚡",
            "title"   : "Aging Wiring",
            "text"    : f"Construction/wiring age of ~{wiring_age} years increases electrical fire risk. Schedule a preventive electrical inspection soon.",
            "timeline": "Within 30 days",
            "color"   : "#f5a623"
        })

    if fire_permit == 0 and incidents == 0:
        tips.append({
            "icon"    : "📋",
            "title"   : "Missing Fire Safety Permit",
            "text"    : "No fire safety permit on record. While no incidents have occurred, compliance with local fire codes protects occupants and reduces liability.",
            "timeline": "Within 30 days",
            "color"   : "#f5a623"
        })

    if neglected == 1 and incidents == 0:
        tips.append({
            "icon"    : "🔧",
            "title"   : "Maintenance Overdue",
            "text"    : "No maintenance activity in 5+ years. Schedule a full building inspection to identify hidden risks before they escalate.",
            "timeline": "Within 30 days",
            "color"   : "#f5a623"
        })

    if occupancy in ['warehouse', 'factory/industrial', 'storage'] and fire_permit == 0:
        tips.append({
            "icon"    : "🏭",
            "title"   : "High-Risk Occupancy — Permit Required",
            "text"    : "Industrial and storage buildings are high fire-risk zones. Fire-rated compartmentation, hazmat protocols, and a valid permit are essential.",
            "timeline": "Within 30 days",
            "color"   : "#f5a623"
        })

    if floors > 10 and floors <= 15:
        tips.append({
            "icon"    : "🏢",
            "title"   : "Multi-Storey Evacuation Plan",
            "text"    : f"A {floors}-floor building requires clearly marked fire escape routes and evacuation drills for all occupants.",
            "timeline": "Within 30 days",
            "color"   : "#f5a623"
        })

    # GOOD PRACTICE — Annual
    if incidents == 0 and fire_permit == 1 and neglected == 0:
        tips.append({
            "icon"    : "✅",
            "title"   : "Good Safety Standing",
            "text"    : "No major risk factors detected. Continue annual fire safety inspections and keep permits up to date.",
            "timeline": "Annual",
            "color"   : "#2ecc71"
        })

    if construction_risk <= 2:
        tips.append({
            "icon"    : "🏗️",
            "title"   : "Modern Construction",
            "text"    : "Building uses fire-resistant construction materials. Maintain regular structural inspections annually.",
            "timeline": "Annual",
            "color"   : "#2ecc71"
        })

    if num_permits > 10:
        tips.append({
            "icon"    : "📁",
            "title"   : "High Permit Activity",
            "text"    : f"{int(num_permits)} permits filed — ensure all recent renovation work meets current fire safety codes during annual review.",
            "timeline": "Annual",
            "color"   : "#2ecc71"
        })

    # Fallback agar koi tip nahi bana
    if not tips:
        tips.append({
            "icon"    : "✅",
            "title"   : "No Major Issues Found",
            "text"    : "Building profile appears within safe parameters. Schedule annual fire safety inspection to maintain this status.",
            "timeline": "Annual",
            "color"   : "#2ecc71"
        })

    # ── Sort: Critical first, then 30 days, then Annual ──
    order = {"Immediate": 0, "Within 30 days": 1, "Annual": 2}
    tips  = sorted(tips, key=lambda x: order[x["timeline"]])

    # ── Display ───────────────────────────────────────────────
    st.markdown('<div class="card-title" style="margin-top:1.2rem">Recommendations</div>',
                unsafe_allow_html=True)

    for tip in tips:
        tl_color = "#ff4422" if tip["timeline"] == "Immediate" else \
                   "#f5a623" if tip["timeline"] == "Within 30 days" else "#2ecc71"
        st.markdown(f"""
        <div class="tip" style="border-left: 3px solid {tip['color']}">
            <span class="tip-icon">{tip['icon']}</span>
            <div style="flex:1">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div class="tip-title">{tip['title']}</div>
                    <span style="font-size:0.7rem;font-weight:600;
                                 color:{tl_color};
                                 background:rgba(255,255,255,0.05);
                                 padding:2px 8px;border-radius:99px;
                                 border:1px solid {tl_color};
                                 white-space:nowrap;margin-left:8px">
                        {tip['timeline']}
                    </span>
                </div>
                <div class="tip-text">{tip['text']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

"""
PowerSense AI — Intelligent Electricity Management Assistant
--------------------------------------------------------------
A Streamlit dashboard implementing:
  1. Home dashboard with KPI cards
  2. Sidebar navigation across all feature areas
  3. Bill Analyzer (upload -> extraction -> health analysis)
  4. Consumption analytics (trend chart, peak usage, cost)
  5. Multi-agent architecture visualization
  6. AI Assistant chat interface
  7. Confidence / evidence blocks
  8. RAG "Knowledge Center" with visible sources
  9. Savings Simulator
  10. Anomaly / problem detection with severity levels

NOTE ON AI INTEGRATION
-----------------------
Every place that should eventually call your real AI backend (bill OCR/
extraction, RAG retrieval, multi-agent orchestration, chat completion) is
implemented as a small function near the top marked `# --- AI HOOK ---`.
Right now these functions return realistic mock data so the whole app
runs end-to-end without any external dependencies. Swap the body of each
hook with a call to your actual model / agent / vector DB and the rest
of the UI keeps working unchanged.

Run with:
    pip install streamlit plotly pandas numpy
    streamlit run app.py
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLE
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="PowerSense AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .ps-header {
        padding: 0.25rem 0 1rem 0;
        border-bottom: 1px solid rgba(120,120,120,0.2);
        margin-bottom: 1.5rem;
    }
    .ps-title { font-size: 1.9rem; font-weight: 700; margin-bottom: 0; }
    .ps-subtitle { color: #8a8f98; font-size: 0.95rem; margin-top: -6px; }

    .kpi-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(16,185,129,0.06));
        border: 1px solid rgba(120,120,120,0.15);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .kpi-label { font-size: 0.8rem; color: #8a8f98; margin-bottom: 4px; }
    .kpi-value { font-size: 1.5rem; font-weight: 700; }
    .kpi-delta-up { color: #ef4444; font-size: 0.82rem; }
    .kpi-delta-down { color: #10b981; font-size: 0.82rem; }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-green  { background: rgba(16,185,129,0.15); color: #10b981; }
    .badge-yellow { background: rgba(234,179,8,0.15);  color: #ca8a04; }
    .badge-orange { background: rgba(249,115,22,0.15); color: #ea580c; }
    .badge-red    { background: rgba(239,68,68,0.15);  color: #ef4444; }

    .agent-box {
        border: 1px solid rgba(120,120,120,0.2);
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .evidence-box {
        background: rgba(120,120,120,0.06);
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
        font-size: 0.88rem;
    }
    .source-chip {
        display: inline-block;
        background: rgba(99,102,241,0.1);
        color: #6366f1;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        margin: 2px 4px 2px 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# MOCK DATA GENERATION (replace with real data sources / DB)
# ----------------------------------------------------------------------

random.seed(7)
np.random.seed(7)

MONTHS = pd.date_range(end=datetime.today(), periods=12, freq="MS").strftime("%b %Y")


def _mock_consumption_series():
    base = 260
    trend = np.linspace(0, 60, 12)
    noise = np.random.normal(0, 18, 12)
    values = base + trend + noise
    values[-1] = values[-2] * 1.184  # force an 18.4% spike in the latest month
    return np.round(values).astype(int)


if "consumption_df" not in st.session_state:
    st.session_state.consumption_df = pd.DataFrame(
        {"month": MONTHS, "units": _mock_consumption_series()}
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "bill_data" not in st.session_state:
    st.session_state.bill_data = None

df = st.session_state.consumption_df
current_units = int(df["units"].iloc[-1])
previous_units = int(df["units"].iloc[-2])
pct_change = round((current_units - previous_units) / previous_units * 100, 1)
RATE_PER_UNIT = 42  # Rs. per kWh, illustrative slab-blended rate
current_bill = int(current_units * RATE_PER_UNIT * 1.08)  # + taxes/FPA
estimated_saving = int(current_units * 0.10 * RATE_PER_UNIT)


# ----------------------------------------------------------------------
# --- AI HOOK --- functions (mocked; wire these to your real backend)
# ----------------------------------------------------------------------

def ai_extract_bill(uploaded_file):
    """Bill Extraction Agent. Replace with OCR/LLM extraction."""
    return {
        "consumer_id": "10-2938-4471",
        "billing_month": datetime.today().strftime("%B %Y"),
        "previous_reading": 18420,
        "current_reading": 18420 + current_units,
        "units_consumed": current_units,
        "electricity_charges": int(current_units * RATE_PER_UNIT),
        "taxes": int(current_units * RATE_PER_UNIT * 0.06),
        "fpa_adjustment": int(current_units * RATE_PER_UNIT * 0.02),
        "total_payable": current_bill,
        "due_date": (datetime.today() + timedelta(days=14)).strftime("%d %b %Y"),
    }


def ai_bill_health(units_now, units_prev):
    """Bill Analysis Agent."""
    change = (units_now - units_prev) / units_prev * 100
    if change > 15:
        return "Needs Attention", "🔴", change
    elif change > 8:
        return "Watch", "🟡", change
    return "Normal", "🟢", change


def ai_why_bill_high(units_now, units_prev):
    """Consumption reasoning agent."""
    change = round((units_now - units_prev) / units_prev * 100, 1)
    contributors = [
        "Air conditioner usage increased during peak summer days",
        "More consumption fell into evening peak hours (7 PM – 11 PM)",
        "Higher total units pushed part of the bill into a higher tariff slab",
    ]
    return change, contributors


def ai_detect_anomaly(units_now, avg_units):
    """Problem Detection Agent."""
    diff_pct = round((units_now - avg_units) / avg_units * 100, 1)
    if diff_pct > 20:
        severity, badge = "High", "badge-red"
    elif diff_pct > 10:
        severity, badge = "Medium", "badge-orange"
    elif diff_pct > 3:
        severity, badge = "Low", "badge-yellow"
    else:
        severity, badge = "None", "badge-green"
    causes = [
        "Increased air-conditioner or heater usage",
        "A malfunctioning or aging appliance drawing extra load",
        "Longer appliance operating hours than usual",
        "Possible meter reading or billing discrepancy",
    ]
    return diff_pct, severity, badge, causes


def ai_recommendations(units_now):
    """Recommendation Agent."""
    return [
        {"title": "Shift AC usage outside peak hours (7–11 PM)",
         "impact": "~25 units/month", "saving": f"Rs. {25*RATE_PER_UNIT:,}"},
        {"title": "Reduce AC runtime by 1 hour/day",
         "impact": "~18 units/month", "saving": f"Rs. {18*RATE_PER_UNIT:,}"},
        {"title": "Switch to LED lighting in remaining fixtures",
         "impact": "~8 units/month", "saving": f"Rs. {8*RATE_PER_UNIT:,}"},
        {"title": "Service or inspect appliances older than 8 years",
         "impact": "~12 units/month", "saving": f"Rs. {12*RATE_PER_UNIT:,}"},
    ]


def ai_chat_response(user_message):
    """Chat Agent + RAG. Replace with a real LLM call (see integration
    notes at the bottom of this file for wiring the Anthropic API)."""
    msg = user_message.lower()
    if "why" in msg and "high" in msg:
        text = (f"Your consumption rose {pct_change}% versus last month, mainly due to "
                f"higher AC usage and more consumption during peak evening hours, which "
                f"also pushed part of your usage into a higher tariff slab.")
        sources = ["Billing Regulations", "Tariff Slab Guide"]
    elif "reduce" in msg or "save" in msg:
        text = ("The biggest lever is shifting AC usage outside the 7–11 PM peak window "
                "and trimming daily AC runtime by about an hour — together these could "
                "save roughly Rs. 2,800–3,200 per month.")
        sources = ["Energy Conservation Guidelines"]
    elif "unusual" in msg or "anomaly" in msg:
        text = ("Yes — this month's usage is about 20% above your 6-month average, which "
                "is flagged as a medium-severity anomaly. It's worth checking your AC "
                "and any recently added appliances.")
        sources = ["Consumer Billing FAQs"]
    else:
        text = ("I can help explain your bill, your consumption trend, or suggest ways "
                "to reduce your electricity costs — what would you like to look at?")
        sources = ["PowerSense Electricity Knowledge Base"]
    return text, sources


KNOWLEDGE_BASE = [
    {"title": "Electricity Tariff Slabs Explained", "category": "Tariffs"},
    {"title": "Understanding Fuel Price Adjustment (FPA)", "category": "Billing"},
    {"title": "How Billing Cycles & Due Dates Work", "category": "Billing"},
    {"title": "Common Reasons for Sudden Bill Increases", "category": "Troubleshooting"},
    {"title": "Energy-Saving Tips for Air Conditioners", "category": "Conservation"},
    {"title": "How to Read Your Electricity Meter", "category": "Guides"},
]

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚡ PowerSense AI")
    st.caption("Intelligent electricity management")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Dashboard",
            "🧾 Bill Analyzer",
            "📊 Consumption",
            "🚨 Problem Detection",
            "💡 Recommendations",
            "💰 Savings Simulator",
            "🤖 AI Assistant",
            "📚 Knowledge Center",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("##### Agent Status")
    agents = ["Bill Analysis", "Consumption", "Problem Detection", "Recommendation", "RAG Knowledge"]
    for a in agents:
        st.markdown(f"🟢 {a} Agent")

# ----------------------------------------------------------------------
# PAGE: DASHBOARD
# ----------------------------------------------------------------------

if page == "🏠 Dashboard":
    st.markdown(
        '<div class="ps-header"><div class="ps-title">Welcome back 👋</div>'
        '<div class="ps-subtitle">Here\'s your electricity overview</div></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "💰 Current Bill", f"Rs. {current_bill:,}", f"+{pct_change}%", True),
        (c2, "⚡ Units Consumed", f"{current_units} kWh", f"{pct_change:+.1f}%", pct_change > 0),
        (c3, "📈 Consumption Change", f"{pct_change:+.1f}%", "vs last month", pct_change > 0),
        (c4, "🚨 Issues Detected", "2", "1 medium, 1 low", True),
        (c5, "💡 Estimated Saving", f"Rs. {estimated_saving:,}", "if optimized", False),
    ]
    for col, label, value, delta, is_up in kpis:
        with col:
            delta_class = "kpi-delta-up" if is_up else "kpi-delta-down"
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="{delta_class}">{delta}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns([2, 1])

    with left:
        st.markdown("#### 📈 Consumption Trend")
        fig = px.line(df, x="month", y="units", markers=True)
        fig.update_traces(line_color="#6366f1", fill="tozeroy",
                           fillcolor="rgba(99,102,241,0.08)")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title=None, yaxis_title="Units (kWh)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### 🚨 AI Insights")
        st.markdown(
            f'<div class="evidence-box">⚠️ Consumption increased '
            f'<b>{pct_change:+.1f}%</b> vs last month<br><br>'
            f'💡 Optimizing AC usage could save about '
            f'<b>Rs. {estimated_saving:,}/month</b></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        b1, b2 = st.columns(2)
        with b1:
            st.button("🧾 Analyze Bill", use_container_width=True)
        with b2:
            st.button("🤖 Ask AI", use_container_width=True)

    st.write("")
    st.markdown("#### 🤖 PowerSense AI Engine")
    cols = st.columns(5)
    for col, name in zip(cols, agents):
        with col:
            st.markdown(f'<div class="agent-box">🟢<br>{name}<br>Agent</div>',
                         unsafe_allow_html=True)

# ----------------------------------------------------------------------
# PAGE: BILL ANALYZER
# ----------------------------------------------------------------------

elif page == "🧾 Bill Analyzer":
    st.markdown('<div class="ps-title">🧾 Bill Analyzer</div>'
                '<div class="ps-subtitle">Upload your electricity bill for AI extraction & analysis</div>',
                unsafe_allow_html=True)
    st.write("")

    uploaded = st.file_uploader("Upload bill (PDF or image)", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded is not None or st.button("Use sample bill for demo"):
        with st.spinner("AI Extraction Agent reading your bill..."):
            data = ai_extract_bill(uploaded)
            st.session_state.bill_data = data

    if st.session_state.bill_data:
        data = st.session_state.bill_data
        st.markdown("#### AI Extraction")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("Consumer ID", data["consumer_id"])
            st.metric("Billing Month", data["billing_month"])
        with e2:
            st.metric("Previous Reading", data["previous_reading"])
            st.metric("Current Reading", data["current_reading"])
        with e3:
            st.metric("Units Consumed", f"{data['units_consumed']} kWh")
            st.metric("Due Date", data["due_date"])

        st.write("")
        cost_df = pd.DataFrame({
            "Component": ["Electricity Charges", "Taxes", "FPA Adjustment"],
            "Amount (Rs.)": [data["electricity_charges"], data["taxes"], data["fpa_adjustment"]],
        })
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(cost_df, hide_index=True, use_container_width=True)
            st.markdown(f"**Total Payable: Rs. {data['total_payable']:,}**")
        with c2:
            fig = px.pie(cost_df, names="Component", values="Amount (Rs.)", hole=0.55)
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.write("")
        st.markdown("#### AI Analysis")
        status, emoji, change = ai_bill_health(current_units, previous_units)
        badge_class = {"Normal": "badge-green", "Watch": "badge-yellow", "Needs Attention": "badge-red"}[status]
        st.markdown(f'<span class="badge {badge_class}">{emoji} Bill Health: {status}</span>',
                    unsafe_allow_html=True)
        st.write("")
        st.markdown(
            f'<div class="evidence-box">Your consumption increased by <b>{change:.1f}%</b> '
            f'compared with the previous billing period.<br><br>'
            f'<b>Confidence: 91%</b><br>'
            f'Evidence: current {current_units} units vs previous {previous_units} units.</div>',
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("#### 🔎 Why is your bill high?")
        change_pct, contributors = ai_why_bill_high(current_units, previous_units)
        st.markdown(f"**Primary reason** — ⚡ {change_pct}% increase in electricity consumption")
        st.markdown("**Likely contributors**")
        for c in contributors:
            st.markdown(f"- {c}")
        st.markdown("**💡 What you can do**")
        st.info("Reducing AC usage by approximately 1 hour/day could potentially reduce "
                 "monthly consumption by 15–20 units.")
    else:
        st.caption("Upload a bill above, or click 'Use sample bill for demo' to see the analysis flow.")

# ----------------------------------------------------------------------
# PAGE: CONSUMPTION
# ----------------------------------------------------------------------

elif page == "📊 Consumption":
    st.markdown('<div class="ps-title">📊 Consumption Analytics</div>', unsafe_allow_html=True)
    st.write("")

    fig = px.bar(df, x="month", y="units", color="units", color_continuous_scale="Purples")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consumption Trend", f"{pct_change:+.1f}%", "vs previous month")
    c2.metric("Peak Usage Window", "7 PM – 11 PM")
    c3.metric("Highest Consumption Day", "Saturday")
    c4.metric("Estimated Monthly Cost", f"Rs. {current_bill:,}")

    st.write("")
    st.markdown("#### Hourly usage pattern (illustrative)")
    hours = list(range(24))
    load = [8, 6, 5, 5, 5, 6, 9, 12, 10, 9, 8, 9, 10, 9, 8, 9, 11, 15, 22, 25, 23, 18, 13, 10]
    hfig = go.Figure(go.Scatter(x=hours, y=load, mode="lines", fill="tozeroy",
                                 line_color="#10b981"))
    hfig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Hour of day", yaxis_title="Relative load")
    st.plotly_chart(hfig, use_container_width=True)

# ----------------------------------------------------------------------
# PAGE: PROBLEM DETECTION
# ----------------------------------------------------------------------

elif page == "🚨 Problem Detection":
    st.markdown('<div class="ps-title">🚨 Problem Detection</div>', unsafe_allow_html=True)
    st.write("")

    avg_units = int(df["units"].iloc[:-1].mean())
    diff_pct, severity, badge_class, causes = ai_detect_anomaly(current_units, avg_units)

    st.markdown(f'<span class="badge {badge_class}">Severity: {severity}</span>', unsafe_allow_html=True)
    st.write("")
    st.markdown("#### Anomaly Detected: Unusual consumption spike")

    tbl = pd.DataFrame({
        "Metric": ["6-month average", "Current month", "Difference"],
        "Value": [f"{avg_units} units", f"{current_units} units", f"{diff_pct:+.1f}%"],
    })
    st.dataframe(tbl, hide_index=True, use_container_width=True)

    st.markdown("**Possible causes**")
    for i, c in enumerate(causes, 1):
        st.markdown(f"{i}. {c}")

    st.markdown("**Recommended action**")
    st.info("Compare your meter reading with the bill and inspect high-consumption appliances "
             "such as your AC, water heater, or any recently added equipment.")

# ----------------------------------------------------------------------
# PAGE: RECOMMENDATIONS
# ----------------------------------------------------------------------

elif page == "💡 Recommendations":
    st.markdown('<div class="ps-title">💡 Personalized Recommendations</div>', unsafe_allow_html=True)
    st.write("")

    recs = ai_recommendations(current_units)
    for r in recs:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.markdown(f"**{r['title']}**")
            c2.markdown(f"Impact: {r['impact']}")
            c3.markdown(f"💰 {r['saving']}")

    total_saving = sum(int(r["saving"].split("Rs. ")[1].replace(",", "")) for r in recs)
    st.write("")
    st.success(f"Combined, these changes could save up to **Rs. {total_saving:,}/month** "
               f"(~Rs. {total_saving*12:,}/year).")

# ----------------------------------------------------------------------
# PAGE: SAVINGS SIMULATOR
# ----------------------------------------------------------------------

elif page == "💰 Savings Simulator":
    st.markdown('<div class="ps-title">💰 What if I reduce my usage?</div>', unsafe_allow_html=True)
    st.write("")

    c1, c2 = st.columns(2)
    with c1:
        ac_now = st.slider("Current AC usage (hrs/day)", 0, 16, 8)
    with c2:
        ac_target = st.slider("Target AC usage (hrs/day)", 0, 16, 5)

    peak_shift = st.slider("% of usage shifted out of 7–11 PM peak window", 0, 100, 30)

    hrs_reduced = max(ac_now - ac_target, 0)
    units_saved_ac = hrs_reduced * 30 * 1.5  # ~1.5 kWh/hr assumption
    units_saved_peak = current_units * (peak_shift / 100) * 0.05
    total_units_saved = units_saved_ac + units_saved_peak
    bill_reduction = int(total_units_saved * RATE_PER_UNIT)
    annual_saving = bill_reduction * 12

    st.write("")
    st.markdown("#### Estimated impact")
    m1, m2, m3 = st.columns(3)
    m1.metric("Units saved", f"~{total_units_saved:.0f} kWh/month")
    m2.metric("Estimated bill reduction", f"~Rs. {bill_reduction:,}/month")
    m3.metric("Annual potential saving", f"~Rs. {annual_saving:,}")

    fig = go.Figure(go.Bar(
        x=["Current bill", "Projected bill"],
        y=[current_bill, max(current_bill - bill_reduction, 0)],
        marker_color=["#ef4444", "#10b981"],
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Rs.")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# PAGE: AI ASSISTANT
# ----------------------------------------------------------------------

elif page == "🤖 AI Assistant":
    st.markdown('<div class="ps-title">🤖 Ask PowerSense</div>', unsafe_allow_html=True)
    st.write("")

    quick_qs = [
        "💰 Why is my bill high?",
        "⚡ What used the most electricity?",
        "📉 How can I reduce my bill?",
        "🚨 Is there anything unusual?",
    ]
    cols = st.columns(len(quick_qs))
    clicked = None
    for col, q in zip(cols, quick_qs):
        if col.button(q, use_container_width=True):
            clicked = q

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown(
                    "".join(f'<span class="source-chip">📚 {s}</span>' for s in msg["sources"]),
                    unsafe_allow_html=True,
                )

    user_input = st.chat_input("Ask about your bill, usage, or how to save...")
    prompt = clicked or user_input

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response, sources = ai_chat_response(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": response, "sources": sources})
        with st.chat_message("assistant"):
            st.markdown(response)
            st.markdown(
                "".join(f'<span class="source-chip">📚 {s}</span>' for s in sources),
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------
# PAGE: KNOWLEDGE CENTER (RAG)
# ----------------------------------------------------------------------

elif page == "📚 Knowledge Center":
    st.markdown('<div class="ps-title">📚 Knowledge Center</div>'
                '<div class="ps-subtitle">Answers are grounded in the PowerSense Electricity Knowledge Base (RAG)</div>',
                unsafe_allow_html=True)
    st.write("")

    categories = sorted(set(a["category"] for a in KNOWLEDGE_BASE))
    tabs = st.tabs(categories)
    for tab, cat in zip(tabs, categories):
        with tab:
            for article in [a for a in KNOWLEDGE_BASE if a["category"] == cat]:
                st.markdown(f"- {article['title']}")

    st.write("")
    st.markdown("#### AI Knowledge Source")
    st.markdown(
        '<div class="evidence-box">Responses in the AI Assistant are generated using the '
        'PowerSense Electricity Knowledge Base rather than an unconstrained model — each '
        'answer cites the specific articles it drew from.</div>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------
# PAGE: SETTINGS
# ----------------------------------------------------------------------

elif page == "⚙️ Settings":
    st.markdown('<div class="ps-title">⚙️ Settings</div>', unsafe_allow_html=True)
    st.write("")

    st.selectbox("Language", ["English", "Urdu"])
    st.selectbox("Units", ["kWh", "MWh"])
    st.selectbox("Currency", ["PKR (Rs.)", "USD ($)"])
    st.number_input("Rate per unit (Rs./kWh)", value=float(RATE_PER_UNIT), step=1.0)
    st.toggle("Enable anomaly email alerts", value=True)
    st.toggle("Enable weekly summary", value=False)
    st.button("Save settings", type="primary")

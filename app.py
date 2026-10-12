import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Academic Horizon | University Cutoff Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# ASSETS & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
ENCODER_PATH = BASE_DIR / "encoders.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

# ==============================================================================
# CUSTOM STYLING (CSS DESIGN SYSTEM)
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Base App Background */
.stApp {
    background: radial-gradient(circle at 10% 20%, #f1f5f9 0%, #f8fafc 100%);
    color: #0f172a;
}

/* App Header & Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
    border-radius: 20px;
    padding: 2.2rem 2.4rem;
    color: #ffffff;
    box-shadow: 0 16px 36px -10px rgba(49, 46, 129, 0.35);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero-banner::after {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}

.hero-badges {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 0.8rem;
}

.hero-badge {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 0.3rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    color: #e0e7ff;
}

.hero-title {
    font-size: 2.15rem;
    font-weight: 800;
    line-height: 1.25;
    margin: 0.2rem 0 0.6rem 0;
    color: #ffffff !important;
}

.hero-subtitle {
    font-size: 1rem;
    color: #c7d2fe;
    max-width: 780px;
    line-height: 1.6;
    margin-bottom: 0;
}

/* Glassmorphism Card System */
.glass-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 1.7rem;
    box-shadow: 0 4px 18px -2px rgba(15, 23, 42, 0.05);
    margin-bottom: 1.5rem;
}

.card-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
}

.card-desc {
    color: #64748b;
    font-size: 0.88rem;
    margin-bottom: 1.2rem;
}

/* Prediction Result Card */
.result-display {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    border-radius: 20px;
    padding: 2.2rem;
    color: white;
    text-align: center;
    box-shadow: 0 16px 36px -10px rgba(15, 23, 42, 0.4);
    position: relative;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-top: 1rem;
    margin-bottom: 1.5rem;
}

.result-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(99, 102, 241, 0.25);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
    margin-bottom: 0.8rem;
}

.result-metric {
    font-size: 3.4rem;
    font-weight: 800;
    line-height: 1;
    color: #ffffff;
    margin: 0.6rem 0;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
}

.result-subtext {
    font-size: 1.05rem;
    color: #cbd5e1;
    font-weight: 500;
}

/* KPI Metric Pill Grid */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1.2rem;
}

.kpi-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem;
    text-align: center;
}

.kpi-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.kpi-val {
    font-size: 1.25rem;
    font-weight: 750;
    color: #1e1b4b;
    margin-top: 0.3rem;
}

/* Custom Primary Button Styling */
div.stButton > button {
    width: 100%;
    border-radius: 14px;
    min-height: 3.25rem;
    font-size: 1.05rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4338ca 0%, #6366f1 100%);
    color: white;
    border: none;
    box-shadow: 0 8px 20px -4px rgba(67, 56, 202, 0.45);
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #3730a3 0%, #4f46e5 100%);
    transform: translateY(-2px);
    box-shadow: 0 12px 24px -4px rgba(67, 56, 202, 0.6);
    color: white;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Footer Styling */
.app-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 3.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# ASSET LOADER (CACHED)
# ==============================================================================
@st.cache_resource(show_spinner="Initializing Machine Learning Engine...")
def load_assets():
    """Load model, encoders, and scaler with thorough diagnostics."""
    for name, path in [("Model", MODEL_PATH), ("Encoders", ENCODER_PATH), ("Scaler", SCALER_PATH)]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required asset: {name} file not found at {path}")

    with open(MODEL_PATH, "rb") as f:
        loaded_model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        loaded_encoders = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        loaded_scaler = pickle.load(f)

    if not isinstance(loaded_encoders, dict):
        raise ValueError("Encoders pickle is expected to be a dictionary.")

    for req in ["le_kind", "le_seat", "le_target"]:
        if req not in loaded_encoders:
            raise KeyError(f"Missing encoder key '{req}' in encoders.pkl.")

    return loaded_model, loaded_encoders, loaded_scaler

# Load models safely
try:
    model, encoders, scaler = load_assets()
    le_kind = encoders["le_kind"]
    le_seat = encoders["le_seat"]
    le_target = encoders["le_target"]
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    load_error = e

# ==============================================================================
# PREDICTION HELPER FUNCTION
# ==============================================================================
def predict_cutoff_value(year_val, kind_val, seat_val, target_val):
    """
    Robust inference engine.
    Scaler transforms 'year' (StandardScaler fitted on 'year').
    Model expects DataFrame with ['year', 'figure_kind', 'seat_type', 'basis'].
    """
    kind_enc = int(le_kind.transform([kind_val])[0])
    seat_enc = int(le_seat.transform([seat_val])[0])
    target_enc = int(le_target.transform([target_val])[0])

    # Scale year using the fitted scaler
    if hasattr(scaler, "transform"):
        year_input = pd.DataFrame([[float(year_val)]], columns=["year"])
        try:
            year_scaled = float(scaler.transform(year_input)[0][0])
        except Exception:
            year_scaled = float(scaler.transform([[float(year_val)]])[0][0])
    else:
        year_scaled = float(year_val)

    # Prepare DataFrame matching model's expected feature names
    feature_names = getattr(model, "feature_names_in_", ["year", "figure_kind", "seat_type", "basis"])
    input_df = pd.DataFrame(
        [[year_scaled, kind_enc, seat_enc, target_enc]],
        columns=feature_names
    )

    pred = float(model.predict(input_df)[0])
    return pred

# ==============================================================================
# LABEL FORMATTERS
# ==============================================================================
KIND_LABELS = {
    "rank": "Cutoff Closing Rank (Merit Standing)",
    "score": "Cutoff Score (Exam Marks Required)"
}

SEAT_LABELS = {
    "Delhi region": "Delhi Region Quota (State Quota)",
    "Outside Delhi": "Outside Delhi Quota (All-India Seat)",
    "All-India open merit": "All-India Open Merit",
    "J&K open merit": "J&K Open Merit (PMSSS Special)",
    "general": "General Merit Quota",
    "Paid Fee Course": "Paid Fee Course (Self-Financed)",
    "Special Fee Course": "Special Fee Course (Sponsored/NRI)"
}

TARGET_LABELS = {
    "official": "Official Declared Cutoff",
    "merit-list": "Merit-List Allocation Basis"
}

# ==============================================================================
# SIDEBAR NAVIGATION & QUICK PRESETS
# ==============================================================================
with st.sidebar:
    st.markdown("### 🏛️ Academic Horizon")
    st.caption("AI-Powered University Cutoff Intelligence")
    st.divider()

    st.markdown("#### ⚡ Quick Presets")
    st.write("Test common admission scenarios instantly:")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("Delhi (Rank)", use_container_width=True):
            st.session_state["sel_year"] = 2026
            st.session_state["sel_kind"] = "rank"
            st.session_state["sel_seat"] = "Delhi region"
            st.session_state["sel_basis"] = "official"
            st.rerun()

        if st.button("All-India (Rank)", use_container_width=True):
            st.session_state["sel_year"] = 2026
            st.session_state["sel_kind"] = "rank"
            st.session_state["sel_seat"] = "All-India open merit"
            st.session_state["sel_basis"] = "official"
            st.rerun()

    with col_p2:
        if st.button("Outside (Score)", use_container_width=True):
            st.session_state["sel_year"] = 2026
            st.session_state["sel_kind"] = "score"
            st.session_state["sel_seat"] = "Outside Delhi"
            st.session_state["sel_basis"] = "merit-list"
            st.rerun()

        if st.button("General (Score)", use_container_width=True):
            st.session_state["sel_year"] = 2026
            st.session_state["sel_kind"] = "score"
            st.session_state["sel_seat"] = "general"
            st.session_state["sel_basis"] = "official"
            st.rerun()

    st.divider()
    st.markdown("#### ℹ️ Model Architecture")
    st.markdown("""
    - **Engine**: Random Forest Regressor
    - **Scaler**: Scikit-Learn StandardScaler (Cycle Year)
    - **Encoders**: Multi-Class LabelEncoders
    - **Metrics**: Real-Time Rank & Score Forecasts
    """)

    st.divider()
    st.markdown("#### 🛡️ About SIH 2026")
    st.caption(
        "Academic Horizon equips university aspirants with transparent "
        "data-backed cutoffs to optimize counselling preferences and admission outcomes."
    )

# ==============================================================================
# MAIN HERO BANNER
# ==============================================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-badges">
        <span class="hero-badge">🎓 SIH 2026 Edition</span>
        <span class="hero-badge">🤖 Random Forest AI</span>
        <span class="hero-badge">⚡ Real-time Inference</span>
        <span class="hero-badge">🌐 Cloud Ready</span>
    </div>
    <h1 class="hero-title">University Admission Cutoff Predictor</h1>
    <p class="hero-subtitle">
        Forecast expected university admission cutoff ranks and qualifying scores across
        admission cycles, reservation seat quotas, and merit quotas with machine-learning accuracy.
    </p>
</div>
""", unsafe_allow_html=True)

# Check asset availability
if not assets_loaded:
    st.error("⚠️ Failed to load machine learning assets.")
    st.info(f"Diagnostics: {load_error}")
    st.stop()

# ==============================================================================
# PARAMETER INPUT SECTION
# ==============================================================================
st.markdown('<div class="card-title">📍 Admission Configuration</div>', unsafe_allow_html=True)
st.markdown('<div class="card-desc">Select the candidate profile and seat quota below to generate cutoff forecasts.</div>', unsafe_allow_html=True)

kind_options = list(le_kind.classes_)
seat_options = list(le_seat.classes_)
target_options = list(le_target.classes_)

# Session state defaults
if "sel_year" not in st.session_state:
    st.session_state["sel_year"] = 2026
if "sel_kind" not in st.session_state:
    st.session_state["sel_kind"] = kind_options[0]
if "sel_seat" not in st.session_state:
    st.session_state["sel_seat"] = seat_options[0]
if "sel_basis" not in st.session_state:
    st.session_state["sel_basis"] = target_options[0]

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        year = st.selectbox(
            "📅 Admission Cycle Year",
            options=[2024, 2025, 2026, 2027],
            index=[2024, 2025, 2026, 2027].index(st.session_state["sel_year"])
                  if st.session_state["sel_year"] in [2024, 2025, 2026, 2027] else 2,
            help="Select the target university academic year."
        )

        figure_kind = st.selectbox(
            "📊 Prediction Metric (Figure Kind)",
            options=kind_options,
            format_func=lambda x: KIND_LABELS.get(x, x.title()),
            index=kind_options.index(st.session_state["sel_kind"])
                  if st.session_state["sel_kind"] in kind_options else 0,
            help="Predict whether you need the required rank ceiling or minimum cutoff marks."
        )

    with col2:
        seat_type = st.selectbox(
            "🎯 Seat Allocation Category",
            options=seat_options,
            format_func=lambda x: SEAT_LABELS.get(x, x.title()),
            index=seat_options.index(st.session_state["sel_seat"])
                  if st.session_state["sel_seat"] in seat_options else 0,
            help="Select the quota, domicile, or seat category."
        )

        basis = st.selectbox(
            "📝 Admission Basis",
            options=target_options,
            format_func=lambda x: TARGET_LABELS.get(x, x.title()),
            index=target_options.index(st.session_state["sel_basis"])
                  if st.session_state["sel_basis"] in target_options else 0,
            help="Choose between official declared cutoffs or final allotment merit-list standards."
        )

st.write("")
predict_clicked = st.button("🔮 Predict Admission Cutoff", use_container_width=True)

# Auto-execute on initial load or button click
if predict_clicked or "prediction_made" not in st.session_state:
    st.session_state["prediction_made"] = True

if st.session_state.get("prediction_made", False):
    try:
        prediction = predict_cutoff_value(year, figure_kind, seat_type, basis)

        is_score = str(figure_kind).strip().lower() == "score"

        if is_score:
            formatted_val = f"{prediction:,.2f}"
            metric_badge = "Target Qualifying Score"
            subtext = "Estimated Qualifying Exam Score Required for Admission"
            comp_level = "High Score Threshold" if prediction > 450 else "Moderate Score Threshold"
            p_bracket = "Top 10-15% Score Bracket" if prediction > 450 else "Top 25-30% Score Bracket"
        else:
            formatted_val = f"{int(np.rint(prediction)):,}"
            metric_badge = "Estimated Closing Rank"
            subtext = "Estimated Cutoff Closing Rank Ceiling"
            comp_level = "🔥 Ultra-Competitive (< 5k Rank)" if prediction < 5000 else "⚡ Standard Competitive Range"
            p_bracket = "Top 2-5% of Candidates" if prediction < 10000 else "Top 15% of Candidates"

        # Display Result Card
        st.markdown(f"""
        <div class="result-display">
            <span class="result-badge">{metric_badge}</span>
            <div class="result-metric">{formatted_val}</div>
            <div class="result-subtext">{subtext} ({year} Cycle)</div>
        </div>
        """, unsafe_allow_html=True)

        # KPI Metrics Cards
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        with kpi_col1:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">Competitiveness Tier</div>
                <div class="kpi-val">{comp_level}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col2:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">Estimated Percentile</div>
                <div class="kpi-val">{p_bracket}</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_col3:
            strategy_tip = "Target Round 1 Allotment" if (not is_score and prediction < 6000) or (is_score and prediction > 440) else "Plan for Round 2 & Mop-Up"
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-label">Recommended Strategy</div>
                <div class="kpi-val">{strategy_tip}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ==============================================================================
        # INTERACTIVE TABS: TRENDS, COMPARISON & COUNSELING
        # ==============================================================================
        tab_trends, tab_category, tab_guide = st.tabs([
            "📈 Multi-Year Cutoff Trends",
            "⚖️ Cross-Category Analysis",
            "💡 Admission & Counseling Playbook"
        ])

        with tab_trends:
            st.markdown("##### 📊 Projected Cutoffs Across Admission Cycles (2024 - 2027)")
            st.caption(f"Category: **{SEAT_LABELS.get(seat_type, seat_type)}** | Basis: **{TARGET_LABELS.get(basis, basis)}**")

            years_list = [2024, 2025, 2026, 2027]
            trend_data = []
            for y in years_list:
                pred_y = predict_cutoff_value(y, figure_kind, seat_type, basis)
                trend_data.append({
                    "Admission Year": str(y),
                    "Predicted Cutoff": round(pred_y, 2) if is_score else int(np.rint(pred_y))
                })

            df_trends = pd.DataFrame(trend_data)
            df_chart = df_trends.set_index("Admission Year")

            st.line_chart(df_chart, use_container_width=True)

            # Tabular breakdown
            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                st.dataframe(df_trends, use_container_width=True, hide_index=True)
            with col_t2:
                delta = trend_data[-2]["Predicted Cutoff"] - trend_data[-3]["Predicted Cutoff"]
                st.info(f"**Year-on-Year Trend**: {'Tightening Cutoff' if delta > 0 and is_score else 'Stable Admission Range'}.\nPlan choice preferences accordingly.")

        with tab_category:
            st.markdown(f"##### ⚖️ Cutoff Benchmarks Across All Seat Quotas ({year})")
            st.caption(f"Figure Kind: **{KIND_LABELS.get(figure_kind, figure_kind)}**")

            cat_data = []
            for s in seat_options:
                pred_s = predict_cutoff_value(year, figure_kind, s, basis)
                cat_data.append({
                    "Seat Category": SEAT_LABELS.get(s, s),
                    "Cutoff Value": round(pred_s, 2) if is_score else int(np.rint(pred_s))
                })

            df_cat = pd.DataFrame(cat_data).set_index("Seat Category")
            st.bar_chart(df_cat, use_container_width=True)

        with tab_guide:
            st.markdown("##### 🎯 High-Impact Counseling Guidelines")
            with st.expander("📌 Choice Filling & Preference Locking Strategy", expanded=True):
                st.write("""
                1. **Ambitious Choices (Top 20%)**: Select dream universities where cutoff is 5–10% higher than your expected score/rank.
                2. **Target Choices (Middle 60%)**: Select colleges closely matching the predicted cutoff displayed above.
                3. **Safe Choices (Bottom 20%)**: Ensure backup colleges with historically relaxed cutoffs to guarantee seat allotment.
                """)

            with st.expander("📄 Verification Documents Checklist"):
                st.write("""
                - Admit Card & Rank Scorecard
                - 10th & 12th Standard Original Marksheets
                - Category / Caste / EWS / Domicile Certificate (if applicable)
                - Government Issued Photo ID (Aadhaar / Passport)
                - Provisional Seat Allotment Letter
                """)

            with st.expander("⚠️ Disclaimer & Model Notice"):
                st.caption(
                    "Predictions are generated using statistical and machine learning modeling based on historic admission patterns. "
                    "Actual cutoffs depend upon applicant volume, seat matrix adjustments, and official university regulatory notices."
                )

    except Exception as err:
        st.error(f"❌ Prediction Calculation Error: {err}")
        st.info("Check parameters or reload the application.")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("""
<div class="app-footer">
    <b>Academic Horizon — University Admission Cutoff Predictor</b><br>
    Built with Streamlit & Scikit-Learn | SIH 2026 Academic Intelligence Suite
</div>
""", unsafe_allow_html=True)

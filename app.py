import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="University Admission Cutoff Predictor",
    page_icon="🎓",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
}

.main .block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    background: linear-gradient(135deg, #4338ca, #6366f1);
    padding: 2rem;
    border-radius: 22px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 30px rgba(67, 56, 202, 0.20);
}

.hero h1 {
    font-size: 2.3rem;
    margin-bottom: 0.5rem;
    font-weight: 800;
}

.hero p {
    font-size: 1rem;
    opacity: 0.92;
    margin-bottom: 0;
}

.card {
    background: white;
    padding: 1.5rem;
    border-radius: 18px;
    margin-bottom: 1.2rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
}

.card-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 1rem;
}

.result-card {
    background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
    border: 1px solid #bbf7d0;
    padding: 1.5rem;
    border-radius: 18px;
    text-align: center;
    margin-top: 1.5rem;
}

.result-title {
    color: #166534;
    font-size: 1rem;
    font-weight: 600;
}

.result-value {
    color: #15803d;
    font-size: 2.5rem;
    font-weight: 800;
    margin-top: 0.3rem;
}

.info-box {
    background: #eff6ff;
    border-left: 5px solid #3b82f6;
    padding: 1rem;
    border-radius: 10px;
    color: #1e3a8a;
    margin-top: 1rem;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.85rem;
    margin-top: 2rem;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3rem;
    font-size: 1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4338ca, #6366f1);
    color: white;
    border: none;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #3730a3, #4f46e5);
    color: white;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <h1>🎓 University Admission Cutoff Predictor</h1>
    <p>
        Predict university admission cutoffs using a trained
        machine learning model.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
ENCODER_PATH = BASE_DIR / "encoders.pkl"


# =========================================================
# LOAD MODEL AND ENCODERS
# =========================================================

@st.cache_resource
def load_assets():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "model.pkl was not found. Please upload model.pkl "
            "to the same folder as app.py."
        )

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            "encoders.pkl was not found. Please upload encoders.pkl "
            "to the same folder as app.py."
        )

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Missing Python module required by model.pkl: {e.name}"
        ) from e

    except ImportError as e:
        raise ImportError(
            f"Missing package required by model.pkl: {e}"
        ) from e

    except Exception as e:
        raise RuntimeError(
            f"Unable to load model.pkl: {e}"
        ) from e

    try:
        with open(ENCODER_PATH, "rb") as f:
            encoders = pickle.load(f)

    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Missing Python module required by encoders.pkl: {e.name}"
        ) from e

    except Exception as e:
        raise RuntimeError(
            f"Unable to load encoders.pkl: {e}"
        ) from e

    return model, encoders


# =========================================================
# LOAD ASSETS
# =========================================================

try:

    model, encoders = load_assets()

    required_encoders = [
        "le_kind",
        "le_seat",
        "le_target"
    ]

    missing_encoders = [
        key for key in required_encoders
        if key not in encoders
    ]

    if missing_encoders:
        st.error(
            "Missing encoder(s) in encoders.pkl: "
            + ", ".join(missing_encoders)
        )
        st.stop()

    le_kind = encoders["le_kind"]
    le_seat = encoders["le_seat"]
    le_target = encoders["le_target"]

except FileNotFoundError as e:

    st.error("❌ Model Files Missing")

    st.markdown(
        f"""
        <div class="info-box">
        {str(e)}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()

except ModuleNotFoundError as e:

    st.error("❌ Required Python Package Missing")

    st.markdown(
        f"""
        <div class="info-box">
        <b>Missing module:</b> {e.name}
        <br><br>
        Your saved machine learning model requires this module.
        Add the corresponding package to
        <b>requirements.txt</b> and redeploy the application.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()

except Exception as e:

    st.error("❌ Model Loading Failed")

    st.markdown(
        f"""
        <div class="info-box">
        <b>Error:</b><br>
        {str(e)}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown("""
<div class="card">
    <div class="card-title">📍 Admission Parameters</div>
</div>
""", unsafe_allow_html=True)


col1, col2 = st.columns(2)


with col1:

    year = st.selectbox(
        "📅 Admission Year",
        [2024, 2025, 2026],
        index=1
    )

    figure_kind = st.selectbox(
        "📊 Figure Kind",
        list(le_kind.classes_)
    )


with col2:

    seat_type = st.selectbox(
        "🎯 Seat Allocation Category",
        list(le_seat.classes_)
    )

    basis = st.selectbox(
        "📝 Admission Basis",
        list(le_target.classes_)
    )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔮 Predict Admission Cutoff"):

    try:

        encoded_kind = le_kind.transform(
            [figure_kind]
        )[0]

        encoded_seat = le_seat.transform(
            [seat_type]
        )[0]

        encoded_basis = le_target.transform(
            [basis]
        )[0]

        # Feature order must match training:
        # ['year', 'figure_kind', 'seat_type', 'basis']

        features = np.array([
            [
                year,
                encoded_kind,
                encoded_seat,
                encoded_basis
            ]
        ])

        prediction = model.predict(features)[0]

        if str(figure_kind).lower() == "score":

            result = f"{prediction:.2f}"
            label = "Predicted Cutoff Score"

        else:

            result = f"{int(np.round(prediction)):,}"
            label = "Predicted Cutoff Rank"


        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">
                    🎉 Prediction Complete
                </div>

                <div class="result-value">
                    {result}
                </div>

                <div style="color:#166534; margin-top:5px;">
                    {label}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        st.markdown(
            f"""
            <div class="info-box">
                <b>Prediction Details</b><br><br>
                Admission Year: <b>{year}</b><br>
                Figure Kind: <b>{figure_kind}</b><br>
                Seat Category: <b>{seat_type}</b><br>
                Admission Basis: <b>{basis}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:

        st.error(
            f"❌ Prediction failed: {str(e)}"
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    University Admission Cutoff Predictor
    • Machine Learning Based Forecasting
</div>
""", unsafe_allow_html=True)

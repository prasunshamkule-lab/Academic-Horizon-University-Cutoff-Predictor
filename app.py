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
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
ENCODER_PATH = BASE_DIR / "encoders.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"


# =========================================================
# CUSTOM FRONTEND CSS
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
    background: linear-gradient(135deg, #3730a3, #6366f1);
    padding: 2rem;
    border-radius: 22px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 30px rgba(67, 56, 202, 0.18);
}

.hero h1 {
    font-size: 2.1rem;
    margin-bottom: 0.7rem;
    font-weight: 800;
    color: white;
}

.hero p {
    font-size: 1rem;
    line-height: 1.7;
    color: white;
    margin-bottom: 0;
}

.section-heading {
    font-size: 1.25rem;
    font-weight: 750;
    color: #1e1b4b;
    margin-top: 1.5rem;
    margin-bottom: 0.4rem;
}

.section-description {
    color: #64748b;
    margin-bottom: 1.2rem;
}

.result-card {
    background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
    border: 1px solid #bbf7d0;
    padding: 1.7rem;
    border-radius: 18px;
    text-align: center;
    margin-top: 1.5rem;
}

.result-title {
    color: #166534;
    font-size: 1rem;
    font-weight: 700;
}

.result-value {
    color: #15803d;
    font-size: 2.7rem;
    font-weight: 800;
    margin: 0.5rem 0;
}

.result-label {
    color: #166534;
    font-size: 1rem;
}

.info-box {
    background: #eff6ff;
    border-left: 5px solid #3b82f6;
    padding: 1rem;
    border-radius: 10px;
    color: #1e3a8a;
    margin-top: 1rem;
    line-height: 1.8;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    min-height: 3rem;
    font-size: 1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4338ca, #6366f1);
    color: white;
    border: none;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #3730a3, #4f46e5);
    color: white;
    border: none;
}

div[data-testid="stSelectbox"] label {
    font-weight: 600;
    color: #334155;
}

@media (max-width: 600px) {
    .hero {
        padding: 1.4rem;
    }

    .hero h1 {
        font-size: 1.7rem;
    }

    .result-value {
        font-size: 2.2rem;
    }
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
        Make informed admission decisions using machine learning.
        Select your admission parameters to estimate the expected
        cutoff score or rank.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_assets():

    required_files = {
        "model.pkl": MODEL_PATH,
        "encoders.pkl": ENCODER_PATH,
        "scaler.pkl": SCALER_PATH
    }

    for filename, filepath in required_files.items():
        if not filepath.is_file():
            raise FileNotFoundError(
                f"{filename} was not found. "
                f"Expected location: {filepath}"
            )

    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        with open(ENCODER_PATH, "rb") as file:
            encoders = pickle.load(file)

        with open(SCALER_PATH, "rb") as file:
            scaler = pickle.load(file)

    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"The saved files require the missing Python module "
            f"'{error.name}'. Add its package to requirements.txt "
            f"and redeploy."
        ) from error

    except Exception as error:
        raise RuntimeError(
            f"Could not load the saved ML files: "
            f"{type(error).__name__}: {error}"
        ) from error

    if not isinstance(encoders, dict):
        raise ValueError(
            "encoders.pkl must contain a dictionary of encoders."
        )

    required_encoders = [
        "le_kind",
        "le_seat",
        "le_target"
    ]

    for key in required_encoders:
        if key not in encoders:
            raise ValueError(
                f"Encoder '{key}' is missing from encoders.pkl."
            )

    return model, encoders, scaler


# =========================================================
# INITIALIZE MODEL
# =========================================================

try:
    model, encoders, scaler = load_assets()

    le_kind = encoders["le_kind"]
    le_seat = encoders["le_seat"]
    le_target = encoders["le_target"]

except Exception as error:

    st.error("⚠️ Unable to initialize the prediction model.")

    st.markdown(
        f"""
        <div class="info-box">
            <b>Error type:</b> {type(error).__name__}<br>
            <b>Details:</b> {str(error)}<br><br>
            Check the model files, installed dependencies,
            and Python package versions.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="section-heading">📍 Admission Parameters</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Configure the following parameters to generate a prediction.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    year = st.selectbox(
        "📅 Admission Year",
        options=[2024, 2025, 2026],
        index=2
    )

    figure_kind = st.selectbox(
        "📊 Figure Kind",
        options=list(le_kind.classes_)
    )

with col2:

    seat_type = st.selectbox(
        "🎯 Seat Allocation Category",
        options=list(le_seat.classes_)
    )

    basis = st.selectbox(
        "📝 Admission Basis",
        options=list(le_target.classes_)
    )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔮 Predict Admission Cutoff"):

    try:

        # Encode categorical inputs using the saved encoders.

        encoded_kind = le_kind.transform(
            [figure_kind]
        )[0]

        encoded_seat = le_seat.transform(
            [seat_type]
        )[0]

        encoded_basis = le_target.transform(
            [basis]
        )[0]

        # Maintain the feature order used during training.
        #
        # IMPORTANT:
        # This order must match the original training code.

        features = np.array(
            [[
                year,
                encoded_kind,
                encoded_seat,
                encoded_basis
            ]],
            dtype=float
        )

        # Apply the saved scaler.
        #
        # This assumes scaler.pkl was fitted on these same
        # four input features in the same order.

        if hasattr(scaler, "n_features_in_"):
            expected_features = scaler.n_features_in_

            if expected_features != features.shape[1]:
                raise ValueError(
                    f"The saved scaler expects {expected_features} "
                    f"features, but the application provides "
                    f"{features.shape[1]}. Check the training "
                    f"feature order and scaler configuration."
                )

        scaled_features = scaler.transform(features)

        # Generate prediction.

        prediction = float(
            np.asarray(
                model.predict(scaled_features)
            ).reshape(-1)[0]
        )

        if not np.isfinite(prediction):
            raise ValueError(
                "The model returned an invalid prediction."
            )

        # Format the result.

        if str(figure_kind).strip().lower() == "score":

            result = f"{prediction:,.2f}"
            result_label = "Predicted Cutoff Score"

        else:

            result = f"{int(np.rint(prediction)):,}"
            result_label = "Predicted Cutoff Rank"

        # Display prediction result.

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">
                    🎉 Prediction Complete
                </div>

                <div class="result-value">
                    {result}
                </div>

                <div class="result-label">
                    {result_label}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display prediction summary.

        st.markdown(
            f"""
            <div class="info-box">
                <b>📋 Prediction Summary</b><br>
                Admission Year: <b>{year}</b><br>
                Figure Kind: <b>{figure_kind}</b><br>
                Seat Category: <b>{seat_type}</b><br>
                Admission Basis: <b>{basis}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.caption(
            "This is a machine-learning estimate, not an official "
            "admission cutoff. Actual cutoffs may vary."
        )

    except Exception as error:

        st.error("❌ Prediction failed.")

        st.code(
            f"{type(error).__name__}: {error}",
            language="text"
        )

        st.info(
            "Verify that the saved model, encoders, scaler, "
            "and feature order match the original training pipeline."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    <b>University Admission Cutoff Predictor</b><br>
    Machine Learning Based Forecasting
</div>
""", unsafe_allow_html=True)

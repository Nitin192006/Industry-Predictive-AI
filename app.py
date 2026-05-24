import streamlit as st
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import plotly.graph_objects as go

from src.model import HybridPredictiveMaintenance

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Industry 4.0 AI Monitor",
    layout="wide",
    page_icon="⚙️"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.metric-box {
    background-color: #262730;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #FF4B4B;
}

.metric-box-safe {
    background-color: #262730;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #00CC96;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("⚙️ Equipment Health Analytics Command Center")

st.markdown(
    "Real-time Multimodal Fusion Engine integrating "
    "Sensor Telemetry and Thermal Imaging."
)

# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():

    model = HybridPredictiveMaintenance()

    checkpoint = torch.load(
        "models/hybrid_production_weights.pth",
        map_location="cpu",
        weights_only=False
    )

    # =====================================================
    # CASE 1:
    # checkpoint is a pure state_dict
    # =====================================================

    if isinstance(checkpoint, dict) and "state_dict" not in checkpoint:

        try:
            model.load_state_dict(checkpoint)

        except Exception:

            # Remove DataParallel prefixes if present
            new_state_dict = {
                k.replace("module.", ""): v
                for k, v in checkpoint.items()
            }

            model.load_state_dict(new_state_dict)

    # =====================================================
    # CASE 2:
    # checkpoint contains a state_dict key
    # =====================================================

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

        new_state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(new_state_dict)

    # =====================================================
    # CASE 3:
    # entire model object was saved directly
    # =====================================================

    else:
        model = checkpoint

    model.eval()

    return model

# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = load_model()

except Exception as e:

    st.error(f"Model loading failed: {e}")

    model = None

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Data Ingestion Gateway")

    sensor_file = st.file_uploader(
        "Upload Sensor Log (CSV)",
        type=["csv"]
    )

    image_file = st.file_uploader(
        "Upload Thermal Image",
        type=["jpg", "png", "jpeg"]
    )

    analyze_btn = st.button(
        "Initialize Deep Scan",
        use_container_width=True,
        type="primary"
    )

# =========================================================
# MAIN INFERENCE PIPELINE
# =========================================================

if analyze_btn and sensor_file and image_file and model:

    # =====================================================
    # READ CSV
    # =====================================================

    try:
        df = pd.read_csv(sensor_file)

    except Exception as e:

        st.error(f"CSV loading failed: {e}")

        st.stop()

    # =====================================================
    # REQUIRED SENSOR COLUMNS
    # =====================================================

    sensor_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    # =====================================================
    # VALIDATE COLUMNS
    # =====================================================

    missing_cols = [
        col for col in sensor_cols
        if col not in df.columns
    ]

    if missing_cols:

        st.error(
            f"Missing required columns: {missing_cols}"
        )

        st.stop()

    # =====================================================
    # VALIDATE ROW COUNT
    # =====================================================

    if len(df) < 10:

        st.error(
            "Dataset must contain at least 10 rows."
        )

        st.stop()

    # =====================================================
    # SENSOR PREPROCESSING
    # =====================================================

    sequence = (
        df[sensor_cols]
        .tail(10)
        .values
        .astype("float32")
    )

    sensor_tensor = (
        torch.tensor(sequence)
        .unsqueeze(0)
    )

    # =====================================================
    # IMAGE PREPROCESSING
    # =====================================================

    try:

        image = Image.open(image_file).convert("L")

    except Exception as e:

        st.error(f"Image loading failed: {e}")

        st.stop()

    transform = transforms.Compose([

        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])

    image_tensor = (
        transform(image)
        .unsqueeze(0)
    )

    # =====================================================
    # MODEL INFERENCE
    # =====================================================

    try:

        with torch.no_grad():

            outputs = model(
                sensor_tensor,
                image_tensor
            )

            probs = torch.softmax(outputs, dim=1)

            failure_prob = (
                probs[0, 1].item() * 100
            )

    except Exception as e:

        st.error(f"Inference failed: {e}")

        st.stop()

    # =====================================================
    # DASHBOARD
    # =====================================================

    st.divider()

    col1, col2, col3 = st.columns(3)

    is_critical = failure_prob > 50

    status_class = (
        "metric-box"
        if is_critical
        else "metric-box-safe"
    )

    status_text = (
        "CRITICAL FAILURE IMMINENT"
        if is_critical
        else "NOMINAL OPERATIONS"
    )

    action_text = (
        "INITIATE SHUTDOWN"
        if is_critical
        else "CONTINUE PRODUCTION"
    )

    # =====================================================
    # METRICS
    # =====================================================

    with col1:

        st.markdown(
            f"""
            <div class='{status_class}'>
                <h4>Status</h4>
                <h2>{status_text}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class='{status_class}'>
                <h4>Failure Risk</h4>
                <h2>{failure_prob:.2f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class='{status_class}'>
                <h4>Action</h4>
                <h2>{action_text}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    # =====================================================
    # VISUALIZATION
    # =====================================================

    st.divider()

    viz1, viz2 = st.columns([2, 1])

    # =====================================================
    # SENSOR CHARTS
    # =====================================================

    with viz1:

        st.subheader("Sensor Trends")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=df['Rotational speed [rpm]'].tail(50),
                name="RPM"
            )
        )

        fig.add_trace(
            go.Scatter(
                y=df['Torque [Nm]'].tail(50),
                name="Torque"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=450
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # =====================================================
    # THERMAL IMAGE
    # =====================================================

    with viz2:

        st.subheader("Thermal View")

        st.image(
            image,
            use_container_width=True
        )

# =========================================================
# EMPTY STATE
# =========================================================

else:

    st.info(
        "Upload CSV + Thermal Image and click "
        "'Initialize Deep Scan'"
    )
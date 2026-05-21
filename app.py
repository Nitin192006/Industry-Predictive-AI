import streamlit as st
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import plotly.graph_objects as go
from src.model import HybridPredictiveMaintenance

st.set_page_config(
    page_title="Industry 4.0 AI Monitor",
    layout="wide",
    page_icon="⚙️"
)

st.markdown("""
<style>
.main { background-color: #0E1117; }
.metric-box { background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B; }
.metric-box-safe { background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #00CC96; }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Equipment Health Analytics Command Center")
st.markdown("Real-time Multimodal Fusion Engine integrating Sensor Telemetry and Thermal Imaging.")

# =========================================================
# MODEL LOADING (FIXED FOR PYTORCH 2.6)
# =========================================================
@st.cache_resource
def load_model():
    model = HybridPredictiveMaintenance()

    state_dict = torch.load(
        "models/hybrid_production_weights.pth",
        map_location="cpu",
        weights_only=True   # IMPORTANT FIX
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

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
    sensor_file = st.file_uploader("Upload Sensor Log (CSV)", type=["csv"])
    image_file = st.file_uploader("Upload Thermal Image", type=["jpg", "png", "jpeg"])
    analyze_btn = st.button("Initialize Deep Scan", use_container_width=True, type="primary")

# =========================================================
# INFERENCE
# =========================================================
if analyze_btn and sensor_file and image_file and model:

    df = pd.read_csv(sensor_file)

    sensor_cols = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    if any(col not in df.columns for col in sensor_cols):
        st.error("Missing required sensor columns.")
        st.stop()

    if len(df) < 10:
        st.error("Need at least 10 rows.")
        st.stop()

    sequence = df[sensor_cols].tail(10).values.astype("float32")
    sensor_tensor = torch.tensor(sequence).unsqueeze(0)

    image = Image.open(image_file).convert("L")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(sensor_tensor, image_tensor)
        probs = torch.softmax(outputs, dim=1)
        failure_prob = probs[0, 1].item() * 100

    # =====================================================
    # DASHBOARD
    # =====================================================
    st.divider()

    col1, col2, col3 = st.columns(3)

    status_class = "metric-box" if failure_prob > 50 else "metric-box-safe"
    status_text = "CRITICAL FAILURE IMMINENT" if failure_prob > 50 else "NOMINAL OPERATIONS"
    action_text = "INITIATE SHUTDOWN" if failure_prob > 50 else "CONTINUE PRODUCTION"

    with col1:
        st.markdown(f"<div class='{status_class}'><h4>Status</h4><h2>{status_text}</h2></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='{status_class}'><h4>Failure Risk</h4><h2>{failure_prob:.2f}%</h2></div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<div class='{status_class}'><h4>Action</h4><h2>{action_text}</h2></div>", unsafe_allow_html=True)

    st.divider()

    viz1, viz2 = st.columns([2, 1])

    with viz1:
        st.subheader("Sensor Trends")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['Rotational speed [rpm]'].tail(50), name="RPM"))
        fig.add_trace(go.Scatter(y=df['Torque [Nm]'].tail(50), name="Torque"))
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with viz2:
        st.subheader("Thermal View")
        st.image(image, use_container_width=True)

else:
    st.info("Upload CSV + Image and click Initialize Deep Scan")
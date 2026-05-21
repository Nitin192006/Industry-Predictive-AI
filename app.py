import streamlit as st
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
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
.main { background-color: #0E1117; }
.metric-box { background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #FF4B4B; }
.metric-box-safe { background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #00CC96; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("⚙️ Equipment Health Analytics Command Center")
st.markdown("Real-time Multimodal Fusion Engine integrating Sensor Telemetry and Thermal Imaging.")

# =========================================================
# LOAD MODEL FROM HUGGING FACE HUB
# =========================================================
@st.cache_resource
def load_production_model():
    model = HybridPredictiveMaintenance()
    
    # Download model from HF Hub
    model_file = hf_hub_download(
        repo_id="Nitin190606/hybrid_production_weights",
        filename="hybrid_production_weights.pth",
        use_auth_token=True
    )
    
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.eval()
    return model

model = None
try:
    model = load_production_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Data Ingestion Gateway")
    st.info("Upload the AI4I formatted sensor CSV and an accompanying thermal image.")
    sensor_file = st.file_uploader("Upload Sensor Log (CSV)", type=["csv"])
    image_file = st.file_uploader("Upload Component Thermal Scan", type=["jpg", "png", "jpeg"])
    analyze_btn = st.button("Initialize Deep Scan", use_container_width=True, type="primary")

# =========================================================
# MAIN INFERENCE PIPELINE
# =========================================================
if analyze_btn and sensor_file and image_file:
    df = pd.read_csv(sensor_file)
    sensor_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                   'Torque [Nm]', 'Tool wear [min]']
    
    missing_cols = [col for col in sensor_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        st.stop()
    
    if len(df) < 10:
        st.error("CSV must contain at least 10 rows.")
        st.stop()
    
    sequence = df[sensor_cols].tail(10).values
    sensor_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
    
    # Image processing
    image = Image.open(image_file).convert('L')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    with st.spinner('Neural networks aligning modalities...'):
        with torch.no_grad():
            outputs = model(sensor_tensor, image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            failure_prob = probabilities[0][1].item() * 100
    
    # DASHBOARD
    st.divider()
    col1, col2, col3 = st.columns(3)
    status_class = "metric-box" if failure_prob > 50 else "metric-box-safe"
    status_text = "CRITICAL FAILURE IMMINENT" if failure_prob > 50 else "NOMINAL OPERATIONS"
    action_text = "INITIATE EMERGENCY SHUTDOWN" if failure_prob > 50 else "CONTINUE PRODUCTION"
    
    with col1:
        st.markdown(f"<div class='{status_class}'><h4>System Status</h4><h2>{status_text}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='{status_class}'><h4>Anomaly Probability</h4><h2>{failure_prob:.2f}%</h2></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='{status_class}'><h4>Maintenance Directive</h4><h2>{action_text}</h2></div>", unsafe_allow_html=True)
    
    st.divider()
    
    # VISUALIZATION
    viz1, viz2 = st.columns([2, 1])
    
    with viz1:
        st.subheader("High-Frequency Sensor Dynamics")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['Rotational speed [rpm]'].tail(50), name="RPM", line=dict(color="#00CC96")))
        fig.add_trace(go.Scatter(y=df['Torque [Nm]'].tail(50), name="Torque", line=dict(color="#FF4B4B")))
        fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    with viz2:
        st.subheader("Spatial Thermal Gradient")
        st.image(image, use_container_width=True)
else:
    st.info("Upload both:\n- Sensor CSV\n- Thermal Image\n\nThen click 'Initialize Deep Scan'")
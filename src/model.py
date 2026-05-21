import torch
import torch.nn as nn
import torchvision.models as models

# =========================================================
# 1. SENSOR LSTM ENCODER
# =========================================================

class SensorLSTM(nn.Module):

    def __init__(
        self,
        num_sensors=5,
        hidden_size=64,
        num_layers=2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):

        # hidden shape:
        # (num_layers, batch_size, hidden_size)

        _, (hidden, _) = self.lstm(x)

        # Return final layer hidden state
        return hidden[-1]

# =========================================================
# 2. THERMAL IMAGE ENCODER
# =========================================================

class ThermalResNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Modern torchvision syntax
        self.resnet = models.resnet18(weights=None)

        # =================================================
        # MODIFY FIRST LAYER FOR GRAYSCALE INPUT
        # =================================================

        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # =================================================
        # REMOVE FINAL CLASSIFICATION LAYER
        # =================================================

        self.resnet.fc = nn.Identity()

    def forward(self, x):

        return self.resnet(x)

# =========================================================
# 3. HYBRID MULTIMODAL MODEL
# =========================================================

class HybridPredictiveMaintenance(nn.Module):

    def __init__(
        self,
        num_sensors=5,
        lstm_hidden=64,
        num_classes=2
    ):

        super().__init__()

        # Sensor branch
        self.sensor_encoder = SensorLSTM(
            num_sensors=num_sensors,
            hidden_size=lstm_hidden
        )

        # Thermal image branch
        self.thermal_encoder = ThermalResNet()

        # Total fused feature size
        fused_dim = 512 + lstm_hidden

        # Classification head
        self.classifier = nn.Sequential(

            nn.Linear(fused_dim, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, sensor_data, thermal_image):

        # =================================================
        # SENSOR FEATURES
        # =================================================
        sensor_features = self.sensor_encoder(sensor_data)

        # =================================================
        # IMAGE FEATURES
        # =================================================
        thermal_features = self.thermal_encoder(thermal_image)

        # =================================================
        # FEATURE FUSION
        # =================================================
        fused_features = torch.cat(
            (sensor_features, thermal_features),
            dim=1
        )

        # =================================================
        # CLASSIFICATION
        # =================================================
        output = self.classifier(fused_features)

        return output

# =========================================================
# 4. MODEL TEST
# =========================================================

if __name__ == "__main__":

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create model
    model = HybridPredictiveMaintenance().to(device)

    print(model)

    # =====================================================
    # DUMMY INPUTS
    # =====================================================

    batch_size = 8
    seq_length = 10
    num_sensors = 5

    # Sensor input:
    # shape = (batch, sequence_length, num_sensors)
    sensor_input = torch.randn(
        batch_size,
        seq_length,
        num_sensors
    ).to(device)

    # Image input:
    # shape = (batch, channels, height, width)
    image_input = torch.randn(
        batch_size,
        1,
        224,
        224
    ).to(device)

    # Forward pass
    outputs = model(sensor_input, image_input)

    print("\nOutput Shape:", outputs.shape)
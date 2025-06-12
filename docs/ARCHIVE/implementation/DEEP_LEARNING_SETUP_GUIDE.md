# Deep Learning Implementation Setup Guide

## 🚀 Quick Start Guide for Task C1 Implementation

This guide provides everything needed to implement the Enhanced U-Net and Hybrid ViT-UNet architectures for kelp detection based on our comprehensive research analysis.

## 📋 Pre-Implementation Checklist

### ✅ Required Accounts & Access
- [ ] **Cloud Provider**: AWS, GCP, or Azure account with GPU quotas
- [ ] **MLOps Platforms**: MLflow (self-hosted) or Weights & Biases account
- [ ] **Data Access**: Google Earth Engine account for Sentinel-2 data
- [ ] **Version Control**: Git LFS setup for large model files
- [ ] **Container Registry**: Docker Hub or cloud registry access

### ✅ Budget Allocation
- **Development Phase**: $750-1,200 (3-4 weeks)
- **Monthly Production**: $130-380 (auto-scaling inference)
- **Emergency Buffer**: +20% for unexpected costs

## 🛠️ Development Environment Setup

### Option 1: Local Development (RTX 4090/A100)
```bash
# Create and activate conda environment
conda create -n kelp-dl python=3.9
conda activate kelp-dl

# Core PyTorch ecosystem
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Computer Vision & Segmentation
pip install timm==0.9.16                    # Pre-trained models
pip install segmentation-models-pytorch==0.3.3  # U-Net implementations
pip install albumentations==1.3.1           # Data augmentation
pip install opencv-python==4.8.1.78         # Image processing

# Transformers & SAM
pip install transformers==4.35.2            # Hugging Face transformers
pip install segment-anything==1.0           # Meta SAM
pip install samgeo==0.15.3                  # Geospatial SAM wrapper
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Satellite Imagery Processing
pip install rasterio==1.3.9                 # Geospatial raster I/O
pip install xarray==2023.10.1               # Multi-dimensional arrays
pip install dask==2023.10.1                 # Distributed computing
pip install earthengine-api==0.1.376        # Google Earth Engine
pip install geopandas==0.14.1               # Vector data processing

# MLOps & Monitoring
pip install mlflow==2.8.1                   # Experiment tracking
pip install wandb==0.16.0                   # Visualization
pip install dvc==3.27.0                     # Data version control
pip install hydra-core==1.3.2               # Configuration management

# Deployment & Optimization
pip install fastapi==0.104.1                # API framework
pip install uvicorn==0.24.0                 # ASGI server
pip install onnx==1.15.0                    # Model optimization
pip install tensorrt                        # NVIDIA optimization (if available)

# Development Tools
pip install jupyter==1.0.0                  # Interactive development
pip install matplotlib==3.8.2               # Visualization
pip install seaborn==0.13.0                 # Statistical plots
pip install plotly==5.17.0                  # Interactive plots
```

### Option 2: Cloud Development (Docker)
```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libgdal-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SAM2 from source
RUN git clone https://github.com/facebookresearch/segment-anything-2.git && \
    cd segment-anything-2 && \
    pip install -e .

# Configure Git LFS and MLflow
RUN git lfs install
ENV MLFLOW_TRACKING_URI=/workspace/mlruns

EXPOSE 8888 6006 5000

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  kelp-dl-dev:
    build: .
    volumes:
      - ./:/workspace
      - ./data:/workspace/data
      - ./models:/workspace/models
    ports:
      - "8888:8888"  # Jupyter Lab
      - "6006:6006"  # TensorBoard
      - "5000:5000"  # MLflow
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
      - GOOGLE_APPLICATION_CREDENTIALS=/workspace/gee-credentials.json
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## 📁 Project Structure Setup

```bash
# Create project structure
mkdir -p kelp-detection-dl/{
src/{models,data,training,inference,utils},
configs/{training,model,data},
data/{raw,processed,labels,splits},
models/{checkpoints,onnx,tensorrt},
notebooks/{exploration,training,evaluation},
scripts/{training,preprocessing,deployment},
tests/{unit,integration,performance},
docs/{api,deployment,model_cards}
}

# Initialize key configuration files
touch kelp-detection-dl/configs/training/unet_config.yaml
touch kelp-detection-dl/configs/training/vit_unet_config.yaml
touch kelp-detection-dl/configs/model/enhanced_unet.yaml
touch kelp-detection-dl/configs/model/vit_unet_hybrid.yaml
```

## 🎯 Implementation Phase Breakdown

### Phase 1: Enhanced U-Net (Week 1-2)

#### 1.1 Data Pipeline Setup
```bash
# Download Sentinel-2 training data
python scripts/download_sentinel2_data.py \
    --regions "kelp_farm_locations.geojson" \
    --date_range "2023-06-01,2023-09-30" \
    --cloud_threshold 10 \
    --output_dir "data/raw/sentinel2"

# Generate SAM-assisted labels
python scripts/generate_sam_labels.py \
    --input_dir "data/raw/sentinel2" \
    --sam_model "segment-anything/sam_vit_h_4b8939.pth" \
    --output_dir "data/labels/sam_assisted"
```

#### 1.2 Enhanced U-Net Implementation
```python
# src/models/enhanced_unet.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from timm import create_model

class EnhancedUNet(nn.Module):
    def __init__(self,
                 encoder_name='resnet50',
                 encoder_weights='imagenet',
                 in_channels=5,  # RGB + NIR + Red-edge
                 classes=1,
                 attention_type='scse'):
        super().__init__()

        self.unet = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            attention_type=attention_type
        )

        # Spectral band attention
        self.spectral_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply spectral attention
        spectral_weights = self.spectral_attention(x)
        x = x * spectral_weights

        # U-Net forward pass
        return self.unet(x)
```

#### 1.3 Training Configuration
```yaml
# configs/training/enhanced_unet.yaml
model:
  name: "enhanced_unet"
  encoder_name: "resnet50"
  encoder_weights: "imagenet"
  in_channels: 5
  classes: 1
  attention_type: "scse"

data:
  train_dir: "data/processed/train"
  val_dir: "data/processed/val"
  batch_size: 16
  num_workers: 8
  augmentations:
    - RandomCrop: {height: 512, width: 512}
    - HorizontalFlip: {p: 0.5}
    - VerticalFlip: {p: 0.5}
    - Rotate: {limit: 90, p: 0.5}
    - ColorJitter: {brightness: 0.2, contrast: 0.2}

training:
  optimizer: "AdamW"
  learning_rate: 1e-4
  weight_decay: 1e-5
  scheduler: "CosineAnnealingLR"
  epochs: 100
  early_stopping_patience: 15
  mixed_precision: true

loss:
  primary: "DiceLoss"
  secondary: "FocalLoss"
  weights: [0.7, 0.3]

metrics:
  - "DiceCoefficient"
  - "IoU"
  - "Precision"
  - "Recall"
  - "F1Score"

logging:
  mlflow:
    experiment_name: "enhanced_unet_kelp_detection"
    run_name: "unet_resnet50_v1"
  wandb:
    project: "kelp-detection"
    entity: "your-username"
```

### Phase 2: Hybrid ViT-UNet (Week 2-3)

#### 2.1 ViT-UNet Architecture
```python
# src/models/vit_unet_hybrid.py
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from timm.models.layers import Involution2d

class ViTUNetHybrid(nn.Module):
    def __init__(self,
                 vit_model_name='google/vit-base-patch16-224',
                 in_channels=5,
                 num_classes=1,
                 img_size=512):
        super().__init__()

        # ViT Encoder Configuration
        vit_config = ViTConfig.from_pretrained(vit_model_name)
        vit_config.image_size = img_size
        vit_config.num_channels = in_channels

        self.vit_encoder = ViTModel(vit_config)

        # Involution-based spectral processing
        self.involution_layers = nn.ModuleList([
            Involution2d(in_channels, kernel_size=7, stride=1),
            Involution2d(in_channels, kernel_size=5, stride=1),
            Involution2d(in_channels, kernel_size=3, stride=1)
        ])

        # U-Net Decoder
        hidden_size = vit_config.hidden_size
        self.decoder = UNetDecoder(
            encoder_channels=[hidden_size//4, hidden_size//2, hidden_size],
            decoder_channels=[256, 128, 64, 32],
            n_blocks=4
        )

        # Final classification head
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Multi-scale involution processing
        inv_features = []
        for inv_layer in self.involution_layers:
            inv_features.append(inv_layer(x))

        # Combine involution features
        spectral_features = torch.cat(inv_features, dim=1)

        # ViT encoding
        vit_outputs = self.vit_encoder(spectral_features)
        hidden_states = vit_outputs.last_hidden_state

        # Reshape for decoder
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_dim = hidden_states.shape[2]

        # Assuming square patches
        patch_size = int(seq_len ** 0.5)
        encoder_features = hidden_states.view(
            batch_size, patch_size, patch_size, hidden_dim
        ).permute(0, 3, 1, 2)

        # U-Net decoding
        decoder_output = self.decoder(encoder_features)

        # Final prediction
        return self.final_conv(decoder_output)
```

### Phase 3: SAM Integration & Validation (Week 3-4)

#### 3.1 SAM-based Validation Pipeline
```python
# src/validation/sam_validation.py
import torch
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

class SAMValidator:
    def __init__(self, sam_checkpoint_path, model_type="vit_h"):
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
        self.predictor = SamPredictor(self.sam)

    def validate_predictions(self, image, model_prediction, confidence_threshold=0.5):
        """Compare model predictions with SAM zero-shot results"""
        self.predictor.set_image(image)

        # Generate points from model prediction
        prediction_mask = model_prediction > confidence_threshold
        y_coords, x_coords = np.where(prediction_mask)

        if len(y_coords) == 0:
            return {"sam_mask": None, "iou": 0.0, "agreement": 0.0}

        # Sample points for SAM prompting
        n_points = min(10, len(y_coords))
        indices = np.random.choice(len(y_coords), n_points, replace=False)
        input_points = np.array([[x_coords[i], y_coords[i]] for i in indices])
        input_labels = np.ones(n_points)

        # Get SAM prediction
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

        # Select best mask
        best_mask = masks[np.argmax(scores)]

        # Calculate metrics
        intersection = np.logical_and(prediction_mask, best_mask).sum()
        union = np.logical_or(prediction_mask, best_mask).sum()
        iou = intersection / union if union > 0 else 0

        return {
            "sam_mask": best_mask,
            "iou": iou,
            "agreement": intersection / prediction_mask.sum() if prediction_mask.sum() > 0 else 0
        }
```

## 🚀 Training Scripts

### Enhanced U-Net Training
```python
# scripts/train_enhanced_unet.py
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
import mlflow
import wandb
from src.models.enhanced_unet import EnhancedUNet
from src.data.kelp_dataset import KelpDataset
from src.training.trainer import Trainer

@hydra.main(config_path="../configs/training", config_name="enhanced_unet")
def train_enhanced_unet(cfg: DictConfig):
    # Initialize tracking
    mlflow.set_experiment(cfg.logging.mlflow.experiment_name)
    wandb.init(project=cfg.logging.wandb.project, config=cfg)

    # Setup model
    model = EnhancedUNet(
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
        attention_type=cfg.model.attention_type
    )

    # Setup data
    train_dataset = KelpDataset(cfg.data.train_dir, cfg.data.augmentations)
    val_dataset = KelpDataset(cfg.data.val_dir, None)

    # Setup trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cfg
    )

    # Train model
    trainer.train()

if __name__ == "__main__":
    train_enhanced_unet()
```

## 📊 Monitoring & Deployment

### Model Performance Dashboard
```python
# src/monitoring/performance_dashboard.py
import streamlit as st
import mlflow
import pandas as pd
import plotly.express as px

def create_performance_dashboard():
    st.title("Kelp Detection Model Performance")

    # Load experiment data
    experiments = mlflow.search_experiments()
    runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id for exp in experiments])

    # Performance comparison
    fig = px.scatter(
        runs_df,
        x='metrics.val_iou',
        y='metrics.val_dice',
        color='tags.model_type',
        hover_data=['metrics.train_time', 'params.learning_rate']
    )
    st.plotly_chart(fig)

    # Cost analysis
    cost_analysis = calculate_training_costs(runs_df)
    st.write("Training Cost Analysis:", cost_analysis)

def calculate_training_costs(runs_df):
    """Calculate estimated training costs based on runtime and GPU type"""
    # A100 cost: ~$3.00/hour, RTX 4090 cost: ~$0.50/hour
    gpu_costs = {"a100": 3.0, "rtx4090": 0.5}

    total_cost = 0
    for _, run in runs_df.iterrows():
        gpu_type = run.get('tags.gpu_type', 'rtx4090')
        runtime_hours = run.get('metrics.train_time', 0) / 3600
        cost = runtime_hours * gpu_costs.get(gpu_type, 0.5)
        total_cost += cost

    return {"total_cost": total_cost, "avg_cost_per_run": total_cost / len(runs_df)}
```

### Production Deployment Configuration
```yaml
# deployment/kubernetes/kelp-detection-api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kelp-detection-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: kelp-detection-api
  template:
    metadata:
      labels:
        app: kelp-detection-api
    spec:
      containers:
      - name: api
        image: kelp-detection:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "2"
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
        env:
        - name: MODEL_PATH
          value: "/models/enhanced_unet_best.pth"
        - name: BATCH_SIZE
          value: "4"
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage
---
apiVersion: v1
kind: Service
metadata:
  name: kelp-detection-service
spec:
  selector:
    app: kelp-detection-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🔍 Testing & Validation Framework

### Automated Testing Pipeline
```python
# tests/integration/test_full_pipeline.py
import pytest
import torch
import numpy as np
from src.models.enhanced_unet import EnhancedUNet
from src.models.vit_unet_hybrid import ViTUNetHybrid
from src.validation.sam_validation import SAMValidator

class TestFullPipeline:
    def test_enhanced_unet_inference(self):
        model = EnhancedUNet(in_channels=5, classes=1)
        input_tensor = torch.randn(1, 5, 512, 512)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (1, 1, 512, 512)
        assert torch.all(torch.sigmoid(output) >= 0)
        assert torch.all(torch.sigmoid(output) <= 1)

    def test_vit_unet_inference(self):
        model = ViTUNetHybrid(in_channels=5, num_classes=1)
        input_tensor = torch.randn(1, 5, 512, 512)

        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape == (1, 1, 512, 512)

    def test_sam_validation(self):
        validator = SAMValidator("path/to/sam_checkpoint.pth")

        # Mock data
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        prediction = np.random.rand(512, 512)

        result = validator.validate_predictions(image, prediction)

        assert "iou" in result
        assert "agreement" in result
        assert 0 <= result["iou"] <= 1
```

## 📈 Performance Benchmarking

### Model Comparison Framework
```python
# scripts/benchmark_models.py
import time
import torch
import numpy as np
from src.models.enhanced_unet import EnhancedUNet
from src.models.vit_unet_hybrid import ViTUNetHybrid

def benchmark_models():
    models = {
        "enhanced_unet": EnhancedUNet(in_channels=5, classes=1),
        "vit_unet_hybrid": ViTUNetHybrid(in_channels=5, num_classes=1)
    }

    input_tensor = torch.randn(1, 5, 512, 512)

    results = {}
    for name, model in models.items():
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                output = model(input_tensor)
                times.append(time.time() - start)

        results[name] = {
            "avg_inference_time": np.mean(times),
            "std_inference_time": np.std(times),
            "parameters": sum(p.numel() for p in model.parameters()),
            "memory_usage": torch.cuda.max_memory_allocated() / 1024**3  # GB
        }

    return results
```

This comprehensive setup guide provides everything needed for successful implementation of the deep learning pipeline with proper cost management, technical infrastructure, and validation frameworks. Future agents can follow this guide to implement state-of-the-art kelp detection capabilities while maintaining budget control and production readiness.

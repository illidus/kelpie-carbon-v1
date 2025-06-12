"""SKEMA CNN Model Implementation.

This module implements the core CNN architecture for kelp detection based on
the Mask R-CNN framework used in SKEMA research from University of Victoria.

References:
- Marquez et al. (2022): Mask R-CNN for giant kelp forests
- Gendall et al. (2023): Multi-satellite mapping framework
- UVic SPECTRAL Lab SKeMa project specifications

Task C1.1: Research SKEMA CNN architecture specifics
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from ..logging_config import get_logger

logger = get_logger(__name__)


class SKEMACNNModel(nn.Module):
    """Core SKEMA CNN architecture for kelp detection.
    
    Based on research specifications from UVic SPECTRAL Lab and published
    Mask R-CNN implementations for kelp forest detection.
    """
    
    def __init__(
        self,
        num_classes: int = 2,  # kelp vs background
        input_channels: int = 4,  # RGB + NIR typical for satellite imagery
        pretrained: bool = True,
        backbone_name: str = 'resnet50'
    ):
        """Initialize SKEMA CNN model.
        
        Args:
            num_classes: Number of output classes (2 for kelp/background)
            input_channels: Input channels (RGB, NIR, red-edge, etc.)
            pretrained: Use pre-trained COCO weights
            backbone_name: Backbone architecture name
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.backbone_name = backbone_name
        
        # Initialize backbone based on research specifications
        if backbone_name == 'resnet50':
            self.backbone = self._create_resnet50_backbone(pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Feature extraction layers
        self.feature_extractor = self._create_feature_extractor()
        
        # Classification head
        self.classifier = self._create_classification_head()
        
        # Segmentation head for pixel-level kelp detection
        self.segmentation_head = self._create_segmentation_head()
        
        logger.info(f"Initialized SKEMA CNN with {num_classes} classes, "
                   f"{input_channels} input channels, backbone: {backbone_name}")
    
    def _create_resnet50_backbone(self, pretrained: bool) -> nn.Module:
        """Create ResNet50 backbone with modifications for satellite imagery."""
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer for multi-channel satellite imagery
        if self.input_channels != 3:
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                self.input_channels, 
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Initialize new channels with appropriate weights
            with torch.no_grad():
                if self.input_channels > 3:
                    # Copy RGB weights and initialize additional channels
                    resnet.conv1.weight[:, :3] = original_conv.weight
                    # Initialize additional channels (NIR, red-edge) with green channel weights
                    for i in range(3, self.input_channels):
                        resnet.conv1.weight[:, i] = original_conv.weight[:, 1]  # Green channel
                else:
                    # Handle case with fewer than 3 channels
                    resnet.conv1.weight[:, :self.input_channels] = original_conv.weight[:, :self.input_channels]
        
        # Remove final classification layers (we'll add our own)
        backbone_layers = list(resnet.children())[:-2]  # Remove avgpool and fc
        return nn.Sequential(*backbone_layers)
    
    def _create_feature_extractor(self) -> nn.Module:
        """Create feature extraction layers following SKEMA specifications."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def _create_classification_head(self) -> nn.Module:
        """Create classification head for kelp detection."""
        return nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, self.num_classes)
        )
    
    def _create_segmentation_head(self) -> nn.Module:
        """Create segmentation head for pixel-level kelp detection."""
        return nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, self.num_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through SKEMA CNN.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Dictionary containing classification and segmentation outputs
        """
        # Extract features through backbone
        backbone_features = self.backbone(x)
        
        # Global features for classification
        global_features = self.feature_extractor(backbone_features)
        
        # Classification output
        classification_logits = self.classifier(global_features)
        
        # Segmentation output
        segmentation_logits = self.segmentation_head(backbone_features)
        
        # Resize segmentation to match input size
        segmentation_logits = F.interpolate(
            segmentation_logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return {
            'classification': classification_logits,
            'segmentation': segmentation_logits,
            'features': global_features
        }


class MaskRCNNKelpDetector(nn.Module):
    """Mask R-CNN implementation for kelp detection.
    
    Based on the specific architecture used in Marquez et al. (2022)
    for kelp forest detection with 87% Jaccard index performance.
    """
    
    def __init__(
        self,
        num_classes: int = 2,  # background + kelp
        pretrained: bool = True,
        pretrained_backbone: bool = True
    ):
        """Initialize Mask R-CNN model for kelp detection.
        
        Args:
            num_classes: Number of classes (background + kelp)
            pretrained: Load pre-trained COCO weights
            pretrained_backbone: Use pre-trained backbone
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pre-trained Mask R-CNN model
        self.model = maskrcnn_resnet50_fpn(
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone
        )
        
        # Modify the classifier head for kelp detection
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        # Modify the mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )
        
        logger.info(f"Initialized Mask R-CNN for kelp detection with {num_classes} classes")
    
    def forward(self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None):
        """Forward pass through Mask R-CNN.
        
        Args:
            images: List of input images
            targets: Training targets (optional, for training mode)
            
        Returns:
            Model predictions or losses (training mode)
        """
        return self.model(images, targets)
    
    def predict(self, images: list[torch.Tensor], confidence_threshold: float = 0.5) -> list[dict[str, torch.Tensor]]:
        """Generate predictions for kelp detection.
        
        Args:
            images: List of input images
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of predictions for each image
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        
        # Filter predictions by confidence threshold
        filtered_predictions = []
        for pred in predictions:
            mask = pred['scores'] > confidence_threshold
            filtered_pred = {
                'boxes': pred['boxes'][mask],
                'labels': pred['labels'][mask],
                'scores': pred['scores'][mask],
                'masks': pred['masks'][mask]
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions


def train_skema_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cuda'
) -> dict[str, list[float]]:
    """Train SKEMA CNN model.
    
    Args:
        model: SKEMA model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Training device
        
    Returns:
        Training history dictionary
    """
    model = model.to(device)
    model.train()
    
    # Optimizer following research specifications
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Loss functions
    classification_criterion = nn.CrossEntropyLoss()
    segmentation_criterion = nn.BCEWithLogitsLoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    logger.info(f"Starting SKEMA model training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            classification_targets = targets['classification'].to(device)
            segmentation_targets = targets['segmentation'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate losses
            cls_loss = classification_criterion(outputs['classification'], classification_targets)
            seg_loss = segmentation_criterion(outputs['segmentation'], segmentation_targets)
            total_loss = cls_loss + seg_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs['classification'].data, 1)
            train_total += classification_targets.size(0)
            train_correct += (predicted == classification_targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                classification_targets = targets['classification'].to(device)
                segmentation_targets = targets['segmentation'].to(device)
                
                outputs = model(images)
                
                cls_loss = classification_criterion(outputs['classification'], classification_targets)
                seg_loss = segmentation_criterion(outputs['segmentation'], segmentation_targets)
                total_loss = cls_loss + seg_loss
                
                val_loss += total_loss.item()
                
                _, predicted = torch.max(outputs['classification'].data, 1)
                val_total += classification_targets.size(0)
                val_correct += (predicted == classification_targets).sum().item()
        
        # Update learning rate
        scheduler.step()
        
        # Calculate metrics
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                   f"Train Loss: {train_loss/len(train_loader):.4f}, "
                   f"Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss/len(val_loader):.4f}, "
                   f"Val Acc: {val_acc:.2f}%")
    
    logger.info("SKEMA model training completed")
    return history


def load_pretrained_skema_model(
    model_path: str,
    model_type: str = 'skema_cnn',
    device: str = 'cuda'
) -> nn.Module:
    """Load pre-trained SKEMA model.
    
    Args:
        model_path: Path to saved model
        model_type: Type of model ('skema_cnn' or 'mask_rcnn')
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if model_type == 'skema_cnn':
        model = SKEMACNNModel()
    elif model_type == 'mask_rcnn':
        model = MaskRCNNKelpDetector()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded pre-trained {model_type} model from {model_path}")
    return model 

"""Deep Learning Module for SKEMA Kelp Detection.

This module implements budget-friendly deep learning approaches for kelp detection,
including zero-cost SAM-based detection and traditional CNN methods.

Task C1: Enhanced SKEMA Deep Learning Integration (Budget-Friendly Approach)
"""

# Budget-friendly implementations (zero training cost)
from .budget_sam_detector import (
    BudgetSAMKelpDetector,
    download_sam_model
)

from .budget_unet_detector import (
    BudgetUNetKelpDetector,
    setup_budget_unet_environment
)

from .classical_ml_enhancer import (
    ClassicalMLEnhancer,
    setup_classical_ml_environment
)

# Traditional CNN implementations (training required)
# Note: These modules will be implemented as needed
# from .skema_cnn import (
#     SKEMACNNModel,
#     MaskRCNNKelpDetector,
#     train_skema_model,
#     load_pretrained_skema_model
# )

# Data pipeline implementations (future)
# from .data_pipeline import (
#     KelpDatasetLoader,
#     SatelliteImageProcessor,
#     prepare_training_data
# )

# Model training implementations (future)
# from .model_training import (
#     SKEMATrainer,
#     evaluate_model_performance,
#     generate_model_predictions
# )

__all__ = [
    # Budget-friendly (zero cost)
    'BudgetSAMKelpDetector',
    'download_sam_model',
    'BudgetUNetKelpDetector',
    'setup_budget_unet_environment',
    'ClassicalMLEnhancer',
    'setup_classical_ml_environment',
    
    # Traditional CNN (future implementation)
    # 'SKEMACNNModel',
    # 'MaskRCNNKelpDetector', 
    # 'train_skema_model',
    # 'load_pretrained_skema_model',
    
    # Data pipeline (future implementation)
    # 'KelpDatasetLoader',
    # 'SatelliteImageProcessor',
    # 'prepare_training_data',
    
    # Model training (future implementation)
    # 'SKEMATrainer',
    # 'evaluate_model_performance',
    # 'generate_model_predictions'
] 
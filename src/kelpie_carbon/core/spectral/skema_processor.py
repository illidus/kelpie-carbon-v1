"""
SKEMA Processor - Integration Interface for Spectral Analysis

This module provides a unified interface to existing SKEMA spectral analysis
capabilities for integration with deep learning models like SAM.
"""

from typing import Any

import numpy as np
import xarray as xr

from ..indices import calculate_indices_from_dataset
from ..core.mask import create_skema_kelp_detection_mask
from ..processing.water_anomaly_filter import WaterAnomalyFilter

# For now, we'll skip the complex config loading and use simple defaults
# from ..config import get_config


class SKEMAProcessor:
    """
    SKEMA Spectral Processor for kelp detection.
    
    Provides a unified interface to existing SKEMA spectral analysis capabilities,
    optimized for integration with deep learning models like SAM.
    """
    
    def __init__(self, config_path: str | None = None):
        """
        Initialize SKEMA processor.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        # Use optimized thresholds from Task A2.7 optimization work
        self.config = {
            "kelp_detection": {
                "ndre_threshold": 0.04,  # Optimized from 0.15 to 0.04
                "ndvi_threshold": 0.1,   # Optimized from 0.2 to 0.1
                "fai_threshold": 0.02,
                "apply_waf": True
            },
            "waf_config": {
                "sunglint_threshold": 0.15,
                "kernel_size": 5,
                "spectral_smoothing": True
            }
        }
        
        # Initialize Water Anomaly Filter
        self.waf = WaterAnomalyFilter(self.config.get("waf_config"))
    
    def process_satellite_data(self, dataset: xr.Dataset) -> dict[str, Any]:
        """
        Process satellite data through SKEMA pipeline.
        
        Args:
            dataset: xarray Dataset with satellite bands
            
        Returns:
            Dictionary containing processed results and spectral indices
        """
        # Apply Water Anomaly Filter
        if self.config["kelp_detection"].get("apply_waf", True):
            filtered_dataset = self.waf.apply_filter(dataset)
        else:
            filtered_dataset = dataset
        
        # Calculate spectral indices
        indices = calculate_indices_from_dataset(filtered_dataset)
        
        # Create kelp detection mask
        kelp_mask = create_skema_kelp_detection_mask(
            filtered_dataset, self.config["kelp_detection"]
        )
        
        return {
            "dataset": filtered_dataset,
            "indices": indices,
            "kelp_mask": kelp_mask,
            "config": self.config
        }
    
    def calculate_spectral_indices(self, 
                                 rgb_array: np.ndarray,
                                 nir_array: np.ndarray | None = None,
                                 red_edge_array: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """
        Calculate spectral indices from numpy arrays.
        
        Args:
            rgb_array: RGB array (H, W, 3) with bands [R, G, B]
            nir_array: Near-infrared array (H, W)
            red_edge_array: Red-edge array (H, W)
            
        Returns:
            Dictionary of calculated spectral indices
        """
        indices = {}
        
        if rgb_array.ndim == 3:
            red = rgb_array[:, :, 0]
        else:
            red = rgb_array  # Assume single band input
        
        # Calculate NDVI if NIR is available
        if nir_array is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (nir_array - red) / (nir_array + red + 1e-8)
                indices["ndvi"] = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate NDRE if red-edge is available
        if red_edge_array is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                ndre = (red_edge_array - red) / (red_edge_array + red + 1e-8)
                indices["ndre"] = np.nan_to_num(ndre, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate Red-edge NDVI if both NIR and red-edge available
        if nir_array is not None and red_edge_array is not None:
            with np.errstate(divide='ignore', invalid='ignore'):
                red_edge_ndvi = (nir_array - red_edge_array) / (nir_array + red_edge_array + 1e-8)
                indices["red_edge_ndvi"] = np.nan_to_num(red_edge_ndvi, nan=0.0, posinf=0.0, neginf=0.0)
        
        return indices
    
    def get_kelp_probability_mask(self,
                                rgb_array: np.ndarray,
                                nir_array: np.ndarray | None = None,
                                red_edge_array: np.ndarray | None = None) -> np.ndarray:
        """
        Generate kelp probability mask using optimized SKEMA thresholds.
        
        Args:
            rgb_array: RGB array (H, W, 3)
            nir_array: Near-infrared array (H, W)
            red_edge_array: Red-edge array (H, W)
            
        Returns:
            Boolean mask where True indicates potential kelp
        """
        # Calculate spectral indices
        indices = self.calculate_spectral_indices(rgb_array, nir_array, red_edge_array)
        
        # Apply optimized thresholds from Task A2.7
        kelp_mask = np.zeros(rgb_array.shape[:2], dtype=bool)
        
        # NDVI threshold (if available)
        if "ndvi" in indices:
            kelp_mask |= indices["ndvi"] > self.config["kelp_detection"]["ndvi_threshold"]
        
        # NDRE threshold (if available) - primary indicator
        if "ndre" in indices:
            kelp_mask |= indices["ndre"] > self.config["kelp_detection"]["ndre_threshold"]
        
        return kelp_mask
    
    def get_optimized_thresholds(self) -> dict[str, float]:
        """
        Get the optimized SKEMA thresholds from Task A2.7 optimization work.
        
        Returns:
            Dictionary of optimized threshold values
        """
        return {
            "ndre_threshold": self.config["kelp_detection"]["ndre_threshold"],
            "ndvi_threshold": self.config["kelp_detection"]["ndvi_threshold"],
            "fai_threshold": self.config["kelp_detection"]["fai_threshold"]
        }
    
    def validate_input_bands(self, 
                           rgb_array: np.ndarray,
                           nir_array: np.ndarray | None = None,
                           red_edge_array: np.ndarray | None = None) -> dict[str, bool]:
        """
        Validate input spectral bands for processing.
        
        Args:
            rgb_array: RGB array
            nir_array: NIR array
            red_edge_array: Red-edge array
            
        Returns:
            Dictionary indicating which bands are available and valid
        """
        validation = {
            "rgb_valid": False,
            "nir_valid": False,
            "red_edge_valid": False,
            "spectral_guidance_possible": False
        }
        
        # Validate RGB
        if rgb_array is not None and rgb_array.ndim >= 2:
            validation["rgb_valid"] = True
        
        # Validate NIR
        if nir_array is not None and nir_array.shape == rgb_array.shape[:2]:
            validation["nir_valid"] = True
        
        # Validate Red-edge
        if red_edge_array is not None and red_edge_array.shape == rgb_array.shape[:2]:
            validation["red_edge_valid"] = True
        
        # Check if spectral guidance is possible
        validation["spectral_guidance_possible"] = (
            validation["rgb_valid"] and 
            (validation["nir_valid"] or validation["red_edge_valid"])
        )
        
        return validation 

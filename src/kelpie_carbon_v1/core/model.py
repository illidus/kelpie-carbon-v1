"""Machine learning model functions for kelp biomass prediction."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


class KelpBiomassModel:
    """Random Forest model for kelp biomass prediction."""

    def __init__(self, model_params: Optional[Dict] = None):
        """Initialize the biomass prediction model.

        Args:
            model_params: Parameters for Random Forest model
        """
        if model_params is None:
            model_params = {
                "n_estimators": 100,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            }

        self.model = RandomForestRegressor(**model_params)
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False

    def extract_features(self, dataset: xr.Dataset) -> pd.DataFrame:
        """Extract features from satellite dataset for biomass prediction.

        Args:
            dataset: xarray Dataset with satellite bands and masks

        Returns:
            DataFrame with extracted features
        """
        features = {}

        # Get valid kelp pixels (where kelp mask is True and data is valid)
        if "kelp_mask" in dataset and "valid_mask" in dataset:
            kelp_pixels = (dataset["kelp_mask"] == 1) & (dataset["valid_mask"] == 1)
        else:
            # Use all valid pixels if no kelp mask
            kelp_pixels = np.ones_like(dataset["red"].values, dtype=bool)
            if "valid_mask" in dataset:
                kelp_pixels = dataset["valid_mask"] == 1

        # Extract spectral bands
        bands = ["red", "red_edge", "nir", "swir1"]
        for band in bands:
            if band in dataset:
                band_data = dataset[band].values[kelp_pixels]
                # Handle empty arrays or all-NaN arrays to avoid RuntimeWarning
                if len(band_data) > 0 and not np.all(np.isnan(band_data)):
                    features[f"{band}_mean"] = [np.nanmean(band_data)]
                    features[f"{band}_std"] = [np.nanstd(band_data)]
                    features[f"{band}_median"] = [np.nanmedian(band_data)]
                    features[f"{band}_p75"] = [np.nanpercentile(band_data, 75)]
                    features[f"{band}_p25"] = [np.nanpercentile(band_data, 25)]
                else:
                    # Default values for empty or all-NaN arrays
                    features[f"{band}_mean"] = [np.float64(0.0)]
                    features[f"{band}_std"] = [np.float64(0.0)]
                    features[f"{band}_median"] = [np.float64(0.0)]
                    features[f"{band}_p75"] = [np.float64(0.0)]
                    features[f"{band}_p25"] = [np.float64(0.0)]

        # Calculate spectral indices
        if all(band in dataset for band in ["red", "red_edge", "nir", "swir1"]):
            indices = self._calculate_spectral_indices(dataset)

            for index_name, index_data in indices.items():
                valid_index = index_data[kelp_pixels]
                # Handle empty arrays or all-NaN arrays to avoid RuntimeWarning
                if len(valid_index) > 0 and not np.all(np.isnan(valid_index)):
                    features[f"{index_name}_mean"] = [np.nanmean(valid_index)]
                    features[f"{index_name}_std"] = [np.nanstd(valid_index)]
                    features[f"{index_name}_median"] = [np.nanmedian(valid_index)]
                else:
                    # Default values for empty or all-NaN arrays
                    features[f"{index_name}_mean"] = [np.float64(0.0)]
                    features[f"{index_name}_std"] = [np.float64(0.0)]
                    features[f"{index_name}_median"] = [np.float64(0.0)]

        # Environmental features
        if "water_mask" in dataset:
            water_coverage = (
                np.sum(dataset["water_mask"] == 1) / dataset["water_mask"].size
            )
            features["water_coverage"] = [np.float64(water_coverage)]

        if "kelp_mask" in dataset:
            kelp_coverage = (
                np.sum(dataset["kelp_mask"] == 1) / dataset["kelp_mask"].size
            )
            features["kelp_coverage"] = [np.float64(kelp_coverage)]

            # Kelp patch characteristics
            kelp_patches = self._analyze_kelp_patches(dataset["kelp_mask"].values)
            # Convert kelp patch values to proper types
            from typing import cast
            kelp_patches_typed = {k: [float(v[0])] if v else [0.0] for k, v in kelp_patches.items()}
            features.update(cast("Dict[str, List[Any]]", kelp_patches_typed))

        # Spatial features  
        spatial_features = self._calculate_spatial_features(dataset, kelp_pixels.values if hasattr(kelp_pixels, 'values') else kelp_pixels)
        # Convert spatial feature values to proper types
        spatial_features_typed = {k: [float(v[0])] if v else [0.0] for k, v in spatial_features.items()}
        features.update(cast("Dict[str, List[Any]]", spatial_features_typed))

        df = pd.DataFrame(features)
        self.feature_names = list(df.columns)

        return df

    def _calculate_spectral_indices(self, dataset: xr.Dataset) -> Dict[str, np.ndarray]:
        """Calculate spectral indices for biomass prediction."""
        red: np.ndarray = dataset["red"].values.astype(np.float32)
        red_edge: np.ndarray = dataset["red_edge"].values.astype(np.float32)
        nir: np.ndarray = dataset["nir"].values.astype(np.float32)
        swir1: np.ndarray = dataset["swir1"].values.astype(np.float32)

        indices = {}

        # NDVI (Normalized Difference Vegetation Index)
        with np.errstate(divide="ignore", invalid="ignore"):
            indices["ndvi"] = (nir - red) / (nir + red)

        # Red Edge NDVI
        with np.errstate(divide="ignore", invalid="ignore"):
            indices["red_edge_ndvi"] = (nir - red_edge) / (nir + red_edge)

        # NDRE (Normalized Difference Red Edge) - SKEMA Enhanced Formula
        # Updated to use proper NDRE formula: (Red_Edge - Red) / (Red_Edge + Red)
        # Uses optimal 740nm band if available for submerged kelp detection
        red_edge_optimal: np.ndarray
        if "red_edge_2" in dataset:
            red_edge_optimal = dataset["red_edge_2"].values.astype(np.float32)  # 740nm
        else:
            red_edge_optimal = red_edge  # fallback to 705nm

        with np.errstate(divide="ignore", invalid="ignore"):
            indices["ndre"] = (red_edge_optimal - red) / (red_edge_optimal + red)

        # FAI (Floating Algae Index)
        lambda_red, lambda_nir, lambda_swir1 = 665, 842, 1610
        with np.errstate(divide="ignore", invalid="ignore"):
            indices["fai"] = nir - (
                red
                + (swir1 - red)
                * (lambda_nir - lambda_red)
                / (lambda_swir1 - lambda_red)
            )

        # NDWI (Normalized Difference Water Index)
        with np.errstate(divide="ignore", invalid="ignore"):
            indices["ndwi"] = (red_edge - nir) / (red_edge + nir)

        # EVI (Enhanced Vegetation Index)
        with np.errstate(divide="ignore", invalid="ignore"):
            indices["evi"] = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * red_edge + 1)

        # Clean up NaN values
        for key in indices:
            indices[key] = np.nan_to_num(indices[key], nan=0.0)

        return indices

    def _analyze_kelp_patches(self, kelp_mask: np.ndarray) -> Dict[str, List[float]]:
        """Analyze kelp patch characteristics."""
        from scipy import ndimage
        from typing import Tuple, cast

        # Label connected components
        labeled_result = ndimage.label(kelp_mask)
        labeled_array, num_patches = cast(Tuple[np.ndarray, int], labeled_result)

        if num_patches == 0:
            return {
                "num_kelp_patches": [0],
                "avg_patch_size": [0],
                "largest_patch_size": [0],
                "patch_density": [0],
            }

        # Calculate patch sizes
        patch_sizes = np.bincount(labeled_array.ravel())[1:]  # Exclude background

        return {
            "num_kelp_patches": [float(num_patches)],
            "avg_patch_size": [float(np.mean(patch_sizes))],
            "largest_patch_size": [float(np.max(patch_sizes))],
            "patch_density": [float(num_patches / kelp_mask.size)],
        }

    def _calculate_spatial_features(
        self, dataset: xr.Dataset, kelp_pixels: Union[np.ndarray, Any]
    ) -> Dict[str, List[float]]:
        """Calculate spatial features from the dataset."""
        features = {}

        # Image dimensions
        height, width = dataset.sizes["y"], dataset.sizes["x"]
        features["image_height"] = [float(height)]
        features["image_width"] = [float(width)]
        features["total_pixels"] = [float(height * width)]

        # Coordinate-based features
        if "x" in dataset.coords and "y" in dataset.coords:
            x_coords = dataset.coords["x"].values
            y_coords = dataset.coords["y"].values

            # Center coordinates
            features["center_lon"] = [float(np.mean(x_coords))]
            features["center_lat"] = [float(np.mean(y_coords))]

            # Extent
            features["lon_range"] = [float(np.ptp(x_coords))]
            features["lat_range"] = [float(np.ptp(y_coords))]

        return features

    def train(self, training_data: List[Tuple[xr.Dataset, float]]) -> Dict[str, float]:
        """Train the Random Forest model on training data.

        Args:
            training_data: List of (dataset, biomass) tuples

        Returns:
            Dictionary with training metrics
        """
        if len(training_data) == 0:
            raise ValueError("No training data provided")

        # Extract features and targets
        feature_list = []
        targets = []

        for dataset, biomass in training_data:
            features = self.extract_features(dataset)
            feature_list.append(features)
            targets.append(biomass)

        # Combine all features
        X = pd.concat(feature_list, ignore_index=True)
        y = np.array(targets)

        # Handle missing values
        X = X.fillna(0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        # Calculate metrics
        metrics = {
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "n_samples": len(training_data),
            "n_features": X.shape[1],
        }

        # Cross-validation score (adjust folds based on sample size)
        cv_folds = min(5, len(training_data))  # Use min of 5 or number of samples
        if cv_folds >= 2:  # Need at least 2 folds for CV
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, cv=cv_folds, scoring="r2"
            )
            metrics["cv_r2_mean"] = np.mean(cv_scores)
            metrics["cv_r2_std"] = np.std(cv_scores)
        else:
            # Skip CV for very small datasets
            metrics["cv_r2_mean"] = metrics["train_r2"]
            metrics["cv_r2_std"] = 0.0

        self.is_trained = True
        return metrics

    def predict(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Predict biomass for a given dataset.

        Args:
            dataset: xarray Dataset with satellite data

        Returns:
            Dictionary with biomass prediction and confidence metrics
        """
        if not self.is_trained:
            # Use default pre-trained model if available
            model_path = self._get_default_model_path()
            if model_path.exists():
                self.load_model(str(model_path))
            else:
                # Train on synthetic data for demonstration
                return self._predict_with_synthetic_model(dataset)

        # Extract features
        features = self.extract_features(dataset)
        features = features.fillna(0)

        # Ensure feature consistency
        if self.feature_names and len(self.feature_names) > 0:
            missing_features = set(self.feature_names) - set(features.columns)
            for feature in missing_features:
                features[feature] = 0
            features = features[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(features)

        # Make prediction
        biomass_prediction = self.model.predict(X_scaled)[0]

        # Calculate prediction confidence using tree predictions
        if hasattr(self.model, 'estimators_') and self.model.estimators_ is not None:
            tree_predictions = np.array(
                [tree.predict(X_scaled)[0] for tree in self.model.estimators_]
            )
        else:
            tree_predictions = np.array([biomass_prediction])
        confidence = 1.0 - (
            np.std(tree_predictions) / (np.mean(tree_predictions) + 1e-8)
        )

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            importance_dict = dict(
                zip(self.feature_names, self.model.feature_importances_)
            )
            top_features = sorted(
                importance_dict.items(), key=lambda x: x[1], reverse=True
            )[:5]
        else:
            top_features = []

        return {
            "biomass_kg_per_hectare": max(0, biomass_prediction),  # Ensure non-negative
            "prediction_confidence": np.clip(confidence, 0, 1),
            "top_features": [
                f"{name}: {importance:.3f}" for name, importance in top_features
            ],
            "model_type": "Random Forest",
            "feature_count": len(self.feature_names),
        }

    def _predict_with_synthetic_model(self, dataset: xr.Dataset) -> Dict[str, Any]:
        """Fallback prediction using a synthetic model based on spectral indices."""
        # Extract basic features
        features = self.extract_features(dataset)

        # Simple biomass estimation based on kelp coverage and spectral indices
        from typing import cast, Any
        kelp_coverage_raw = features.get("kelp_coverage", [0])[0]
        if hasattr(kelp_coverage_raw, "values"):
            kelp_coverage = float(cast(Any, kelp_coverage_raw.values))
        else:
            kelp_coverage = float(cast(Any, kelp_coverage_raw))

        # Base biomass estimation (kg/hectare)
        base_biomass = float(kelp_coverage) * 5000  # 5000 kg/hectare for full coverage

        # Adjust based on spectral indices if available
        if "fai_mean" in features.columns:
            from typing import cast
            fai_val_raw = features["fai_mean"].iloc[0]
            try:
                if hasattr(fai_val_raw, "values"):
                    # Extract scalar value from numpy array or pandas extension
                    val_extracted = fai_val_raw.values
                    if hasattr(val_extracted, 'item'):
                        fai_val = float(val_extracted.item())
                    else:
                        fai_val = float(cast(Any, val_extracted))
                else:
                    # Handle scalar values
                    fai_val = float(cast(Any, fai_val_raw)) if not pd.isna(fai_val_raw) else 0.0
            except (TypeError, ValueError, AttributeError):
                fai_val = 0.0
            fai_factor = max(0, fai_val + 0.01) * 2000
            base_biomass += fai_factor

        if "red_edge_ndvi_mean" in features.columns:
            ndvi_val = features["red_edge_ndvi_mean"].iloc[0]
            if hasattr(ndvi_val, "values"):
                ndvi_val = float(ndvi_val.values)
            ndvi_factor = max(0, float(ndvi_val)) * 3000
            base_biomass += ndvi_factor

        # Add some environmental factors
        if "water_coverage" in features.columns:
            water_val = features["water_coverage"].iloc[0]
            if hasattr(water_val, "values"):
                water_val = float(water_val.values)
            water_factor = float(water_val) * 1000
            base_biomass += water_factor

        # Ensure realistic range (0-15000 kg/hectare)
        biomass = np.clip(base_biomass, 0, 15000)

        return {
            "biomass_kg_per_hectare": biomass,
            "prediction_confidence": 0.7,  # Moderate confidence for synthetic model
            "top_features": [
                "kelp_coverage: 0.400",
                "fai_mean: 0.300",
                "water_coverage: 0.200",
            ],
            "model_type": "Synthetic (Spectral Index Based)",
            "feature_count": len(features.columns),
        }

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.is_trained = model_data["is_trained"]

    def _get_default_model_path(self) -> Path:
        """Get the path to the default pre-trained model."""
        return Path(__file__).parent / "models" / "kelp_biomass_rf.joblib"


def predict_biomass(
    dataset: xr.Dataset, model_path: Optional[str] = None
) -> Dict[str, Any]:
    """Main function to predict biomass from satellite dataset.

    Args:
        dataset: xarray Dataset with satellite bands and masks
        model_path: Optional path to pre-trained model

    Returns:
        Dictionary with biomass prediction and metadata
    """
    model = KelpBiomassModel()

    if model_path and os.path.exists(model_path):
        model.load_model(model_path)

    return model.predict(dataset)


def generate_training_data(n_samples: int = 100) -> List[Tuple[xr.Dataset, float]]:
    """Generate synthetic training data for model development.

    Args:
        n_samples: Number of training samples to generate

    Returns:
        List of (dataset, biomass) tuples
    """
    training_data = []
    np.random.seed(42)

    for i in range(n_samples):
        # Generate synthetic satellite data
        height, width = 50, 50

        # Create realistic spectral values
        red = np.random.normal(0.1, 0.03, (height, width)).clip(0, 1)
        red_edge = np.random.normal(0.15, 0.04, (height, width)).clip(0, 1)
        nir = np.random.normal(0.3, 0.1, (height, width)).clip(0, 1)
        swir1 = np.random.normal(0.2, 0.05, (height, width)).clip(0, 1)

        # Generate correlated kelp and water masks
        kelp_coverage = np.random.beta(2, 5)  # Skewed towards lower coverage
        water_coverage = np.random.uniform(0.7, 1.0)  # High water coverage

        kelp_mask = np.random.choice(
            [0, 1], size=(height, width), p=[1 - kelp_coverage, kelp_coverage]
        )
        water_mask = np.random.choice(
            [0, 1], size=(height, width), p=[1 - water_coverage, water_coverage]
        )
        valid_mask = np.ones((height, width))

        # Create coordinate arrays for consistency
        lons = np.linspace(-123.5, -123.0, width)
        lats = np.linspace(49.5, 49.0, height)

        # Create dataset
        dataset = xr.Dataset(
            {
                "red": (["y", "x"], red),
                "red_edge": (["y", "x"], red_edge),
                "nir": (["y", "x"], nir),
                "swir1": (["y", "x"], swir1),
                "kelp_mask": (["y", "x"], kelp_mask),
                "water_mask": (["y", "x"], water_mask),
                "valid_mask": (["y", "x"], valid_mask),
            },
            coords={"x": lons, "y": lats},
        )

        # Generate realistic biomass based on spectral characteristics
        # Higher biomass for areas with higher kelp coverage and good spectral indices
        base_biomass = kelp_coverage * 4000  # Base biomass from coverage

        # Add noise and environmental factors
        biomass = base_biomass + np.random.normal(0, 500)
        biomass = max(0, biomass)  # Ensure non-negative

        training_data.append((dataset, biomass))

    return training_data

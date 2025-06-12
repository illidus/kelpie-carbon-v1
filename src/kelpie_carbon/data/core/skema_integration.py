"""
SKEMA Data Integration Module

This module provides integration with the SKEMA (Satellite-based Kelp Mapping) project
from the University of Victoria's SPECTRAL Remote Sensing Laboratory.

SKEMA focuses on developing satellite-based tools for kelp forest monitoring using
Sentinel-2 imagery and deep learning methods, primarily for British Columbia waters.
"""

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import requests

logger = logging.getLogger(__name__)


@dataclass
class SKEMAValidationPoint:
    """A validation point from SKEMA research."""

    lat: float
    lng: float
    kelp_present: bool
    species: str | None
    confidence: float
    biomass_estimate: float | None
    observation_date: date | None
    source: str = "skema_uvic"


class SKEMADataIntegrator:
    """Integrates with University of Victoria's SKEMA kelp mapping data."""

    def __init__(self):
        self.base_url = (
            "https://spectral.geog.uvic.ca/skema"  # Hypothetical API endpoint
        )
        self.session = requests.Session()

    def fetch_validation_points(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        confidence_threshold: float = 0.8,
    ) -> list[SKEMAValidationPoint]:
        """
        Fetch validation points from SKEMA dataset.

        Args:
            bbox: Bounding box (min_lat, min_lng, max_lat, max_lng) for spatial filtering
            confidence_threshold: Minimum confidence score for validation points

        Returns:
            List of SKEMA validation points
        """
        try:
            # For now, return simulated SKEMA-style data
            # In a real implementation, this would query the SKEMA API
            return self._get_simulated_skema_data(bbox, confidence_threshold)

        except Exception as e:
            logger.error(f"Failed to fetch SKEMA validation points: {e}")
            return []

    def _get_simulated_skema_data(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        confidence_threshold: float = 0.8,
    ) -> list[SKEMAValidationPoint]:
        """Generate simulated SKEMA-style validation data for testing."""

        # Simulated validation points from BC coast (Vancouver Island area)
        simulated_data = [
            # Kelp forest locations with high confidence
            SKEMAValidationPoint(
                49.7481,
                -125.1342,
                True,
                "Macrocystis pyrifera",
                0.95,
                1250.0,
                date(2023, 8, 15),
            ),
            SKEMAValidationPoint(
                49.7512,
                -125.1289,
                True,
                "Nereocystis luetkeana",
                0.91,
                850.0,
                date(2023, 8, 15),
            ),
            SKEMAValidationPoint(
                49.7523,
                -125.1256,
                True,
                "Macrocystis pyrifera",
                0.88,
                920.0,
                date(2023, 8, 16),
            ),
            SKEMAValidationPoint(
                49.7456,
                -125.1398,
                True,
                "Saccharina sessilis",
                0.86,
                650.0,
                date(2023, 8, 16),
            ),
            SKEMAValidationPoint(
                49.7489,
                -125.1323,
                True,
                "Macrocystis pyrifera",
                0.93,
                1100.0,
                date(2023, 8, 17),
            ),
            # Non-kelp locations with high confidence
            SKEMAValidationPoint(
                49.7445, -125.1378, False, None, 0.92, 0.0, date(2023, 8, 15)
            ),
            SKEMAValidationPoint(
                49.7467, -125.1334, False, None, 0.89, 0.0, date(2023, 8, 16)
            ),
            SKEMAValidationPoint(
                49.7434, -125.1356, False, None, 0.94, 0.0, date(2023, 8, 17)
            ),
            # Lower confidence points
            SKEMAValidationPoint(
                49.7502,
                -125.1278,
                True,
                "Macrocystis pyrifera",
                0.78,
                450.0,
                date(2023, 8, 17),
            ),
            SKEMAValidationPoint(
                49.7478, -125.1367, False, None, 0.82, 0.0, date(2023, 8, 18)
            ),
        ]

        # Apply confidence threshold filter
        filtered_data = [
            point
            for point in simulated_data
            if point.confidence >= confidence_threshold
        ]

        # Apply spatial filter if provided
        if bbox:
            min_lat, min_lng, max_lat, max_lng = bbox
            filtered_data = [
                point
                for point in filtered_data
                if min_lat <= point.lat <= max_lat and min_lng <= point.lng <= max_lng
            ]

        logger.info(f"Generated {len(filtered_data)} simulated SKEMA validation points")
        return filtered_data

    def get_species_info(self) -> dict[str, dict]:
        """Get information about kelp species tracked by SKEMA."""
        return {
            "Macrocystis pyrifera": {
                "common_name": "Giant Kelp",
                "typical_biomass_range": (800, 1500),  # kg/ha
                "preferred_depth": (5, 30),  # meters
                "growth_season": "spring_summer",
            },
            "Nereocystis luetkeana": {
                "common_name": "Bull Kelp",
                "typical_biomass_range": (600, 1200),  # kg/ha
                "preferred_depth": (3, 20),  # meters
                "growth_season": "spring_summer",
            },
            "Saccharina sessilis": {
                "common_name": "Sugar Kelp",
                "typical_biomass_range": (400, 800),  # kg/ha
                "preferred_depth": (1, 15),  # meters
                "growth_season": "winter_spring",
            },
        }

    def validate_model_predictions(
        self, predictions: list[dict], validation_points: list[SKEMAValidationPoint]
    ) -> dict:
        """
        Validate model predictions against SKEMA ground truth data.

        Args:
            predictions: List of model predictions with 'lat', 'lng', 'kelp_present', 'biomass'
            validation_points: SKEMA validation points

        Returns:
            Validation metrics dictionary
        """
        if not predictions or not validation_points:
            return {"error": "Insufficient data for validation"}

        matches = []
        tolerance = 0.001  # ~100m tolerance for lat/lng matching

        for pred in predictions:
            # Find nearest validation point
            nearest_point = None
            min_distance = float("inf")

            for val_point in validation_points:
                distance = np.sqrt(
                    (pred["lat"] - val_point.lat) ** 2
                    + (pred["lng"] - val_point.lng) ** 2
                )
                if distance < min_distance and distance <= tolerance:
                    min_distance = distance
                    nearest_point = val_point

            if nearest_point:
                matches.append(
                    {
                        "predicted_kelp": pred.get("kelp_present", False),
                        "actual_kelp": nearest_point.kelp_present,
                        "predicted_biomass": pred.get("biomass", 0.0),
                        "actual_biomass": nearest_point.biomass_estimate or 0.0,
                        "confidence": nearest_point.confidence,
                    }
                )

        if not matches:
            return {"error": "No matching validation points found"}

        # Calculate metrics
        correct_kelp_predictions = sum(
            1 for m in matches if m["predicted_kelp"] == m["actual_kelp"]
        )

        kelp_accuracy = correct_kelp_predictions / len(matches)

        # Biomass RMSE for kelp-present locations
        kelp_matches = [m for m in matches if m["actual_kelp"]]
        biomass_rmse = 0.0
        if kelp_matches:
            biomass_errors = [
                (m["predicted_biomass"] - m["actual_biomass"]) ** 2
                for m in kelp_matches
            ]
            biomass_rmse = np.sqrt(np.mean(biomass_errors))

        return {
            "total_matches": len(matches),
            "kelp_presence_accuracy": kelp_accuracy,
            "biomass_rmse": biomass_rmse,
            "high_confidence_matches": len(
                [m for m in matches if m["confidence"] > 0.9]
            ),
            "validation_source": "skema_uvic",
        }


# Convenience function for easy integration
def get_skema_validation_data(
    bbox: tuple[float, float, float, float] | None = None,
    confidence_threshold: float = 0.8,
) -> list[SKEMAValidationPoint]:
    """
    Convenience function to get SKEMA validation data.

    Args:
        bbox: Bounding box for spatial filtering
        confidence_threshold: Minimum confidence score

    Returns:
        List of SKEMA validation points
    """
    integrator = SKEMADataIntegrator()
    return integrator.fetch_validation_points(bbox, confidence_threshold)

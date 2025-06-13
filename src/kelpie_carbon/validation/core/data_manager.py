"""ValidationDataManager - Task 2.1/2.2
Comprehensive data management for field validation campaigns.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationCampaign:
    """Represents a field validation campaign."""

    campaign_id: str
    site_name: str
    date_start: datetime
    date_end: datetime
    satellite_overpass_time: datetime
    weather_conditions: dict[str, Any]
    personnel: list[str]
    equipment_used: list[str]
    coordinates: tuple[float, float]  # (lat, lng)


@dataclass
class GroundTruthMeasurement:
    """Individual ground truth measurement point."""

    measurement_id: str
    campaign_id: str
    lat: float
    lng: float
    depth_m: float
    kelp_present: bool
    kelp_species: str | None
    kelp_density: str  # 'none', 'sparse', 'moderate', 'dense'
    canopy_type: str  # 'surface', 'submerged', 'mixed'
    timestamp: datetime
    spectral_data: dict[str, float] | None = None


class ValidationDataManager:
    """Manages validation data storage and retrieval."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize the validation data manager."""
        self.data_dir = data_dir or Path("validation_data")
        self.data_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.data_dir / "field_campaigns").mkdir(exist_ok=True)
        (self.data_dir / "satellite_data").mkdir(exist_ok=True)
        (self.data_dir / "validation_results").mkdir(exist_ok=True)

        # Initialize database
        self.db_path = self.data_dir / "validation.db"
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with validation schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Validation campaigns table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS validation_campaigns (
                    campaign_id VARCHAR(50) PRIMARY KEY,
                    site_name VARCHAR(100),
                    date_start DATE,
                    date_end DATE,
                    satellite_overpass_time TIMESTAMP,
                    weather_conditions JSON,
                    personnel TEXT,
                    equipment_used TEXT,
                    lat DECIMAL(10,7),
                    lng DECIMAL(10,7)
                )
            """
            )

            # Ground truth measurements table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ground_truth_kelp (
                    measurement_id VARCHAR(50) PRIMARY KEY,
                    campaign_id VARCHAR(50),
                    lat DECIMAL(10,7),
                    lng DECIMAL(10,7),
                    depth_m DECIMAL(5,2),
                    kelp_present BOOLEAN,
                    kelp_species VARCHAR(100),
                    kelp_density VARCHAR(20),
                    canopy_type VARCHAR(20),
                    timestamp TIMESTAMP,
                    spectral_data JSON,
                    FOREIGN KEY (campaign_id) REFERENCES validation_campaigns(campaign_id)
                )
            """
            )

            conn.commit()

    def create_campaign(self, campaign: ValidationCampaign) -> str:
        """Create a new validation campaign."""
        campaign_dir = self.data_dir / "field_campaigns" / campaign.campaign_id
        campaign_dir.mkdir(exist_ok=True)

        # Create subdirectories for campaign data
        for subdir in ["gps_data", "spectral_data", "environmental", "metadata"]:
            (campaign_dir / subdir).mkdir(exist_ok=True)

        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO validation_campaigns
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    campaign.campaign_id,
                    campaign.site_name,
                    campaign.date_start.isoformat(),
                    campaign.date_end.isoformat(),
                    campaign.satellite_overpass_time.isoformat(),
                    json.dumps(campaign.weather_conditions),
                    json.dumps(campaign.personnel),
                    json.dumps(campaign.equipment_used),
                    campaign.coordinates[0],
                    campaign.coordinates[1],
                ),
            )

        # Save campaign metadata
        metadata_file = campaign_dir / "metadata" / "campaign_summary.json"
        with open(metadata_file, "w") as f:
            json.dump(asdict(campaign), f, indent=2, default=str)

        logger.info(f"Created validation campaign: {campaign.campaign_id}")
        return campaign.campaign_id

    def add_ground_truth(self, measurement: GroundTruthMeasurement) -> str:
        """Add ground truth measurement to campaign."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ground_truth_kelp
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    measurement.measurement_id,
                    measurement.campaign_id,
                    measurement.lat,
                    measurement.lng,
                    measurement.depth_m,
                    measurement.kelp_present,
                    measurement.kelp_species,
                    measurement.kelp_density,
                    measurement.canopy_type,
                    measurement.timestamp.isoformat(),
                    (
                        json.dumps(measurement.spectral_data)
                        if measurement.spectral_data
                        else None
                    ),
                ),
            )

        logger.info(f"Added ground truth measurement: {measurement.measurement_id}")
        return measurement.measurement_id

    def get_campaign(self, campaign_id: str) -> ValidationCampaign | None:
        """Retrieve campaign by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM validation_campaigns WHERE campaign_id = ?
            """,
                (campaign_id,),
            )

            row = cursor.fetchone()
            if row:
                return ValidationCampaign(
                    campaign_id=row[0],
                    site_name=row[1],
                    date_start=datetime.fromisoformat(row[2]),
                    date_end=datetime.fromisoformat(row[3]),
                    satellite_overpass_time=datetime.fromisoformat(row[4]),
                    weather_conditions=json.loads(row[5]),
                    personnel=json.loads(row[6]),
                    equipment_used=json.loads(row[7]),
                    coordinates=(row[8], row[9]),
                )
        return None

    def get_ground_truth_for_campaign(
        self, campaign_id: str
    ) -> list[GroundTruthMeasurement]:
        """Get all ground truth measurements for a campaign."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT * FROM ground_truth_kelp WHERE campaign_id = ?
            """,
                (campaign_id,),
            )

            measurements = []
            for row in cursor.fetchall():
                spectral_data = json.loads(row[10]) if row[10] else None
                measurements.append(
                    GroundTruthMeasurement(
                        measurement_id=row[0],
                        campaign_id=row[1],
                        lat=row[2],
                        lng=row[3],
                        depth_m=row[4],
                        kelp_present=bool(row[5]),
                        kelp_species=row[6],
                        kelp_density=row[7],
                        canopy_type=row[8],
                        timestamp=datetime.fromisoformat(row[9]),
                        spectral_data=spectral_data,
                    )
                )

        return measurements

    def export_validation_dataset(self, campaign_id: str) -> dict[str, pd.DataFrame]:
        """Export campaign data as structured DataFrames for analysis."""
        campaign = self.get_campaign(campaign_id)
        ground_truth = self.get_ground_truth_for_campaign(campaign_id)

        # Convert to DataFrames
        if campaign is None:
            raise ValueError(f"Campaign {campaign_id} not found")
        campaign_df = pd.DataFrame([asdict(campaign)])

        ground_truth_data = []
        for measurement in ground_truth:
            data = asdict(measurement)
            data["timestamp"] = measurement.timestamp.isoformat()
            ground_truth_data.append(data)

        ground_truth_df = pd.DataFrame(ground_truth_data)

        return {"campaign": campaign_df, "ground_truth": ground_truth_df}

    def validate_data_quality(self, campaign_id: str) -> dict[str, Any]:
        """Validate data quality for a campaign."""
        ground_truth = self.get_ground_truth_for_campaign(campaign_id)

        quality_report = {
            "campaign_id": campaign_id,
            "total_measurements": len(ground_truth),
            "spatial_coverage": {
                "lat_range": (
                    min(m.lat for m in ground_truth),
                    max(m.lat for m in ground_truth),
                ),
                "lng_range": (
                    min(m.lng for m in ground_truth),
                    max(m.lng for m in ground_truth),
                ),
            },
            "temporal_coverage": {
                "start": min(m.timestamp for m in ground_truth).isoformat(),
                "end": max(m.timestamp for m in ground_truth).isoformat(),
            },
            "kelp_statistics": {
                "kelp_present_count": sum(1 for m in ground_truth if m.kelp_present),
                "kelp_absent_count": sum(1 for m in ground_truth if not m.kelp_present),
            },
        }

        return quality_report

    def close(self):
        """Close any open resources."""
        # ValidationDataManager uses context managers, so no explicit cleanup needed
        pass

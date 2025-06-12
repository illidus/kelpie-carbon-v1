"""
MockValidationGenerator - Task 2.3
Generates realistic mock validation data for BC kelp forests.
"""

import random
from datetime import datetime, timedelta

from .data_manager import (
    GroundTruthMeasurement,
    ValidationCampaign,
    ValidationDataManager,
)


class MockValidationGenerator:
    """Generates realistic mock validation data for BC coastal waters."""

    def __init__(self, data_manager: ValidationDataManager):
        self.data_manager = data_manager

        # BC validation sites accessible from Victoria
        self.sites = {
            "saanich_inlet": {
                "name": "Saanich Inlet Kelp Forest",
                "lat": 48.583,
                "lng": -123.500,
                "kelp_coverage": 0.25,
                "species": "nereocystis_luetkeana",
            },
            "haro_strait": {
                "name": "Haro Strait (Gulf Islands)",
                "lat": 48.500,
                "lng": -123.167,
                "kelp_coverage": 0.35,
                "species": "nereocystis_luetkeana",
            },
        }

        # BC bull kelp spectral signatures
        self.kelp_spectra = {
            "665": 0.02,  # Red
            "705": 0.08,  # Red Edge 1
            "740": 0.12,  # Red Edge 2 (optimal)
            "783": 0.15,  # Red Edge 3
            "842": 0.25,  # NIR
        }

    def create_bc_validation_dataset(self, site_key: str = "saanich_inlet") -> str:
        """Create complete BC validation dataset."""
        site = self.sites[site_key]
        campaign_date = datetime(2023, 8, 15)

        # Create campaign
        from typing import cast
        campaign = ValidationCampaign(
            campaign_id=f"bc_{site_key}_20230815",
            site_name=str(site["name"]),
            date_start=campaign_date,
            date_end=campaign_date + timedelta(days=1),
            satellite_overpass_time=campaign_date.replace(hour=19, minute=20),
            weather_conditions={"wind_ms": 3.5, "clouds": 15, "temp_c": 16},
            personnel=["Marine biologist", "Remote sensing tech"],
            equipment_used=["GPS", "Hyperspectral radiometer"],
            coordinates=(cast(float, site["lat"]), cast(float, site["lng"])),
        )

        campaign_id = self.data_manager.create_campaign(campaign)

        # Generate grid measurements
        measurements = []
        for i in range(50):  # 50 measurement points
            lat = cast(float, site["lat"]) + random.uniform(-0.01, 0.01)
            lng = cast(float, site["lng"]) + random.uniform(-0.01, 0.01)

            kelp_present = random.random() < cast(float, site["kelp_coverage"])
            depth = random.uniform(5, 25)

            measurement = GroundTruthMeasurement(
                measurement_id=f"{campaign_id}_m{i:03d}",
                campaign_id=campaign_id,
                lat=lat,
                lng=lng,
                depth_m=depth,
                kelp_present=kelp_present,
                kelp_species=str(site["species"]) if kelp_present else None,
                kelp_density="moderate" if kelp_present else "none",
                canopy_type="surface" if depth < 10 else "submerged",
                timestamp=campaign.satellite_overpass_time
                + timedelta(minutes=random.randint(-60, 60)),
                spectral_data=self.kelp_spectra if kelp_present else None,
            )

            measurements.append(measurement)
            self.data_manager.add_ground_truth(measurement)

        print(f"Created BC validation dataset: {campaign_id}")
        print(f"Location: {site['name']}")
        print(f"Measurements: {len(measurements)}")
        print(f"Kelp present: {sum(1 for m in measurements if m.kelp_present)}")

        return campaign_id

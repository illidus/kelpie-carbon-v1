"""
FieldCampaignProtocols - Task 2.1
Protocols for field validation campaigns in BC waters.
"""

from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class FieldProtocol:
    """Field measurement protocol specifications."""

    protocol_name: str
    equipment_required: List[str]
    measurement_steps: List[str]
    timing_requirements: str


class FieldCampaignProtocols:
    """Standard protocols for BC kelp validation campaigns."""

    def get_bc_protocols(self) -> Dict[str, FieldProtocol]:
        """Get BC-specific field protocols."""

        return {
            "gps_mapping": FieldProtocol(
                protocol_name="GPS Kelp Boundary Mapping",
                equipment_required=[
                    "GPS unit (±1m accuracy)",
                    "Waterproof tablet",
                    "Boat access",
                    "Polarized sunglasses",
                ],
                measurement_steps=[
                    "Navigate to kelp forest edge",
                    "Record GPS waypoints every 50m",
                    "Note kelp density and species",
                    "Document water depth",
                    "Photograph representative areas",
                ],
                timing_requirements="±2 hours of satellite overpass",
            ),
            "spectral_measurements": FieldProtocol(
                protocol_name="Hyperspectral Measurements",
                equipment_required=[
                    "Hyperspectral radiometer (350-2500nm)",
                    "White reference panel",
                    "Dive equipment",
                    "Data logger",
                ],
                measurement_steps=[
                    "Calibrate with white reference",
                    "Measure over kelp canopy",
                    "Measure over open water",
                    "Submerged measurements at multiple depths",
                    "Record environmental conditions",
                ],
                timing_requirements="±1 hour of satellite overpass",
            ),
            "environmental": FieldProtocol(
                protocol_name="Environmental Monitoring",
                equipment_required=[
                    "Water quality sonde",
                    "Secchi disk",
                    "Current meter",
                    "Weather station",
                ],
                measurement_steps=[
                    "Record tide height and timing",
                    "Measure water temperature/salinity",
                    "Assess water clarity",
                    "Record current and wind conditions",
                    "Document weather conditions",
                ],
                timing_requirements="Continuous during campaign",
            ),
        }

    def get_site_recommendations(self, site_name: str) -> Dict[str, str]:
        """Get BC site-specific recommendations."""

        recommendations = {
            "saanich_inlet": {
                "access": "Brentwood Bay Marina",
                "conditions": "Slack tide, morning calm",
                "season": "Peak June-September",
                "safety": "Ferry traffic, strong currents",
            },
            "haro_strait": {
                "access": "Sidney ferry to Salt Spring",
                "conditions": "Low current slack periods",
                "season": "Peak July-October",
                "safety": "High currents, whale zones",
            },
        }

        return recommendations.get(site_name, {})

    def calculate_timing(self, overpass_time: datetime) -> Dict[str, datetime]:
        """Calculate optimal measurement timing."""

        return {
            "start": overpass_time - timedelta(hours=2),
            "spectral": overpass_time - timedelta(minutes=30),
            "overpass": overpass_time,
            "end": overpass_time + timedelta(hours=2),
        }

"""
Unit tests for real data acquisition functionality.

Tests the real SKEMA CSV data loader and validation without external downloads.
"""

import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from kelpie_carbon.data.real_data_acquisition import RealDataAcquisition, SatelliteScene


class TestRealDataAcquisition(unittest.TestCase):
    """Test real data acquisition functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_acquisition = RealDataAcquisition(self.temp_dir.name)

        # Create test sample data directory inside the temp directory
        self.sample_data_dir = Path(self.temp_dir.name) / "sample_data"
        self.sample_data_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def _create_test_csv(
        self, filename: str, data: dict, extra_columns: dict | None = None
    ):
        """Create test CSV files helper."""
        df_data = data.copy()
        if extra_columns:
            df_data.update(extra_columns)
        df = pd.DataFrame(df_data)
        csv_path = self.sample_data_dir / filename
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_load_skema_csv_data_valid(self):
        """Test loading valid SKEMA CSV data."""
        # Create valid test CSV
        test_data = {
            "lat": [50.0833, 50.0845, 50.0820],
            "lon": [-126.1667, -126.1655, -126.1680],
            "dry_weight_kg_m2": [2.45, 3.12, 1.87],
        }
        self._create_test_csv("test_site_skema.csv", test_data)

        # Test loading
        df = self.data_acquisition.load_skema_csv_data("test_site")

        # Verify results
        self.assertEqual(len(df), 3)
        self.assertListEqual(list(df.columns), ["lat", "lon", "dry_weight_kg_m2"])
        self.assertAlmostEqual(df["lat"].iloc[0], 50.0833)
        self.assertAlmostEqual(df["lon"].iloc[0], -126.1667)
        self.assertAlmostEqual(df["dry_weight_kg_m2"].iloc[0], 2.45)

    def test_load_skema_csv_data_with_extra_columns(self):
        """Test loading SKEMA CSV with extra columns (should be filtered)."""
        test_data = {
            "lat": [36.8000, 36.8015],
            "lon": [-121.9000, -121.8985],
            "dry_weight_kg_m2": [6.78, 7.23],
            "extra_column": ["A", "B"],
            "another_extra": [1, 2],
        }
        self._create_test_csv("test_site2_skema.csv", test_data)

        df = self.data_acquisition.load_skema_csv_data("test_site2")

        # Should only return required columns
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ["lat", "lon", "dry_weight_kg_m2"])

    def test_load_skema_csv_data_file_not_found(self):
        """Test FileNotFoundError when CSV file doesn't exist."""
        with self.assertRaises(FileNotFoundError) as context:
            self.data_acquisition.load_skema_csv_data("nonexistent_site")

        self.assertIn("SKEMA CSV file not found", str(context.exception))

    def test_load_skema_csv_data_missing_required_columns(self):
        """Test ValueError when required columns are missing."""
        # Missing 'dry_weight_kg_m2' column
        test_data = {"lat": [50.0833, 50.0845], "lon": [-126.1667, -126.1655]}
        self._create_test_csv("missing_column_skema.csv", test_data)

        with self.assertRaises(ValueError) as context:
            self.data_acquisition.load_skema_csv_data("missing_column")

        self.assertIn("Missing required columns", str(context.exception))
        self.assertIn("dry_weight_kg_m2", str(context.exception))

    def test_load_skema_csv_data_invalid_data_types(self):
        """Test ValueError when data types are invalid."""
        # Non-numeric latitude
        test_data = {
            "lat": ["invalid", "50.0845"],
            "lon": [-126.1667, -126.1655],
            "dry_weight_kg_m2": [2.45, 3.12],
        }
        self._create_test_csv("invalid_types_skema.csv", test_data)

        with self.assertRaises(ValueError) as context:
            self.data_acquisition.load_skema_csv_data("invalid_types")

        self.assertIn("must be numeric", str(context.exception))

    def test_load_skema_csv_data_invalid_coordinate_ranges(self):
        """Test ValueError when coordinates are out of valid ranges."""
        # Invalid latitude (> 90)
        test_data = {
            "lat": [95.0, 50.0845],
            "lon": [-126.1667, -126.1655],
            "dry_weight_kg_m2": [2.45, 3.12],
        }
        self._create_test_csv("invalid_coords_skema.csv", test_data)

        with self.assertRaises(ValueError) as context:
            self.data_acquisition.load_skema_csv_data("invalid_coords")

        self.assertIn(
            "Latitude values must be between -90 and 90", str(context.exception)
        )

    def test_load_skema_csv_data_negative_dry_weight(self):
        """Test ValueError when dry weight values are negative."""
        test_data = {
            "lat": [50.0833, 50.0845],
            "lon": [-126.1667, -126.1655],
            "dry_weight_kg_m2": [-1.0, 3.12],
        }
        self._create_test_csv("negative_weight_skema.csv", test_data)

        with self.assertRaises(ValueError) as context:
            self.data_acquisition.load_skema_csv_data("negative_weight")

        self.assertIn(
            "dry_weight_kg_m2 values must be non-negative", str(context.exception)
        )

    def test_create_real_ground_truth_from_skema(self):
        """Test creating real ground truth data from SKEMA CSV."""
        # Create valid test CSV
        test_data = {
            "lat": [50.0833, 50.0845, 50.0820],
            "lon": [-126.1667, -126.1655, -126.1680],
            "dry_weight_kg_m2": [2.45, 3.12, 1.87],
        }
        self._create_test_csv("broughton_archipelago_skema.csv", test_data)

        # Create mock satellite scenes
        import datetime

        scenes = [
            SatelliteScene(
                scene_id="test_scene_001",
                acquisition_date=datetime.datetime(2024, 6, 15),
                site_id="broughton_archipelago",
                cloud_coverage=10.0,
                data_quality="excellent",
            )
        ]

        # Test creating ground truth
        ground_truth = self.data_acquisition.create_real_ground_truth_from_skema(
            "broughton_archipelago", scenes
        )

        # Verify results
        self.assertEqual(len(ground_truth), 1)
        gt = ground_truth[0]
        self.assertEqual(gt.site_id, "broughton_archipelago")
        self.assertEqual(gt.data_type, "skema_field_measurement")
        self.assertEqual(gt.source, "SKEMA CSV data")
        self.assertGreater(gt.confidence, 0.0)
        self.assertLessEqual(gt.confidence, 1.0)
        self.assertGreaterEqual(gt.kelp_coverage_percent, 0.0)
        self.assertLessEqual(gt.kelp_coverage_percent, 100.0)
        self.assertIn(gt.kelp_density, ["sparse", "moderate", "dense", "very_dense"])

        # Check metadata
        self.assertIn("skema_measurements", gt.metadata)
        self.assertIn("avg_dry_weight_kg_m2", gt.metadata)
        self.assertIn("lat_range", gt.metadata)
        self.assertIn("lon_range", gt.metadata)
        self.assertEqual(gt.metadata["skema_measurements"], 3)

    def test_create_real_ground_truth_fallback_to_synthetic(self):
        """Test fallback to synthetic when SKEMA data is unavailable."""
        # Use an existing site but ensure no CSV exists
        site_id = "point_reyes"  # Use a different site that we haven't created CSV for

        # Make sure the CSV file doesn't exist
        csv_path = self.sample_data_dir / f"{site_id}_skema.csv"
        if csv_path.exists():
            csv_path.unlink()

        # Create mock satellite scenes
        import datetime

        scenes = [
            SatelliteScene(
                scene_id="test_scene_001",
                acquisition_date=datetime.datetime(2024, 6, 15),
                site_id=site_id,
                cloud_coverage=10.0,
                data_quality="excellent",
            )
        ]

        # Test creating ground truth (should fallback to synthetic since no CSV exists for this site)
        ground_truth = self.data_acquisition.create_real_ground_truth_from_skema(
            site_id,
            scenes,  # Use existing site but no CSV file exists
        )

        # Should still return ground truth (synthetic fallback)
        self.assertEqual(len(ground_truth), 1)
        gt = ground_truth[0]
        self.assertEqual(gt.site_id, site_id)
        # Should be synthetic data type, not SKEMA
        self.assertEqual(gt.data_type, "synthetic_validation")

    @patch("kelpie_carbon.data.real_data_acquisition.Path.exists")
    def test_create_validation_dataset_with_real_data(self, mock_exists):
        """Test creating validation dataset with real SKEMA data."""
        # Mock the CSV file exists
        mock_exists.return_value = True

        # Create valid test CSV
        test_data = {
            "lat": [50.0833, 50.0845, 50.0820],
            "lon": [-126.1667, -126.1655, -126.1680],
            "dry_weight_kg_m2": [2.45, 3.12, 1.87],
        }
        self._create_test_csv("broughton_archipelago_skema.csv", test_data)

        # Test creating dataset with real data (use_synthetic=False)
        dataset = self.data_acquisition.create_validation_dataset(
            "broughton_archipelago", num_scenes=2, use_synthetic=False
        )

        # Verify dataset structure
        self.assertEqual(dataset.site.site_id, "broughton_archipelago")
        self.assertEqual(len(dataset.satellite_scenes), 2)
        self.assertEqual(len(dataset.ground_truth), 2)

        # Verify ground truth is from SKEMA data
        gt = dataset.ground_truth[0]
        self.assertEqual(gt.data_type, "skema_field_measurement")
        self.assertEqual(gt.source, "SKEMA CSV data")

    def test_density_classification(self):
        """Test kelp density classification based on dry weight."""
        test_cases = [
            ({"dry_weight_kg_m2": [0.5, 0.8]}, "sparse", "broughton_archipelago"),
            ({"dry_weight_kg_m2": [1.5, 2.5]}, "moderate", "monterey_bay"),
            ({"dry_weight_kg_m2": [4.0, 5.0]}, "dense", "saanich_inlet"),
            ({"dry_weight_kg_m2": [8.0, 10.0]}, "very_dense", "puget_sound"),
        ]

        for test_data, expected_density, site_id in test_cases:
            with self.subTest(density=expected_density):
                # Add required columns
                full_data = {
                    "lat": [50.0833, 50.0845],
                    "lon": [-126.1667, -126.1655],
                    **test_data,
                }
                csv_filename = f"{site_id}_skema.csv"
                self._create_test_csv(csv_filename, full_data)

                # Create mock scene
                import datetime

                from kelpie_carbon.data.real_data_acquisition import (
                    SatelliteScene,
                )

                scenes = [
                    SatelliteScene(
                        scene_id="test_scene",
                        acquisition_date=datetime.datetime(2024, 6, 15),
                        site_id=site_id,
                        cloud_coverage=10.0,
                        data_quality="excellent",
                    )
                ]

                # Test ground truth generation
                ground_truth = (
                    self.data_acquisition.create_real_ground_truth_from_skema(
                        site_id, scenes
                    )
                )

                self.assertEqual(ground_truth[0].kelp_density, expected_density)

    @patch("kelpie_carbon.data.real_data_acquisition.pystac_client.Client.open")
    @patch("kelpie_carbon.data.real_data_acquisition.requests.get")
    def test_download_scenes(self, mock_requests_get, mock_client_open):
        """Test downloading Sentinel-2 scenes with monkeypatched STAC client."""
        # Create mock STAC item
        mock_item = MagicMock()
        mock_item.id = "S2A_31UDS_20240615_0_L2A"
        mock_item.collection_id = "sentinel-2-l2a"
        mock_item.properties = {
            "datetime": "2024-06-15T10:30:00Z",
            "eo:cloud_cover": 5.0,
            "platform": "sentinel-2a",
            "instruments": ["msi"],
            "sun_elevation": 45.0,
            "sun_azimuth": 180.0,
        }

        # Mock assets
        mock_b04_asset = MagicMock()
        mock_b04_asset.href = "s3://sentinel-placeholder/B04.tif"
        mock_b08_asset = MagicMock()
        mock_b08_asset.href = "s3://sentinel-placeholder/B08.tif"

        mock_item.assets = {
            "B04": mock_b04_asset,
            "B08": mock_b08_asset,
        }

        # Mock search results
        mock_search = MagicMock()
        mock_search.items.return_value = [mock_item]

        # Mock catalog and client
        mock_catalog = MagicMock()
        mock_catalog.search.return_value = mock_search
        mock_client_open.return_value = mock_catalog

        # Mock HTTP requests
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"dummy_geotiff_data"]
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        # Test download_scenes method
        scenes = self.data_acquisition.download_scenes(
            site_id="broughton_archipelago",
            start="2024-06-01",
            end="2024-06-30",
            max_cloud_pct=10.0,
        )

        # Verify the results
        self.assertEqual(len(scenes), 1)
        scene = scenes[0]

        # Check scene properties
        self.assertEqual(scene.scene_id, "S2A_31UDS_20240615_0_L2A")
        self.assertEqual(scene.site_id, "broughton_archipelago")
        self.assertEqual(scene.cloud_coverage, 5.0)
        self.assertEqual(scene.data_quality, "excellent")

        # Check acquisition date
        expected_date = datetime.datetime(2024, 6, 15, 10, 30, tzinfo=datetime.UTC)
        self.assertEqual(scene.acquisition_date, expected_date)

        # Check file path exists
        self.assertIsNotNone(scene.file_path)
        self.assertTrue(Path(scene.file_path).exists())

        # Check metadata
        self.assertEqual(scene.metadata["platform"], "sentinel-2a")
        self.assertEqual(scene.metadata["collection"], "sentinel-2-l2a")
        self.assertIn("B04", scene.metadata["bands_downloaded"])
        self.assertIn("B08", scene.metadata["bands_downloaded"])

        # Verify mock calls
        mock_client_open.assert_called_once_with(
            "https://earth-search.aws.element84.com/v1"
        )

        # Check that search was called with correct parameters
        mock_catalog.search.assert_called_once()
        call_args = mock_catalog.search.call_args
        self.assertEqual(call_args[1]["collections"], ["sentinel-2-l2a"])
        self.assertEqual(call_args[1]["datetime"], "2024-06-01/2024-06-30")
        self.assertEqual(call_args[1]["query"]["eo:cloud_cover"]["lt"], 10.0)

        # Check bbox is correct (Broughton Archipelago coordinates: 50.0833, -126.1667)
        expected_bbox = [-126.2167, 50.0333, -126.1167, 50.1333]  # 0.05° buffer
        actual_bbox = call_args[1]["bbox"]

        # Compare with tolerance for floating point precision
        self.assertEqual(len(actual_bbox), 4)
        self.assertAlmostEqual(actual_bbox[0], expected_bbox[0], places=4)
        self.assertAlmostEqual(actual_bbox[1], expected_bbox[1], places=4)
        self.assertAlmostEqual(actual_bbox[2], expected_bbox[2], places=4)
        self.assertAlmostEqual(actual_bbox[3], expected_bbox[3], places=4)

        # Verify file downloads were attempted
        self.assertEqual(mock_requests_get.call_count, 2)  # B04 and B08

    def test_download_scenes_invalid_site(self):
        """Test download_scenes with invalid site_id."""
        with self.assertRaises(ValueError) as context:
            self.data_acquisition.download_scenes(
                site_id="nonexistent_site", start="2024-06-01", end="2024-06-30"
            )

        self.assertIn("Unknown site_id", str(context.exception))


if __name__ == "__main__":
    unittest.main()

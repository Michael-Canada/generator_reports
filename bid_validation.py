# Environment: placebo_api_local

"""
Advanced Bid Validation Module

This module provides comprehensive validation tests for generator supply curves
(bids) stored in Google Cloud. It integrates with the generator analysis framework
to identify potential issues with bid configurations.

Key Validation Tests:
1. First Block Quantity vs Pmin Validation
2. Last Block Quantity vs Generation Percentiles (80th percentile, non-nuclear generators only)
3. Pmax vs Generation Analysis (Pmax below 90th percentile of generation)
4. Bid Curve Consistency Checks
5. Multi-unit Resource Logging (informational only - no unit-level capacity data available)

Output Columns:
- generator_name: Generator identifier
- plant_id: Plant EIA ID
- unit_id: Unit identifier
- fuel_type: Generator fuel type (energy source code - NG, BIT, NUC, etc.)
- validation_type: Type of validation issue detected (first_block_below_pmin, last_block_insufficient, pmax_below_generation, bid_curve_inconsistent, unrealistic_price_jumps)
- severity: Issue severity level (critical, high, medium, low, info)
- message: Detailed issue description
- details: Additional validation-specific data
- recommendations: Suggested actions
- timestamp: When issue was detected

Usage:
    # Standalone execution
    python bid_validation.py

    # Programmatic usage
    from bid_validation import BidValidator

    validator = BidValidator(market="miso", config=your_config)
    results = validator.run_comprehensive_validation()
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from google.cloud import storage
from scipy import stats
import requests
import io


class BidValidationLevel(Enum):
    """Severity levels for bid validation issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class BidValidationType(Enum):
    """Types of bid validation issues."""

    # Minimum Capacity Issues
    FIRST_BLOCK_BELOW_PMIN = "first_block_below_pmin"
    PMIN_PMAX_MISMATCH = "pmin_pmax_mismatch"

    # Maximum Capacity Issues
    PMAX_BELOW_GENERATION = "pmax_below_generation"
    LAST_BLOCK_INSUFFICIENT = "last_block_insufficient"

    # Bid Curve Structure Issues
    BID_CURVE_INCONSISTENT = "bid_curve_inconsistent"
    UNREALISTIC_PRICE_JUMPS = "unrealistic_price_jumps"
    BID_BLOCKS_NON_MONOTONIC = "bid_blocks_non_monotonic"

    # Market Participation Issues
    MISSING_BID_DATA = "missing_bid_data"
    INCOMPLETE_BID_CURVE = "incomplete_bid_curve"
    ZERO_QUANTITY_BLOCKS = "zero_quantity_blocks"

    # MULTI_UNIT_INCONSISTENCY = "multi_unit_inconsistency"  # DISABLED - no unit-level capacity data


@dataclass
class BidValidationResult:
    """Container for bid validation results."""

    generator_name: str
    plant_id: Optional[int]
    unit_id: Optional[str]
    validation_type: BidValidationType
    severity: BidValidationLevel
    message: str
    details: Dict[str, Any]
    recommendations: List[str]
    timestamp: str


class BidValidator:
    """Main class for validating generator supply curves."""

    def __init__(self, market: str = "miso", config: Optional[Dict] = None):
        """
        Initialize the BidValidator.

        Args:
            market: Market identifier (miso, spp, ercot, pjm)
            config: Configuration dictionary with API settings
        """
        self.market = market.lower()
        self.config = config or self._get_default_config()

        # Initialize data containers
        self.supply_curves = {}
        self.resource_db = {}
        self.generation_data = {}

        # Validation thresholds
        self.validation_thresholds = {
            "pmin_tolerance": 0.05,  # 5% tolerance for first block vs pmin
            "generation_percentile": 80,  # 80th percentile check
            "pmax_ratio_threshold": 0.9,  # 90% of pmax threshold
            "price_jump_factor": 10.0,  # Factor for detecting unrealistic price jumps
            "min_data_points": 168,  # Minimum hours of generation data (1 week)
            "lookback_hours": 1000,  # Hours to look back for generation analysis
        }

        # Results storage
        self.validation_results = []

    def _get_default_config(self) -> Dict:
        """Get default configuration for the validator."""
        return {
            "URL_ROOT": "https://api1.marginalunit.com/muse/api",
            "GO_TO_GCLOUD": True,
            "gcs_config": {
                "bucket_name": "marginalunit-placebo-metadata",
                "base_paths": {
                    "miso": "metadata/miso.resourcedb/2024-11-19/",
                    "spp": "metadata/spp.resourcedb/2024-11-19/",
                    "ercot": "metadata/ercot.resourcedb.v2/2024-11-25/",
                    "pjm": "metadata/pjm.resourcedb/2024-11-19/",
                },
            },
            "reflow_collections": {
                "miso": "miso-se",
                "spp": "spp-se",
                "ercot": "ercot-rt-se",
                "pjm": "pjm-se",
            },
        }

    def _get_auth(self) -> tuple:
        """Get API authentication credentials."""
        import os

        return tuple(os.environ["MU_API_AUTH"].split(":"))

    def _get_data_from_url(self, url: str) -> Optional[pd.DataFrame]:
        """Fetch data from API URL."""
        try:
            auth = self._get_auth()
            resp = requests.get(url, auth=auth)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text))
        except Exception as e:
            print(f"Error fetching data from {url}: {e}")
            return None

    def load_cloud_data(self) -> bool:
        """
        Load supply curves and resource database from Google Cloud.

        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            client = storage.Client()
            bucket = client.bucket(self.config["gcs_config"]["bucket_name"])
            base_path = self.config["gcs_config"]["base_paths"][self.market]

            # Load supply curves
            supply_curves_path = f"{base_path}supply_curves.json"
            supply_curves_blob = bucket.blob(supply_curves_path)
            supply_curves_json = supply_curves_blob.download_as_text()
            self.supply_curves = json.loads(supply_curves_json)

            # Load resource database
            resources_path = f"{base_path}resources.json"
            resources_blob = bucket.blob(resources_path)
            resources_json = resources_blob.download_as_text()
            resource_list = json.loads(resources_json)
            self.resource_db = {resource["uid"]: resource for resource in resource_list}

            # # Only keep 500 supply curves in the dict self.supply_curves
            # self.supply_curves = {k: v for k, v in list(self.supply_curves.items())[1000:1500]}

            # # only keep 500 resources in the dict self.resource_db
            # self.resource_db = {k: v for k, v in list(self.resource_db.items())[1000:1500]}

            print(
                f"Loaded {len(self.supply_curves)} supply curves and {len(self.resource_db)} resources"
            )
            return True

        except Exception as e:
            print(f"Error loading cloud data: {e}")
            return False

    def get_generator_actual_generation(
        self, generator_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch actual generation data for a generator.

        Args:
            generator_name: Name of the generator

        Returns:
            DataFrame with actual generation data or None if error
        """
        try:
            collection = self.config["reflow_collections"][self.market]
            encoded_name = generator_name.replace(" ", "%20")
            url = f"https://api1.marginalunit.com/reflow/{collection}/generator?name={encoded_name}"

            data = self._get_data_from_url(url)
            if data is not None and len(data) > 0:
                self.generation_data[generator_name] = data
            return data

        except Exception as e:
            print(f"Error fetching generation data for {generator_name}: {e}")
            return None

    def validate_first_block_vs_pmin(
        self, generator_name: str
    ) -> Optional[BidValidationResult]:
        """
        Validate that the first block quantity is not lower than Pmin.

        Args:
            generator_name: Name of the generator to validate

        Returns:
            BidValidationResult if issue found, None otherwise
        """
        try:
            # Check if generator exists in resource database
            if generator_name not in self.resource_db:
                return None

            # Check if generator is missing from supply curves entirely
            if generator_name not in self.supply_curves:
                return None  # Skip reporting missing supply curves

            # Get supply curve data
            supply_curve = self.supply_curves[generator_name]
            offer_curve = supply_curve.get("offer_curve", {})
            blocks = offer_curve.get("blocks", [])

            # Check if blocks array is empty - skip reporting these
            if not blocks:
                return None  # Skip reporting empty bid blocks

            # Get resource data for Pmin validation
            resource = self.resource_db[generator_name]
            physical_props = resource.get("physical_properties", {})
            pmin = physical_props.get("pmin", 0)

            # Check first block
            first_block = blocks[0]
            first_block_quantity = first_block.get("quantity", 0)

            # Apply tolerance
            tolerance = self.validation_thresholds["pmin_tolerance"]
            threshold = pmin * (1 - tolerance)

            if first_block_quantity < threshold:
                severity = (
                    BidValidationLevel.HIGH
                    if first_block_quantity < pmin * 0.8
                    else BidValidationLevel.MEDIUM
                )

                return BidValidationResult(
                    generator_name=generator_name,
                    plant_id=self._get_plant_id(generator_name),
                    unit_id=self._get_unit_id(generator_name),
                    validation_type=BidValidationType.FIRST_BLOCK_BELOW_PMIN,
                    severity=severity,
                    message=f"First bid block quantity ({first_block_quantity:.2f} MW) is below Pmin ({pmin:.2f} MW)",
                    details={
                        "first_block_quantity": first_block_quantity,
                        "pmin": pmin,
                        "ratio": first_block_quantity / pmin if pmin > 0 else 0,
                        "threshold_used": threshold,
                        "first_block_price": first_block.get("price", 0),
                        "fuel_type": self._get_fuel_type(generator_name),
                    },
                    recommendations=[
                        "Increase first block quantity to at least match Pmin",
                        "Review generator minimum stable level configuration",
                        "Check if generator has changed operational characteristics",
                    ],
                    timestamp=datetime.now().isoformat(),
                )

        except Exception as e:
            print(f"Error validating first block for {generator_name}: {e}")

        return None

    def validate_last_block_vs_generation(
        self, generator_name: str
    ) -> Optional[BidValidationResult]:
        """
        Validate that the last block quantity is sufficient compared to observed generation.
        Only runs on non-nuclear generators and checks if last block < 80th percentile of generation.

        Args:
            generator_name: Name of the generator to validate

        Returns:
            BidValidationResult if issue found, None otherwise
        """
        try:
            # Skip validation if generator missing from supply curves or has no blocks
            if generator_name not in self.supply_curves:
                return None

            supply_curve = self.supply_curves[generator_name]
            offer_curve = supply_curve.get("offer_curve", {})
            blocks = offer_curve.get("blocks", [])

            if not blocks:
                return None  # Already handled in validate_first_block_vs_pmin

            # Get resource data
            if generator_name not in self.resource_db:
                return None

            resource = self.resource_db[generator_name]

            # Skip nuclear generators
            fuel_type = self._get_fuel_type(generator_name)
            if fuel_type == "NUC":
                return None  # Skip nuclear generators

            # Get generation data
            generation_data = self.get_generator_actual_generation(generator_name)
            if generation_data is None or len(generation_data) == 0:
                return None

            # Analyze recent generation
            lookback_hours = min(
                self.validation_thresholds["lookback_hours"], len(generation_data)
            )
            recent_generation = generation_data.tail(lookback_hours)["pg"].values

            if len(recent_generation) < self.validation_thresholds["min_data_points"]:
                return None

            # Calculate statistics
            generation_percentile = self.validation_thresholds["generation_percentile"]
            gen_percentile_value = np.percentile(
                recent_generation, generation_percentile
            )
            max_observed_gen = np.max(recent_generation)

            # Get last block quantity
            last_block = blocks[-1]
            last_block_quantity = last_block.get("quantity", 0)

            # Check only the generation percentile condition (removed Pmax condition)
            if last_block_quantity < gen_percentile_value:
                # Determine severity
                ratio_to_max = (
                    last_block_quantity / max_observed_gen
                    if max_observed_gen > 0
                    else 1
                )
                if ratio_to_max < 0.7:
                    severity = BidValidationLevel.CRITICAL
                elif ratio_to_max < 0.8:
                    severity = BidValidationLevel.HIGH
                else:
                    severity = BidValidationLevel.MEDIUM

                return BidValidationResult(
                    generator_name=generator_name,
                    plant_id=self._get_plant_id(generator_name),
                    unit_id=self._get_unit_id(generator_name),
                    validation_type=BidValidationType.LAST_BLOCK_INSUFFICIENT,
                    severity=severity,
                    message=f"Last bid block quantity ({last_block_quantity:.2f} MW) is below {generation_percentile}th percentile of generation ({gen_percentile_value:.2f} MW)",
                    details={
                        "last_block_quantity": last_block_quantity,
                        "generation_percentile_value": gen_percentile_value,
                        "generation_percentile": generation_percentile,
                        "max_observed_generation": max_observed_gen,
                        "ratio_to_max_observed": ratio_to_max,
                        "last_block_price": last_block.get("price", 0),
                        "data_points_analyzed": len(recent_generation),
                        "fuel_type": self._get_fuel_type(generator_name),
                    },
                    recommendations=[
                        f"Increase last block quantity to at least {gen_percentile_value:.2f} MW",
                        "Review recent generation patterns and market conditions",
                        "Consider adding additional bid blocks for higher generation levels",
                    ],
                    timestamp=datetime.now().isoformat(),
                )

        except Exception as e:
            print(f"Error validating last block for {generator_name}: {e}")

        return None

    def validate_pmax_vs_generation(
        self, generator_name: str
    ) -> Optional[BidValidationResult]:
        """
        Validate that Pmax is sufficient compared to the 90th percentile of observed generation.

        Args:
            generator_name: Name of the generator to validate

        Returns:
            BidValidationResult if issue found, None otherwise
        """
        try:
            # Get resource data
            if generator_name not in self.resource_db:
                return None

            resource = self.resource_db[generator_name]
            physical_props = resource.get("physical_properties", {})
            pmax = physical_props.get("pmax", 0)

            if pmax <= 0:
                return None  # Skip generators with no valid Pmax

            # Get generation data
            generation_data = self.get_generator_actual_generation(generator_name)
            if generation_data is None or len(generation_data) == 0:
                return None

            # Analyze recent generation
            lookback_hours = min(
                self.validation_thresholds["lookback_hours"], len(generation_data)
            )
            recent_generation = generation_data.tail(lookback_hours)["pg"].values

            if len(recent_generation) < self.validation_thresholds["min_data_points"]:
                return None

            # Calculate 90th percentile of generation
            gen_90th_percentile = np.percentile(recent_generation, 90)
            max_observed_gen = np.max(recent_generation)

            # Check if Pmax is below 90th percentile of generation
            if pmax < gen_90th_percentile:
                # Determine severity
                ratio_pmax_to_90th = (
                    pmax / gen_90th_percentile if gen_90th_percentile > 0 else 1
                )
                if ratio_pmax_to_90th < 0.8:
                    severity = BidValidationLevel.CRITICAL
                elif ratio_pmax_to_90th < 0.9:
                    severity = BidValidationLevel.HIGH
                else:
                    severity = BidValidationLevel.MEDIUM

                return BidValidationResult(
                    generator_name=generator_name,
                    plant_id=self._get_plant_id(generator_name),
                    unit_id=self._get_unit_id(generator_name),
                    validation_type=BidValidationType.PMAX_BELOW_GENERATION,
                    severity=severity,
                    message=f"Pmax ({pmax:.2f} MW) is below 90th percentile of generation ({gen_90th_percentile:.2f} MW)",
                    details={
                        "pmax": pmax,
                        "generation_90th_percentile": gen_90th_percentile,
                        "max_observed_generation": max_observed_gen,
                        "ratio_pmax_to_90th_percentile": ratio_pmax_to_90th,
                        "data_points_analyzed": len(recent_generation),
                        "fuel_type": self._get_fuel_type(generator_name),
                    },
                    recommendations=[
                        f"Review Pmax setting - generator appears capable of producing up to {gen_90th_percentile:.2f} MW",
                        "Verify generator nameplate capacity with actual performance",
                        "Check if recent upgrades or operational changes increased capacity",
                        "Consider updating resource database with correct Pmax value",
                    ],
                    timestamp=datetime.now().isoformat(),
                )

        except Exception as e:
            print(f"Error validating Pmax for {generator_name}: {e}")

        return None

    def validate_bid_curve_consistency(
        self, generator_name: str
    ) -> List[BidValidationResult]:
        """
        Validate bid curve consistency (monotonic quantities, reasonable price jumps).

        Args:
            generator_name: Name of the generator to validate

        Returns:
            List of BidValidationResult for any issues found
        """
        results = []

        try:
            # Skip validation if generator missing from supply curves or has no blocks
            # (this will be caught by validate_first_block_vs_pmin)
            if generator_name not in self.supply_curves:
                return results

            supply_curve = self.supply_curves[generator_name]
            offer_curve = supply_curve.get("offer_curve", {})
            blocks = offer_curve.get("blocks", [])

            if len(blocks) < 2:
                return results  # Need at least 2 blocks for consistency checks

            # Check for monotonic quantities - use specific type for non-monotonic issues
            for i in range(1, len(blocks)):
                current_quantity = blocks[i].get("quantity", 0)
                previous_quantity = blocks[i - 1].get("quantity", 0)

                if current_quantity <= previous_quantity:
                    results.append(
                        BidValidationResult(
                            generator_name=generator_name,
                            plant_id=self._get_plant_id(generator_name),
                            unit_id=self._get_unit_id(generator_name),
                            validation_type=BidValidationType.BID_BLOCKS_NON_MONOTONIC,
                            severity=BidValidationLevel.HIGH,
                            message=f"Non-monotonic bid curve: Block {i+1} quantity ({current_quantity:.2f}) <= Block {i} quantity ({previous_quantity:.2f})",
                            details={
                                "block_index": i + 1,
                                "current_quantity": current_quantity,
                                "previous_quantity": previous_quantity,
                                "current_price": blocks[i].get("price", 0),
                                "previous_price": blocks[i - 1].get("price", 0),
                                "fuel_type": self._get_fuel_type(generator_name),
                            },
                            recommendations=[
                                "Ensure bid block quantities are strictly increasing",
                                "Review bid curve generation logic",
                                "Check for data entry errors in bid blocks",
                            ],
                            timestamp=datetime.now().isoformat(),
                        )
                    )

            # Check for zero quantity blocks
            for i, block in enumerate(blocks):
                quantity = block.get("quantity", 0)
                if quantity <= 0:
                    results.append(
                        BidValidationResult(
                            generator_name=generator_name,
                            plant_id=self._get_plant_id(generator_name),
                            unit_id=self._get_unit_id(generator_name),
                            validation_type=BidValidationType.ZERO_QUANTITY_BLOCKS,
                            severity=BidValidationLevel.MEDIUM,
                            message=f"Zero quantity in bid block {i+1}: {quantity:.2f} MW",
                            details={
                                "block_index": i + 1,
                                "quantity": quantity,
                                "price": block.get("price", 0),
                                "fuel_type": self._get_fuel_type(generator_name),
                            },
                            recommendations=[
                                "Remove or fix zero-quantity bid blocks",
                                "Verify minimum operating parameters",
                                "Check bid generation algorithm",
                            ],
                            timestamp=datetime.now().isoformat(),
                        )
                    )

            # Check for unrealistic price jumps
            price_jump_factor = self.validation_thresholds["price_jump_factor"]
            for i in range(1, len(blocks)):
                current_price = blocks[i].get("price", 0)
                previous_price = blocks[i - 1].get("price", 0)

                if (
                    previous_price > 0
                    and current_price > previous_price * price_jump_factor
                ):
                    results.append(
                        BidValidationResult(
                            generator_name=generator_name,
                            plant_id=self._get_plant_id(generator_name),
                            unit_id=self._get_unit_id(generator_name),
                            validation_type=BidValidationType.UNREALISTIC_PRICE_JUMPS,
                            severity=BidValidationLevel.MEDIUM,
                            message=f"Large price jump in bid curve: Block {i+1} price (${current_price:.2f}) is {current_price/previous_price:.1f}x Block {i} price (${previous_price:.2f})",
                            details={
                                "block_index": i + 1,
                                "current_price": current_price,
                                "previous_price": previous_price,
                                "price_ratio": (
                                    current_price / previous_price
                                    if previous_price > 0
                                    else float("inf")
                                ),
                                "threshold_factor": price_jump_factor,
                                "fuel_type": self._get_fuel_type(generator_name),
                            },
                            recommendations=[
                                "Review price escalation logic in bid curve",
                                "Consider more gradual price increases between blocks",
                                "Verify fuel cost calculations",
                            ],
                            timestamp=datetime.now().isoformat(),
                        )
                    )

        except Exception as e:
            print(f"Error validating bid curve consistency for {generator_name}: {e}")

        return results

    def validate_multi_unit_consistency(
        self, generator_name: str
    ) -> List[BidValidationResult]:
        """
        Log multi-unit resources for informational purposes.

        NOTE: Unit-level capacity validation is DISABLED because unit-level
        capacity data (Pmin/Pmax) does not exist in the data sources.
        Only resource-level aggregated capacity is available.

        Args:
            generator_name: Name of the generator to validate

        Returns:
            Empty list (no validation performed)
        """
        results = []

        try:
            if generator_name not in self.resource_db:
                return results

            resource = self.resource_db[generator_name]
            generators = resource.get("generators", [])

            if len(generators) <= 1:
                return results  # Not a multi-unit resource

            # Log multi-unit resource for informational purposes only
            physical_props = resource.get("physical_properties", {})
            resource_pmin = physical_props.get("pmin", 0)
            resource_pmax = physical_props.get("pmax", 0)

            print(f"INFO: Multi-unit resource found: {generator_name}")
            print(
                f"      Units: {len(generators)}, Resource Pmax: {resource_pmax:.1f} MW"
            )
            for i, gen in enumerate(generators, 1):
                eia_uid = gen.get("eia_uid", {})
                plant_id = eia_uid.get("eia_id", "N/A")
                unit_id = eia_uid.get("unit_id", "N/A")
                print(f"        Unit {i}: Plant {plant_id}, Unit {unit_id}")

            # NOTE: No validation performed because unit-level capacity data doesn't exist
            # Previous validation logic would have checked:
            # - sum(unit_pmin) vs resource_pmin
            # - sum(unit_pmax) vs resource_pmax
            # But unit-level pmin/pmax fields are not available in the data

        except Exception as e:
            print(f"Error logging multi-unit resource {generator_name}: {e}")

        return results

    def validate_bid_completeness(
        self, generator_name: str
    ) -> Optional[BidValidationResult]:
        """
        Validate that bid curves are complete and properly structured.

        Args:
            generator_name: Name of the generator to validate

        Returns:
            BidValidationResult if incomplete bid curve found, None otherwise
        """
        try:
            # Skip validation if generator missing from supply curves
            if generator_name not in self.supply_curves:
                return None

            supply_curve = self.supply_curves[generator_name]
            offer_curve = supply_curve.get("offer_curve", {})
            blocks = offer_curve.get("blocks", [])

            # Check if bid curve exists but is very short for a generator with capacity
            if generator_name in self.resource_db:
                resource = self.resource_db[generator_name]
                physical_props = resource.get("physical_properties", {})
                pmax = physical_props.get("pmax", 0)

                # For generators with significant capacity, expect more than 1-2 bid blocks
                if pmax > 50 and len(blocks) == 0:  # 50 MW threshold
                    return BidValidationResult(
                        generator_name=generator_name,
                        plant_id=self._get_plant_id(generator_name),
                        unit_id=self._get_unit_id(generator_name),
                        validation_type=BidValidationType.INCOMPLETE_BID_CURVE,
                        severity=BidValidationLevel.MEDIUM,
                        message=f"Incomplete bid curve: Only {len(blocks)} blocks for {pmax:.1f} MW generator",
                        details={
                            "num_blocks": len(blocks),
                            "generator_pmax": pmax,
                            "expected_min_blocks": 3,
                            "fuel_type": self._get_fuel_type(generator_name),
                        },
                        recommendations=[
                            f"Consider expanding bid curve to better represent {pmax:.1f} MW capacity",
                            "Add additional bid blocks for operational flexibility",
                            "Review bid curve generation parameters",
                        ],
                        timestamp=datetime.now().isoformat(),
                    )

        except Exception as e:
            print(f"Error validating bid completeness for {generator_name}: {e}")

        return None

    def validate_single_generator(
        self, generator_name: str
    ) -> List[BidValidationResult]:
        """
        Run all validation tests for a single generator.

        Args:
            generator_name: Name of the generator to validate

        Returns:
            List of all validation results for the generator
        """
        results = []

        # Run individual validation tests
        first_block_result = self.validate_first_block_vs_pmin(generator_name)
        if first_block_result:
            results.append(first_block_result)

        last_block_result = self.validate_last_block_vs_generation(generator_name)
        if last_block_result:
            results.append(last_block_result)

        # Pmax vs generation validation
        pmax_result = self.validate_pmax_vs_generation(generator_name)
        if pmax_result:
            results.append(pmax_result)

        # Bid curve consistency checks
        consistency_results = self.validate_bid_curve_consistency(generator_name)
        results.extend(consistency_results)

        # Bid completeness validation
        completeness_result = self.validate_bid_completeness(generator_name)
        if completeness_result:
            results.append(completeness_result)

        # Multi-unit resource logging (informational only - no validation)
        multi_unit_results = self.validate_multi_unit_consistency(generator_name)
        results.extend(
            multi_unit_results
        )  # Will be empty list since no validation is performed

        return results

    def run_comprehensive_validation(
        self, generator_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run comprehensive validation on all or specified generators.

        Args:
            generator_list: Optional list of specific generators to validate

        Returns:
            DataFrame with all validation results
        """
        print("Starting comprehensive bid validation...")

        # Load data if not already loaded
        if not self.supply_curves or not self.resource_db:
            print("Loading cloud data...")
            if not self.load_cloud_data():
                print("❌ Failed to load cloud data")
                return pd.DataFrame()

        # Debug data availability
        print(f"📊 Data Status:")
        print(
            f"  • Supply curves available: {len(self.supply_curves) if self.supply_curves else 0}"
        )
        print(
            f"  • Resource DB available: {len(self.resource_db) if self.resource_db else 0}"
        )

        # Determine generators to validate
        if generator_list is None:
            # Only validate generators that have non-empty bid blocks and are not retired
            if not self.supply_curves:
                print("❌ No supply curves data available")
                return pd.DataFrame()

            # Filter to only include generators with actual bid blocks and not retired and not nuclear
            generators_to_validate = []
            retired_count = 0
            nuclear_count = 0
            for generator_name in self.supply_curves.keys():
                # Check if generator is retired
                if self._is_generator_retired(generator_name):
                    retired_count += 1
                    continue

                # Check if generator is nuclear (exclude from bid validation)
                fuel_type = self._get_fuel_type(generator_name)
                if fuel_type and fuel_type.upper() in ["NUC", "NUCLEAR"]:
                    nuclear_count += 1
                    continue

                supply_curve = self.supply_curves[generator_name]
                offer_curve = supply_curve.get("offer_curve", {})
                blocks = offer_curve.get("blocks", [])
                if blocks:  # Only include generators with non-empty bid blocks
                    generators_to_validate.append(generator_name)
        else:
            # For specified list, only validate generators with bid blocks and not retired and not nuclear
            generators_to_validate = []
            retired_count = 0
            nuclear_count = 0
            for generator_name in generator_list:
                # Check if generator is retired
                if self._is_generator_retired(generator_name):
                    retired_count += 1
                    continue

                # Check if generator is nuclear (exclude from bid validation)
                fuel_type = self._get_fuel_type(generator_name)
                if fuel_type and fuel_type.upper() in ["NUC", "NUCLEAR"]:
                    nuclear_count += 1
                    continue

                if generator_name in self.supply_curves and self.supply_curves[
                    generator_name
                ].get("offer_curve", {}).get("blocks", []):
                    generators_to_validate.append(generator_name)

        print(
            f"Validating {len(generators_to_validate)} active generators with bid blocks..."
        )
        print(f"  - Generators with supply curves: {len(self.supply_curves)}")
        print(
            f"  - Active generators with non-empty bid blocks: {len(generators_to_validate)}"
        )
        print(f"  - Generators in resource database: {len(self.resource_db)}")

        if retired_count > 0:
            print(f"  - Retired generators (excluded from validation): {retired_count}")
        if nuclear_count > 0:
            print(f"  - Nuclear generators (excluded from validation): {nuclear_count}")

        # Count generators with empty bid blocks for reporting
        empty_blocks_count = 0
        if self.supply_curves:
            for generator_name in self.supply_curves.keys():
                if not self._is_generator_retired(
                    generator_name
                ):  # Only count active generators
                    supply_curve = self.supply_curves[generator_name]
                    offer_curve = supply_curve.get("offer_curve", {})
                    blocks = offer_curve.get("blocks", [])
                    if not blocks:
                        empty_blocks_count += 1

        if empty_blocks_count > 0:
            print(
                f"  - Active generators with empty bid blocks (excluded from validation): {empty_blocks_count}"
            )

        # Run validation for each generator
        all_results = []
        for i, generator_name in enumerate(generators_to_validate):
            if i % 100 == 0:
                print(
                    f"Progress: {i}/{len(generators_to_validate)} generators processed"
                )

            generator_results = self.validate_single_generator(generator_name)
            all_results.extend(generator_results)

        # Store results
        self.validation_results = all_results

        # Convert to DataFrame
        if all_results:
            results_data = []
            for result in all_results:
                row = {
                    "generator_name": result.generator_name,
                    "plant_id": result.plant_id,
                    "unit_id": result.unit_id,
                    "validation_type": result.validation_type.value,
                    "severity": result.severity.value,
                    "message": result.message,
                    "timestamp": result.timestamp,
                    **result.details,  # Flatten details into columns
                }
                results_data.append(row)

            results_df = pd.DataFrame(results_data)
            print(
                f"\nValidation complete. Found {len(all_results)} issues across {len(generators_to_validate)} generators."
            )
            return results_df
        else:
            print("No validation issues found.")
            return pd.DataFrame()

    def generate_summary_report(self) -> pd.DataFrame:
        """
        Generate a summary report of validation results.

        Returns:
            DataFrame with summary statistics
        """
        if not self.validation_results:
            return pd.DataFrame()

        # Count issues by type and severity
        issue_counts = {}
        severity_counts = {}

        for result in self.validation_results:
            issue_type = result.validation_type.value
            severity = result.severity.value

            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        # Create summary
        summary_data = {
            "total_issues": len(self.validation_results),
            "unique_generators_affected": len(
                set(r.generator_name for r in self.validation_results)
            ),
            "timestamp": datetime.now().isoformat(),
            **{f"issues_{k}": v for k, v in issue_counts.items()},
            **{f"severity_{k}": v for k, v in severity_counts.items()},
        }

        return pd.DataFrame([summary_data])

    def save_results(
        self, filename_prefix: str = "bid_validation", generate_pdf: bool = True
    ) -> None:
        """
        Save validation results to CSV files and generate comprehensive PDF report.

        Args:
            filename_prefix: Prefix for output filenames
            generate_pdf: Whether to generate PDF report (set to False when integrated with performance analysis)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results if we have validation results
        if self.validation_results:
            # Convert validation results to DataFrame
            results_data = []
            for result in self.validation_results:
                row = {
                    "generator_name": result.generator_name,
                    "plant_id": result.plant_id,
                    "unit_id": result.unit_id,
                    "fuel_type": self._get_fuel_type(result.generator_name)
                    or "Unknown",
                    "validation_type": result.validation_type.value,
                    "severity": result.severity.value,
                    "message": result.message,
                    "timestamp": result.timestamp,
                    **result.details,  # Flatten details into columns
                }
                results_data.append(row)

            if results_data:
                results_df = pd.DataFrame(results_data)
                detailed_filename = (
                    f"{filename_prefix}_detailed_{self.market}_{timestamp}.csv"
                )
                results_df.to_csv(detailed_filename, index=False)
                print(f"✅ Detailed results saved to: {detailed_filename}")

                # Store results for PDF generation and integration
                self.final_results = results_df

                ## Generate PDF report only if requested (standalone mode)
                if generate_pdf:
                    try:
                        from performance_report_generator import (
                            PerformanceReportGenerator,
                        )

                        print("\n📄 GENERATING BID VALIDATION PDF REPORT...")
                        report_generator = PerformanceReportGenerator(
                            self.config if hasattr(self, "config") else None
                        )

                        # Create proper results_df for ALL generators with validation issues
                        # This provides data for Performance Classification, Advanced Metrics, and Statistical Anomaly sections
                        unique_generators = results_df["generator_name"].unique()

                        # Create performance-style data for each generator with validation issues
                        performance_data = []
                        for gen_name in unique_generators:
                            gen_issues = results_df[
                                results_df["generator_name"] == gen_name
                            ]

                            # Calculate bid validation performance score based on severity
                            critical_count = len(
                                gen_issues[gen_issues["severity"] == "critical"]
                            )
                            high_count = len(
                                gen_issues[gen_issues["severity"] == "high"]
                            )
                            medium_count = len(
                                gen_issues[gen_issues["severity"] == "medium"]
                            )
                            low_count = len(gen_issues[gen_issues["severity"] == "low"])

                            # Calculate score (0-100 based on issue severity)
                            total_issues = len(gen_issues)
                            if critical_count > 0:
                                classification = "critical"
                                score = max(10.0, 50.0 - (critical_count * 20))
                            elif high_count > 0:
                                classification = "poor"
                                score = max(25.0, 70.0 - (high_count * 15))
                            elif medium_count > 0:
                                classification = "fair"
                                score = max(50.0, 80.0 - (medium_count * 10))
                            elif low_count > 0:
                                classification = "good"
                                score = max(70.0, 90.0 - (low_count * 5))
                            else:
                                classification = "excellent"
                                score = 95.0

                            # Get additional information from first issue
                            first_issue = gen_issues.iloc[0]
                            fuel_type = first_issue.get("fuel_type", "Unknown")
                            plant_id = first_issue.get("plant_id", None)
                            unit_id = first_issue.get("unit_id", None)

                            # Create R-squared value based on validation performance (inverted for sorting)
                            r_squared = score / 100.0

                            # Calculate synthetic metrics for advanced analysis
                            rmse_value = max(
                                1.0, (100 - score) * 0.5
                            )  # Higher issues = higher RMSE
                            mae_value = rmse_value * 0.8
                            consistency_score = min(1.0, score / 100.0)
                            volatility_score = rmse_value * 1.2

                            performance_data.append(
                                {
                                    "name": gen_name,
                                    "performance_classification": classification,
                                    "RMSE_over_generation": rmse_value,
                                    "MAE_over_generation": mae_value,
                                    "R_SQUARED": r_squared,
                                    "fuel_type": fuel_type,
                                    "plant_id": plant_id,
                                    "unit_id": unit_id,
                                    "performance_score": score,
                                    "total_bid_issues": total_issues,
                                    "critical_issues": critical_count,
                                    "high_issues": high_count,
                                    "medium_issues": medium_count,
                                    "low_issues": low_count,
                                    "consistency_score": consistency_score,
                                    "volatility_score": volatility_score,
                                    "zone_uid": first_issue.get("zone_uid", "Unknown"),
                                }
                            )

                        # If no validation issues found, create summary entry
                        if not performance_data:
                            performance_data.append(
                                {
                                    "name": "No Issues Found",
                                    "performance_classification": "excellent",
                                    "RMSE_over_generation": 0.0,
                                    "MAE_over_generation": 0.0,
                                    "R_SQUARED": 1.0,
                                    "fuel_type": "All",
                                    "plant_id": None,
                                    "unit_id": None,
                                    "performance_score": 100.0,
                                    "total_bid_issues": 0,
                                    "critical_issues": 0,
                                    "high_issues": 0,
                                    "medium_issues": 0,
                                    "low_issues": 0,
                                    "consistency_score": 1.0,
                                    "volatility_score": 0.0,
                                    "zone_uid": "All",
                                }
                            )

                        performance_results_df = pd.DataFrame(performance_data)

                        # Create chronic error data structure for bid validation context
                        # Since bid validation doesn't have time-series chronic errors, create empty structure
                        chronic_anomalies_df = pd.DataFrame()
                        chronic_alerts = []

                        # Create bid validation specific alerts from high severity issues
                        high_severity_issues = results_df[
                            results_df["severity"].isin(["critical", "high"])
                        ]
                        bid_alerts = []
                        for _, issue in high_severity_issues.iterrows():
                            alert = {
                                "timestamp": issue.get(
                                    "timestamp", datetime.now().isoformat()
                                ),
                                "generator": issue["generator_name"],
                                "alert_type": f"BID_VALIDATION_{issue['validation_type'].upper()}",
                                "severity": issue["severity"],
                                "message": issue["message"],
                                "details": {
                                    "validation_type": issue["validation_type"],
                                    "fuel_type": issue.get("fuel_type", "Unknown"),
                                    "plant_id": issue.get("plant_id"),
                                    "unit_id": issue.get("unit_id"),
                                },
                            }
                            bid_alerts.append(alert)

                        pdf_filename = (
                            f"bid_validation_report_{self.market}_{timestamp}.pdf"
                        )

                        # Generate PDF with bid validation focus and proper data for all sections
                        report_generator.generate_comprehensive_report(
                            results_df=performance_results_df,
                            anomalies_df=chronic_anomalies_df,
                            alerts=bid_alerts,
                            bid_validation_results=results_df,
                            market=self.market,
                            output_filename=pdf_filename,
                        )
                        print(f"📄 Bid Validation PDF Report generated: {pdf_filename}")
                        print(
                            f"   Report includes {len(performance_data)} generators with validation results"
                        )

                    except ImportError:
                        print(
                            "⚠️  Warning: matplotlib/seaborn not available - PDF report generation skipped"
                        )
                        print("   Install with: pip install matplotlib seaborn")
                    except Exception as e:
                        print(f"⚠️  Warning: PDF report generation failed: {e}")
                        print("   CSV reports will still be generated normally")

        # Save summary report
        summary_df = self.generate_summary_report()
        if not summary_df.empty:
            summary_filename = (
                f"{filename_prefix}_summary_{self.market}_{timestamp}.csv"
            )
            summary_df.to_csv(summary_filename, index=False)
            print(f"✅ Summary report saved to: {summary_filename}")

    def _get_plant_id(self, generator_name: str) -> Optional[int]:
        """Get plant ID for a generator."""
        try:
            if generator_name in self.resource_db:
                generators = self.resource_db[generator_name].get("generators", [])
                if generators:
                    return generators[0].get("eia_uid", {}).get("eia_id")
        except:
            pass
        return None

    def _get_unit_id(self, generator_name: str) -> Optional[str]:
        """Get unit ID for a generator."""
        try:
            if generator_name in self.resource_db:
                generators = self.resource_db[generator_name].get("generators", [])
                if generators:
                    if len(generators) == 1:
                        return generators[0].get("eia_uid", {}).get("unit_id")
                    else:
                        # Multi-unit resource
                        unit_ids = [
                            gen.get("eia_uid", {}).get("unit_id") for gen in generators
                        ]
                        return f"MULTI_{len(generators)}_UNITS"
        except:
            pass
        return None

    def _get_fuel_type(self, generator_name: str) -> Optional[str]:
        """Get fuel type (energy source code) for a generator."""
        try:
            if generator_name in self.resource_db:
                return self.resource_db[generator_name].get("energy_source_code")
        except:
            pass
        return None

    def _is_generator_retired(self, generator_name: str) -> bool:
        """Check if a generator has been retired (end_date is in the past)."""
        try:
            if generator_name in self.resource_db:
                end_date = self.resource_db[generator_name].get("end_date")
                if end_date is not None:
                    from datetime import datetime, date

                    # Parse the end_date string (format: YYYY-MM-DD)
                    retirement_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                    today = date.today()
                    return retirement_date < today
        except Exception as e:
            # If there's any error parsing the date, assume not retired
            pass
        return False


# Example usage and integration functions
def integrate_with_generator_analyzer(analyzer_instance, run_validation: bool = True):
    """
    Integration function to add bid validation to the GeneratorAnalyzer class.

    Args:
        analyzer_instance: Instance of GeneratorAnalyzer
        run_validation: Whether to run bid validation
    """
    if not run_validation:
        return

    print("Initializing bid validation...")
    validator = BidValidator(market=analyzer_instance.config.MARKET)

    # Add validator to analyzer instance
    analyzer_instance.bid_validator = validator

    # Run validation on generators being analyzed
    generator_names = analyzer_instance.generators["name"].tolist()
    validation_results = validator.run_comprehensive_validation(generator_names)

    if not validation_results.empty:
        # Save validation results
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"bid_validation_{analyzer_instance.config.MARKET}_{timestamp}.csv"
        validation_results.to_csv(filename, index=False)

        # Add validation summary to final reports
        critical_issues = validation_results[
            validation_results["severity"] == "critical"
        ]
        high_issues = validation_results[validation_results["severity"] == "high"]

        print(f"\n=== BID VALIDATION SUMMARY ===")
        print(f"Total validation issues found: {len(validation_results)}")
        print(f"Critical issues: {len(critical_issues)}")
        print(f"High severity issues: {len(high_issues)}")
        print(f"Detailed results saved to: {filename}")


def main():
    """Main execution function for bid validation."""
    try:
        print("=" * 60)
        print("COMPREHENSIVE BID VALIDATION ANALYSIS")
        print("=" * 60)

        # Initialize validator
        print("Initializing validator...")
        validator = BidValidator(market="miso")

        # Test data loading first
        print("Loading cloud data...")
        load_success = validator.load_cloud_data()
        if not load_success:
            print("❌ Failed to load cloud data. Cannot proceed with validation.")
            return

        print(f"✅ Data loaded successfully:")
        print(f"  • Supply curves: {len(validator.supply_curves)}")
        print(f"  • Resource database: {len(validator.resource_db)}")

        # Run validation on all generators
        print("\nStarting comprehensive bid validation...")
        results = validator.run_comprehensive_validation()

        print(f"\n� VALIDATION RESULTS:")
        if not results.empty:
            print(f"  • Total validation issues found: {len(results)}")

            # Severity breakdown
            severity_counts = results["severity"].value_counts()
            for severity in ["critical", "high", "medium", "low", "info"]:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    print(f"  • {severity.capitalize()} issues: {count}")

            # Issue type breakdown
            print(f"\n🔍 ISSUE TYPES:")
            type_counts = results["validation_type"].value_counts()
            for issue_type, count in type_counts.items():
                print(f"  • {issue_type.replace('_', ' ').title()}: {count}")

            # Show critical and high severity issues
            critical_high = results[results["severity"].isin(["critical", "high"])]
            if not critical_high.empty:
                print(f"\n⚠️  CRITICAL AND HIGH SEVERITY ISSUES:")
                for _, row in critical_high.head(10).iterrows():  # Show first 10
                    print(f"  • {row['generator_name']}: {row['message']}")
                if len(critical_high) > 10:
                    print(
                        f"  ... and {len(critical_high) - 10} more critical/high issues"
                    )

            # Save results
            print(f"\n💾 SAVING RESULTS...")
            validator.save_results("bid_validation_analysis")

            # Generate and display summary
            summary = validator.generate_summary_report()
            if not summary.empty:
                print(f"\n📊 SUMMARY STATISTICS:")
                print(
                    f"  • Unique generators affected: {summary['unique_generators_affected'].iloc[0]}"
                )
                print(f"  • Analysis timestamp: {summary['timestamp'].iloc[0]}")

        else:
            print("  • No validation issues found!")
            print("✅ All generator bids are properly configured!")

        print("\n" + "=" * 60)
        print("BID VALIDATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Bid validation failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

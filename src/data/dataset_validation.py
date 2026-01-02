#!/usr/bin/python3
"""
Data validation script for L-PBF dataset

Checks:
- Merge integrity (no data loss)
- Missing values
- Data types
- Value ranges (physical constraints)
- Outliers
- Duplicates
"""

import polars as pl
from pathlib import Path
from typing import Dict, List
import json

from src.logging_config import get_logger

logger = get_logger(__name__)


def validate_merge_integrity(
    base_df: pl.DataFrame, merged_df: pl.DataFrame, properties_df: pl.DataFrame
) -> Dict:
    """Check that merge didn't lose or corrupt data"""

    issues = []

    # Check row count
    if len(merged_df) != len(base_df):
        issues.append(f"Row count mismatch: {len(base_df)} → {len(merged_df)}")

    # Check for unmatched materials
    base_materials = set(base_df["Material"].unique())
    property_materials = set(properties_df["Material"].unique())

    unmatched = base_materials - property_materials
    if unmatched:
        issues.append(f"Materials without properties: {unmatched}")

    # Check for nulls in material property columns
    property_cols = [col for col in merged_df.columns if col not in base_df.columns]
    null_counts = merged_df.select(property_cols).null_count()

    for col in property_cols:
        null_count = null_counts[col][0]
        if null_count > 0:
            issues.append(f"Column '{col}' has {null_count} null values")

    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "base_rows": len(base_df),
        "merged_rows": len(merged_df),
        "base_materials": len(base_materials),
        "property_materials": len(property_materials),
    }


def check_missing_values(df: pl.DataFrame) -> Dict:
    """Report missing values by column"""

    null_counts = df.null_count()
    total_rows = len(df)

    missing_report = {}

    for col in df.columns:
        null_count = null_counts[col][0]
        if null_count > 0:
            missing_report[col] = {
                "count": null_count,
                "percentage": (null_count / total_rows) * 100,
            }

    return {
        "total_columns": len(df.columns),
        "columns_with_missing": len(missing_report),
        "details": missing_report,
    }


def check_data_types(df: pl.DataFrame) -> Dict:
    """Validate expected data types"""

    expected_numeric = [
        "Laser power (W)",
        "Scan speed (mm/s)",
        "Hatch distance (μm)",
        "Layer thickness (μm)",
        "Spot size (mm)",
        "D50 (μm)",
        "Relative density (%)",
        "Melting Point (K)",
        "Thermal Conductivity (W/mK)",
        "Density (g/cm^3)",
        "Specific Heat Capacity (J/kgK)",
    ]

    type_issues = []

    for col in expected_numeric:
        if col in df.columns:
            dtype = df[col].dtype
            if not dtype.is_numeric():
                type_issues.append(f"Column '{col}' is {dtype}, expected numeric")

    return {
        "status": "PASS" if not type_issues else "FAIL",
        "issues": type_issues,
        "schema": {col: str(df[col].dtype) for col in df.columns},
    }


def check_value_ranges(df: pl.DataFrame) -> Dict:
    """Check if values are within physically reasonable ranges"""

    # Define expected ranges for L-PBF parameters
    ranges = {
        "Laser power (W)": (50, 500),
        "Scan speed (mm/s)": (100, 3000),
        "Hatch distance (μm)": (30, 250),
        "Layer thickness (μm)": (10, 150),
        "Spot size (mm)": (0.01, 0.5),
        "D50 (μm)": (10, 100),
        "Relative density (%)": (
            50,
            101,
        ),
        "Melting Point (K)": (500, 4000),
        "Thermal Conductivity (W/mK)": (1, 500),
        "Density (g/cm^3)": (1, 100),
        "Specific Heat Capacity (J/kgK)": (100, 2000),
    }

    range_issues = []

    for col, (min_val, max_val) in ranges.items():
        if col not in df.columns:
            continue

        col_data = df[col]

        # Skip null values
        col_data = col_data.drop_nulls()

        if len(col_data) == 0:
            continue

        actual_min = col_data.min()
        actual_max = col_data.max()

        if actual_min < min_val:
            range_issues.append(
                f"{col}: min value {actual_min:.2f} below expected {min_val}"
            )

        if actual_max > max_val:
            range_issues.append(
                f"{col}: max value {actual_max:.2f} above expected {max_val}"
            )

    return {"status": "PASS" if not range_issues else "WARNING", "issues": range_issues}


def check_outliers(df: pl.DataFrame, columns: List[str] | None = None) -> Dict:
    """Detect outliers using IQR method"""

    if columns is None:
        columns = [
            "Laser power (W)",
            "Scan speed (mm/s)",
            "Hatch distance (μm)",
            "Relative density (%)",
        ]

    outlier_report = {}

    for col in columns:
        if col not in df.columns:
            continue

        col_data = df[col].drop_nulls()

        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = col_data.filter((col_data < lower_bound) | (col_data > upper_bound))

        if len(outliers) > 0:
            outlier_report[col] = {
                "count": len(outliers),
                "percentage": (len(outliers) / len(col_data)) * 100,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_values": outliers.to_list()[:10],
            }

    return {
        "columns_checked": len(columns),
        "columns_with_outliers": len(outlier_report),
        "details": outlier_report,
    }


def check_duplicates(df: pl.DataFrame) -> Dict:
    """Check for duplicate rows"""

    # Check full duplicates
    duplicate_rows = df.is_duplicated().sum()

    # Check duplicates on key columns (process parameters)
    key_columns = [
        "Material",
        "Laser power (W)",
        "Scan speed (mm/s)",
        "Hatch distance (μm)",
        "Layer thickness (μm)",
    ]

    # Only check if all columns exist
    if all(col in df.columns for col in key_columns):
        duplicate_configs = df.select(key_columns).is_duplicated().sum()
    else:
        duplicate_configs = None

    return {
        "total_rows": len(df),
        "duplicate_full_rows": duplicate_rows,
        "duplicate_process_configs": duplicate_configs,
        "status": "WARNING" if duplicate_rows > 0 else "PASS",
    }


def get_summary_statistics(df: pl.DataFrame) -> Dict:
    """Generate summary statistics for numeric columns"""

    numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]

    stats = {}

    for col in numeric_cols:
        col_data = df[col].drop_nulls()

        if len(col_data) == 0:
            continue

        stats[col] = {
            "count": len(col_data),
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "q1": float(col_data.quantile(0.25)),
            "median": float(col_data.median()),
            "q3": float(col_data.quantile(0.75)),
            "max": float(col_data.max()),
        }

    return stats


def generate_validation_report(
    merged_df: pl.DataFrame,
    base_df: pl.DataFrame = None,
    properties_df: pl.DataFrame = None,
    output_path: Path | None = None,
) -> Dict:
    """
    Run all validation checks and generate comprehensive report

    Args:
        merged_df: The merged dataset to validate
        base_df: Original dataset (for merge validation)
        properties_df: Material properties (for merge validation)
        output_path: Where to save the report (optional)

    Returns:
        Dictionary containing all validation results
    """

    logger.info("Running validation checks...")

    report = {
        "dataset_info": {
            "total_rows": len(merged_df),
            "total_columns": len(merged_df.columns),
            "columns": merged_df.columns,
        }
    }

    # Merge integrity (if base data provided)
    if base_df is not None and properties_df is not None:
        logger.info("  Checking merge integrity...")
        report["merge_integrity"] = validate_merge_integrity(
            base_df, merged_df, properties_df
        )

    # Missing values
    logger.info("  Checking missing values...")
    report["missing_values"] = check_missing_values(merged_df)

    # Data types
    logger.info("  Checking data types...")
    report["data_types"] = check_data_types(merged_df)

    # Value ranges
    logger.info("  Checking value ranges...")
    report["value_ranges"] = check_value_ranges(merged_df)

    # Outliers
    logger.info("  Checking outliers...")
    report["outliers"] = check_outliers(merged_df)

    # Duplicates
    logger.info("  Checking duplicates...")
    report["duplicates"] = check_duplicates(merged_df)

    # Summary statistics
    logger.info("  Generating statistics...")
    report["summary_statistics"] = get_summary_statistics(merged_df)

    # Overall status
    critical_failures = []
    warnings = []

    if report.get("merge_integrity", {}).get("status") == "FAIL":
        critical_failures.append("Merge integrity check failed")

    if report["data_types"]["status"] == "FAIL":
        critical_failures.append("Data type validation failed")

    if report["missing_values"]["columns_with_missing"] > 0:
        warnings.append(
            f"{report['missing_values']['columns_with_missing']} columns have missing values"
        )

    if report["value_ranges"]["status"] == "WARNING":
        warnings.append("Some values outside expected ranges")

    if report["duplicates"]["status"] == "WARNING":
        warnings.append("Duplicate rows detected")

    report["overall_status"] = {
        "status": "FAIL" if critical_failures else ("WARNING" if warnings else "PASS"),
        "critical_failures": critical_failures,
        "warnings": warnings,
    }

    # Save report if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {output_path}")

    return report


def print_validation_summary(report: Dict):
    """Print human-readable summary of validation report"""

    logger.info("=" * 60)
    logger.info("VALIDATION REPORT SUMMARY")
    logger.info("=" * 60)

    # Overall status
    status = report["overall_status"]["status"]
    status_symbol = "✓" if status == "PASS" else ("⚠" if status == "WARNING" else "✗")
    logger.info(f"Overall Status: {status_symbol} {status}")

    if report["overall_status"]["critical_failures"]:
        logger.error("CRITICAL FAILURES:")
        for failure in report["overall_status"]["critical_failures"]:
            logger.error(f"  ✗ {failure}")

    if report["overall_status"]["warnings"]:
        logger.warning("\nWARNINGS:")
        for warning in report["overall_status"]["warnings"]:
            logger.warning(f"  ⚠ {warning}")

    # Dataset info
    logger.info(
        f"Dataset: {report['dataset_info']['total_rows']} rows, "
        f"{report['dataset_info']['total_columns']} columns"
    )

    # Missing values
    missing = report["missing_values"]
    if missing["columns_with_missing"] > 0:
        logger.warning(f"\nMissing Values:")
        for col, info in missing["details"].items():
            logger.warning(f"  • {col}: {info['count']} ({info['percentage']:.1f}%)")
    else:
        logger.info("\nNo missing values")

    # Outliers
    outliers = report["outliers"]
    if outliers["columns_with_outliers"] > 0:
        logger.warning(f"\nOutliers Detected:")
        for col, info in outliers["details"].items():
            logger.warning(
                f"  • {col}: {info['count']} outliers ({info['percentage']:.1f}%)"
            )

    # Duplicates
    dupes = report["duplicates"]
    if dupes["duplicate_full_rows"] > 0:
        logger.warning(f"\n⚠ {dupes['duplicate_full_rows']} duplicate rows found")

    # Value ranges
    if report["value_ranges"]["issues"]:
        logger.warning(f"\nValue Range Issues:")
        for issue in report["value_ranges"]["issues"][:5]:  # First 5
            logger.warning(f"  • {issue}")

    logger.info("\n" + "=" * 60)


def main():
    """Run validation on merged dataset"""

    # Load data
    merged_df = pl.read_csv(Path("data/processed/lpbf_enriched_v1.csv"))

    # Optionally load base data for merge validation
    try:
        base_df = pl.read_excel(Path("data/raw/Barrionuevo_et_al_dataset.xlsx"))
        properties_df = pl.read_csv(
            Path("data/material_properties/material_properties.csv")
        )
    except Exception as e:
        logger.warning(f"Could not load base data for merge validation: {e}")
        base_df = None
        properties_df = None
        exit(1)

    # Run validation
    report = generate_validation_report(
        merged_df=merged_df,
        base_df=base_df,
        properties_df=properties_df,
        output_path=Path("reports/data_quality_report.json"),
    )

    # Print summary
    print_validation_summary(report)

    # Return exit code based on status
    if report["overall_status"]["status"] == "FAIL":
        exit(1)
    elif report["overall_status"]["status"] == "WARNING":
        exit(0)  # Warnings don't fail the pipeline
    else:
        exit(0)


if __name__ == "__main__":
    main()

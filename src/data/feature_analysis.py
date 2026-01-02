"""
Feature space analysis for training data

Functions to analyse and visualise the valid input ranges based on training data.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from src.config import PROJECT_ROOT
from src.logging_config import get_logger

logger = get_logger(__name__)


def compute_feature_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute statistics for numeric and categorical features

    Args:
        df: Training data DataFrame

    Returns:
        Dictionary with feature statistics
    """
    numeric_features = [
        "Laser Power (W)",
        "Scan Speed (mm/s)",
        "Hatch space (mm)",
        "Layer thickness (mm)",
        "Spot size (mm)",
        "D50 μm",
        "Melting Point (K)",
        "Thermal Conductivity (W/mK)",
        "Density (g/cm^3)",
        "Specific Heat Capacity (J/kgK)",
    ]

    categorical_features = [
        "Material",
        "Density Measurement Method",
        "Atmosphere",
        "Printer Model",
    ]

    stats = {
        "numeric": {},
        "categorical": {},
        "target": {},
    }

    # Numeric feature statistics
    logger.info("Computing numeric feature statistics...")
    for feature in numeric_features:
        if feature not in df.columns:
            logger.warning(f"Feature '{feature}' not found in data")
            continue

        data = df[feature].dropna()

        stats["numeric"][feature] = {
            "count": int(len(data)),
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": float(data.mean()),
            "std": float(data.std()),
            "median": float(data.median()),
            "q1": float(data.quantile(0.25)),
            "q3": float(data.quantile(0.75)),
            "p5": float(data.quantile(0.05)),  # 5th percentile
            "p95": float(data.quantile(0.95)),  # 95th percentile
        }

        logger.info(
            f"{feature}: [{stats['numeric'][feature]['min']:.2f}, "
            f"{stats['numeric'][feature]['max']:.2f}]"
        )

    # Categorical feature values
    logger.info("Computing categorical feature statistics...")
    for feature in categorical_features:
        if feature not in df.columns:
            logger.warning(f"Feature '{feature}' not found in data")
            continue

        values = df[feature].dropna().unique().tolist()
        counts = df[feature].value_counts().to_dict()

        stats["categorical"][feature] = {
            "values": values,
            "counts": counts,
            "n_unique": len(values),
        }

        logger.info(f"{feature}: {len(values)} unique values")

    # Target statistics
    target_col = "RD (%)"
    if target_col in df.columns:
        target_data = df[target_col].dropna()
        stats["target"] = {
            "column": target_col,
            "min": float(target_data.min()),
            "max": float(target_data.max()),
            "mean": float(target_data.mean()),
            "std": float(target_data.std()),
        }

    return stats


def save_feature_statistics(stats: Dict, output_path: Path) -> None:
    """
    Save feature statistics to JSON file

    Args:
        stats: Feature statistics dictionary
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Statistics saved to {output_path}")


def load_feature_statistics(stats_path: Path) -> Dict:
    """
    Load feature statistics from JSON file

    Args:
        stats_path: Path to JSON file

    Returns:
        Feature statistics dictionary
    """
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Feature statistics not found at {stats_path}. "
            "Run 'python scripts/analyse_feature_space.py' first."
        )

    with open(stats_path) as f:
        stats = json.load(f)

    logger.info("Loaded feature statistics")
    return stats


def plot_feature_distributions(
    df: pd.DataFrame, stats: Dict, output_path: Path
) -> None:
    """
    Create visualization of feature distributions

    Args:
        df: Training data DataFrame
        stats: Feature statistics dictionary
        output_path: Path to save plot
    """
    logger.info("Creating feature distribution plots...")

    numeric_features = list(stats["numeric"].keys())
    n_features = len(numeric_features)

    # Create subplots
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx, feature in enumerate(numeric_features):
        ax = axes[idx]

        if feature not in df.columns:
            ax.text(0.5, 0.5, f"{feature}\nNot Found", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        data = df[feature].dropna()
        feature_stats = stats["numeric"][feature]

        # Histogram
        ax.hist(data, bins=30, alpha=0.7, color="skyblue", edgecolor="black")

        # Add vertical lines for key statistics
        ax.axvline(
            feature_stats["mean"],
            color="red",
            linestyle="--",
            label="Mean",
            linewidth=2,
        )
        ax.axvline(
            feature_stats["median"],
            color="green",
            linestyle="--",
            label="Median",
            linewidth=2,
        )
        ax.axvline(
            feature_stats["p5"],
            color="orange",
            linestyle=":",
            label="5th/95th %ile",
            linewidth=1.5,
        )
        ax.axvline(feature_stats["p95"], color="orange", linestyle=":", linewidth=1.5)

        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.set_title(
            f"{feature}\n[{feature_stats['min']:.1f}, {feature_stats['max']:.1f}]",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Distribution plot saved to {output_path}")

    plt.close()


def create_summary_table(stats: Dict, output_path: Path) -> pd.DataFrame:
    """
    Create a summary table of feature ranges

    Args:
        stats: Feature statistics dictionary
        output_path: Path to save CSV file

    Returns:
        Summary DataFrame
    """
    logger.info("Creating feature summary table...")

    rows = []
    for feature, feature_stats in stats["numeric"].items():
        rows.append(
            {
                "Feature": feature,
                "Min": f"{feature_stats['min']:.2f}",
                "5th %": f"{feature_stats['p5']:.2f}",
                "Mean": f"{feature_stats['mean']:.2f}",
                "95th %": f"{feature_stats['p95']:.2f}",
                "Max": f"{feature_stats['max']:.2f}",
                "Samples": feature_stats["count"],
            }
        )

    summary_df = pd.DataFrame(rows)

    # Save as CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Summary table saved to {output_path}")

    return summary_df


def print_statistics_summary(stats: Dict, summary_df: pd.DataFrame) -> None:
    """
    Print feature statistics to console

    Args:
        stats: Feature statistics dictionary
        summary_df: Summary DataFrame
    """
    # Print numeric summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING DATA FEATURE RANGES")
    logger.info("=" * 80)
    logger.info("\n" + summary_df.to_string(index=False))
    logger.info("\n" + "=" * 80)

    # Print categorical summaries
    logger.info("\nCATEGORICAL FEATURES:")
    for feature, cat_stats in stats["categorical"].items():
        logger.info(f"\n{feature}:")
        for value, count in cat_stats["counts"].items():
            logger.info(f"  - {value}: {count} samples")


def analyse_feature_space(data_path: Path) -> Dict:
    """
    Complete feature space analysis pipeline

    Args:
        data_path: Path to training data CSV

    Returns:
        Feature statistics dictionary
    """
    logger.info("Starting feature space analysis...")
    logger.info(f"Loading data from {data_path}")

    # Load data
    df = pd.read_csv(data_path)

    # Compute statistics
    stats = compute_feature_statistics(df)

    # Save statistics
    stats_output = PROJECT_ROOT / "reports" / "feature_space_stats.json"
    save_feature_statistics(stats, stats_output)

    # Create visualizations
    plot_output = PROJECT_ROOT / "reports" / "figures" / "feature_distributions.png"
    plot_feature_distributions(df, stats, plot_output)

    # Create summary table
    table_output = PROJECT_ROOT / "reports" / "feature_ranges.csv"
    summary_df = create_summary_table(stats, table_output)

    # Print summary
    print_statistics_summary(stats, summary_df)

    logger.info("\nFeature space analysis complete!")
    logger.info("  - Statistics: reports/feature_space_stats.json")
    logger.info("  - Distributions: reports/figures/feature_distributions.png")
    logger.info("  - Summary table: reports/feature_ranges.csv")

    return stats

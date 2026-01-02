#!/usr/bin/env python3
"""
Script to analyse training data feature space
"""

from pathlib import Path
from src.data.feature_analysis import analyse_feature_space


def main():
    """Run feature space analysis on training data"""
    data_path = Path("data/processed/lpbf_enriched_v1.csv")
    analyse_feature_space(data_path)


if __name__ == "__main__":
    main()

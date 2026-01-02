#!/usr/bin/python3
import polars as pl
from pathlib import Path


def load_material_properties(props_path: Path) -> pl.DataFrame:
    """Load material thermophysical properties"""
    df_materials = pl.read_csv(props_path)

    return df_materials


def load_base_dataset(dataset_path: Path) -> pl.DataFrame:
    """Load base dataset"""

    df_excel = pl.read_excel(dataset_path)

    return df_excel


def merge_dataset(base_df: pl.DataFrame, properties_df: pl.DataFrame) -> pl.DataFrame:
    """Merge base dataset with material properties"""
    df_merged = base_df.join(
        properties_df,
        left_on="Material",  # column name in Excel file
        right_on="Material",  # column name in CSV file
        how="left",
    )

    return df_merged


def main():
    """Run script"""
    df_properties = load_material_properties(
        Path("data/material_properties/material_properties.csv")
    )
    df_base = load_base_dataset(Path("data/raw/Barrionuevo_et_al_dataset.xlsx"))

    df_merged = merge_dataset(df_base, df_properties)

    df_merged.write_csv(Path("data/processed/lpbf_enriched_v1.csv"))


if __name__ == "__main__":
    main()

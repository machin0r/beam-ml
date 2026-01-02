# BEAM-ML

**B**uild **E**valuation for **A**dditive **M**anufacturing

Machine learning system for predicting relative density in Laser Powder Bed Fusion (L-PBF) additive manufacturing. Combines process parameters with material thermophysical properties to improve density prediction accuracy.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate        # Linux/Mac

# 3. Install project
pip install -e .

# 4. Train models
python scripts/train_model.py

# 5. View results in MLflow UI
mlflow ui
# Then open http://localhost:5000
```

**With uv:**
```bash
uv venv && .venv\Scripts\Activate.ps1 && uv pip install -e .
```

## Project Structure

```
├── data/               # Raw and processed datasets
├── scripts/            # Training and analysis scripts
├── src/                # Core package
│   ├── config.py      # Configuration
│   ├── data/          # Data processing
│   ├── models/        # Model implementations
│   └── logging_config.py
├── tests/             # Test suite
└── logs/              # Application logs
```

## Data Provenance

This project builds upon the L-PBF dataset published by Barrionuevo et al.:

**Base Dataset:**
- Barrionuevo, G.O., La Fé-Perdomo, I. & Ramos-Grez, J.A. (2025)
- *Laser powder bed fusion dataset for relative density prediction of commercial metallic alloys*
- Scientific Data 12, 375
- DOI: [10.1038/s41597-025-04576-x](https://doi.org/10.1038/s41597-025-04576-x)
- Dataset: [Harvard Dataverse](https://doi.org/10.7910/DVN/VPBQK8)
- License: CC0 1.0 (Public Domain)

**Our Contribution:**
We extend this dataset by enriching it with thermophysical properties (melting point, thermal conductivity, density, specific heat capacity) for each material from NIST, ASM Handbook, and manufacturer datasheets.


- https://asm.matweb.com/search/specificmaterial.asp?bassnum=ninc34
- https://www.renishaw.com/resourcecentre/download/data-sheet-renam-500-series-aluminium-alsi10mg-material-data-sheet--138868?userLanguage=en&&srsltid=AfmBOorf51bVWj96eGtwhx7_hEHThwE6vD8TriwUm3IzQQOD-tN2TpAi
- https://xometry.pro/wp-content/uploads/2023/08/Aluminium-1706-1.pdf
- https://www.matweb.com/search/datasheet_print.aspx?matguid=1336be6d0c594b55afb5ca8bf1f3e042
- https://www.aurubis.com/dam/jcr:d6d50d64-69d6-4742-821d-8fb8b48c9044/cucrzr-c18160-pna-372_en.pdf


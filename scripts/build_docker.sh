#!/bin/bash
# Build script for Docker image
# This script exports the Production model and builds the Docker image

set -e  # Exit on error

echo "========================================="
echo "Building LPBF Density Prediction API"
echo "========================================="

# Step 1: Export the Production model
echo ""
echo "Step 1: Exporting Production model..."
python scripts/export_production_model.py

# Step 2: Build Docker image
echo ""
echo "Step 2: Building Docker image..."
docker build -t lpbf-api:latest .

echo ""
echo "========================================="
echo "Build complete!"
echo "========================================="
echo ""
echo "To run the API:"
echo "  docker-compose up"
echo ""
echo "Or run directly:"
echo "  docker run -p 8080:8080 lpbf-api:latest"
echo ""

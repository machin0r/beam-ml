@echo off
REM Build script for Docker image (Windows)
REM This script exports the Production model and builds the Docker image

echo =========================================
echo Building LPBF Density Prediction API
echo =========================================

REM Step 1: Export the Production model
echo.
echo Step 1: Exporting Production model...
python scripts\export_production_model.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to export model
    exit /b %ERRORLEVEL%
)

REM Step 2: Build Docker image
echo.
echo Step 2: Building Docker image...
docker build -t lpbf-api:latest .
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to build Docker image
    exit /b %ERRORLEVEL%
)

echo.
echo =========================================
echo Build complete!
echo =========================================
echo.
echo To run the API:
echo   docker-compose up
echo.
echo Or run directly:
echo   docker run -p 8080:8080 lpbf-api:latest
echo.

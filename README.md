# MMAE Treatment Failure Risk Calculator

Machine learning model predicting treatment failure after middle meningeal artery embolization (MMAE) for subdural hematoma (SDH). This repository accompanies the manuscript submitted to *Neurosurgery Practice*.

**Live Calculator:** [https://bd2708.github.io/mmae-treatment-failure-calculator/](https://bd2708.github.io/mmae-treatment-failure-calculator/)

## Quick Start

### Option 1: Use the Web Calculator (No Installation Required)

1. Open `calculator/index.html` in any modern web browser (Chrome, Firefox, Safari, Edge), or visit the [live calculator](https://bd2708.github.io/mmae-treatment-failure-calculator/).
2. Enter patient parameters: demographics, laboratory values, imaging findings, and medical history.
3. Click **Calculate Risk** to obtain the predicted 90-day treatment failure probability.
4. The calculator classifies patients into Low (<5%), Intermediate (5–15%), or High (>15%) risk categories with corresponding management considerations.

All inference runs locally in the browser using [ONNX Runtime Web](https://onnxruntime.ai/). No data is transmitted to any server.

### Option 2: Run the Model Programmatically

```bash
# Clone the repository
git clone https://github.com/bd2708/mmae-treatment-failure-calculator.git
cd mmae-treatment-failure-calculator

# Install dependencies
pip install -r requirements.txt

# Run verification tests (uses synthetic data only)
python tests/test_calculator.py
```

### Option 3: Retrain on Your Own Data

```bash
# Prepare your institutional data as data/training_data.csv
# (see data/README.md for the required 51-feature format)

# Run the full training and export pipeline
python src/export_model_to_onnx.py

# This will produce:
#   calculator/model.onnx       - Trained ONNX model
#   calculator/model_params.json - Feature names and imputation parameters
```

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Gradient Boosting Classifier (scikit-learn) |
| Estimators | 500 |
| Learning rate | 0.02 |
| Max depth | 6 |
| Subsample | 0.8 |
| Features | 64 (51 base + 13 engineered) |
| Calibration | Isotonic (3-fold CV) |
| Validation AUC | 0.856 |
| Export format | ONNX (opset 12) |

**Outcome definition:** 90-day treatment failure, defined as surgical intervention (craniotomy or burr hole drainage), repeat MMAE, or SDH-related death.

**Preprocessing:** Median imputation for missing values and standard scaling are embedded in the ONNX pipeline. Pass raw, unscaled clinical values.

## Repository Structure

```
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── calculator/
│   ├── index.html               # Self-contained web calculator
│   ├── model.onnx               # Trained model in ONNX format
│   └── model_params.json        # Feature names and imputer medians
├── src/
│   ├── README.md                # Source code documentation
│   └── export_model_to_onnx.py  # Full training and export pipeline
├── config/
│   ├── README.md                # Configuration documentation
│   └── model_config.json        # Model hyperparameters
├── data/
│   └── README.md                # Required data format (no patient data)
└── tests/
    ├── README.md                # Test documentation
    └── test_calculator.py       # Verification with synthetic inputs
```

## Data Availability

Patient-level data from the MESH (Middle meningeal Embolization for Subdural Hematoma) International Registry cannot be shared due to privacy restrictions and multi-institutional data use agreements. The `data/` directory contains a description of the required input format for researchers who wish to apply this model to their own institutional data.

## Requirements

- Python >= 3.8
- scikit-learn >= 1.3.0
- onnxruntime >= 1.16.0
- skl2onnx >= 1.16.0
- pandas >= 2.0.0
- numpy >= 1.24.0

For the web calculator only: any modern browser with WebAssembly support (Chrome 57+, Firefox 53+, Safari 11+, Edge 16+).

## Citation

> Machine Learning Prediction of Treatment Failure After Middle Meningeal Artery Embolization for Subdural Hematoma. *Neurosurgery*. 2026. [Under review]

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

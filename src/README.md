# Source Code

## `export_model_to_onnx.py`

Demonstrates the full machine learning pipeline for training and exporting the treatment failure prediction model:

1. **Data loading** - Reads a CSV file formatted as described in `data/README.md`
2. **Feature engineering** - Creates 13 derived features from 51 base clinical variables (64 total)
3. **Preprocessing** - Median imputation for missing values, standard scaling
4. **Model training** - Gradient Boosting Classifier with isotonic calibration
5. **ONNX export** - Converts the trained model to ONNX format for browser-based inference
6. **Verification** - Confirms ONNX predictions match scikit-learn predictions

### Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Place your institutional data as data/training_data.csv
# (see data/README.md for required format)

# Run the pipeline
python src/export_model_to_onnx.py
```

### Output

- `calculator/model.onnx` - Trained model in ONNX format
- `calculator/model_params.json` - Feature names and imputation parameters

### Note

Patient-level data from the international registry is not included. To reproduce the model, you will need a dataset with the 51 base features described in `data/README.md` and a binary `treatment_failure` outcome variable.

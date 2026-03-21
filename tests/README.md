# Tests

## `test_calculator.py`

Verification tests for the ONNX model and web calculator using synthetic (non-patient) inputs only.

### Tests

| Test | Description |
|------|-------------|
| `test_model_loads` | Verifies the ONNX model file loads and accepts 64-feature input |
| `test_params_file` | Validates `model_params.json` structure (64 feature names and medians) |
| `test_prediction_zeros` | Confirms model produces valid probability from all-zero input |
| `test_prediction_medians` | Confirms model produces valid probability from median-value input |
| `test_batch_prediction` | Verifies batch inference with 5 synthetic samples |

### Usage

```bash
pip install onnxruntime numpy
python tests/test_calculator.py
```

All test inputs are synthetic. No patient data is used or required.

#!/usr/bin/env python3
"""
Verify that the ONNX model loads and produces valid predictions
using synthetic (non-patient) inputs.
"""

import os
import json
import numpy as np

# Requires: pip install onnxruntime
import onnxruntime as ort


def test_model_loads():
    """Test that the ONNX model file loads successfully."""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "calculator", "model.onnx"
    )
    assert os.path.exists(model_path), f"Model file not found: {model_path}"

    session = ort.InferenceSession(model_path)
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    assert len(inputs) == 1, f"Expected 1 input, got {len(inputs)}"
    assert inputs[0].name == "float_input"
    assert inputs[0].shape == [None, 64], f"Expected shape [None, 64], got {inputs[0].shape}"
    print("PASS: Model loads with correct input shape [None, 64]")


def test_params_file():
    """Test that model_params.json is valid and has expected structure."""
    params_path = os.path.join(
        os.path.dirname(__file__), "..", "calculator", "model_params.json"
    )
    assert os.path.exists(params_path), f"Params file not found: {params_path}"

    with open(params_path) as f:
        params = json.load(f)

    assert "feature_names" in params, "Missing feature_names"
    assert "imputer_medians" in params, "Missing imputer_medians"
    assert len(params["feature_names"]) == 64, (
        f"Expected 64 features, got {len(params['feature_names'])}"
    )
    assert len(params["imputer_medians"]) == 64, (
        f"Expected 64 medians, got {len(params['imputer_medians'])}"
    )
    print("PASS: model_params.json has 64 features and 64 medians")


def test_prediction_zeros():
    """Test prediction with all-zero synthetic input."""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "calculator", "model.onnx"
    )
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # All zeros - obviously synthetic, not a real patient
    synthetic_input = np.zeros((1, 64), dtype=np.float32)
    results = session.run(None, {input_name: synthetic_input})

    # results[1] contains class probabilities
    probs = results[1]
    assert len(probs) == 1, f"Expected 1 prediction, got {len(probs)}"

    prob_failure = probs[0][1]
    assert 0.0 <= prob_failure <= 1.0, (
        f"Probability out of range: {prob_failure}"
    )
    print(f"PASS: All-zeros input -> probability = {prob_failure:.4f}")


def test_prediction_medians():
    """Test prediction using imputer medians from model_params.json as input."""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "calculator", "model.onnx"
    )
    params_path = os.path.join(
        os.path.dirname(__file__), "..", "calculator", "model_params.json"
    )

    with open(params_path) as f:
        params = json.load(f)

    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Use imputer medians as synthetic input
    median_input = np.array(params["imputer_medians"], dtype=np.float32).reshape(1, 64)
    results = session.run(None, {input_name: median_input})

    prob_failure = results[1][0][1]
    assert 0.0 <= prob_failure <= 1.0, (
        f"Probability out of range: {prob_failure}"
    )
    print(f"PASS: Median input -> probability = {prob_failure:.4f}")


def test_batch_prediction():
    """Test that model handles batch predictions."""
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "calculator", "model.onnx"
    )
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # Batch of 5 synthetic inputs
    batch_input = np.random.rand(5, 64).astype(np.float32)
    results = session.run(None, {input_name: batch_input})

    probs = results[1]
    assert len(probs) == 5, f"Expected 5 predictions, got {len(probs)}"

    for i, p in enumerate(probs):
        assert 0.0 <= p[1] <= 1.0, f"Sample {i}: probability out of range: {p[1]}"
    print(f"PASS: Batch of 5 synthetic inputs -> all probabilities in [0, 1]")


if __name__ == "__main__":
    print("Running calculator verification tests...\n")
    test_model_loads()
    test_params_file()
    test_prediction_zeros()
    test_prediction_medians()
    test_batch_prediction()
    print("\nAll tests passed.")

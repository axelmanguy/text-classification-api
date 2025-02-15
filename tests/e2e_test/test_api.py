import pytest
import httpx
import pandas as pd
from src.utils.logger import test_logger as logger


API_URL = "http://127.0.0.1:7000"
# Load test dataset (Ensure "test_data.csv" exists)
TEST_DF = pd.read_json("../../data/raw/stages-votes.json").sample(100)
TEST_DF["label"] = TEST_DF["sol"].apply(lambda x: 1 if x == "ok" else 0)  # Convert to binary

@pytest.fixture(scope="module")
def client():
    """Create an HTTP client for API testing."""
    with httpx.Client(base_url=API_URL) as client:
        yield client

def test_health_check(client):
    """Test the /health/ endpoint."""
    logger.info("Testing /health/ endpoint...")
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "model_unavailable"]
    logger.info("[E2E TESTS] /health/ endpoint works correctly.")

def test_metrics(client):
    """Test the /metrics/ endpoint."""
    logger.info("Testing /metrics/ endpoint...")
    response = client.get("/metrics/")
    assert response.status_code == 200
    assert "api_requests_total" in response.text  # Prometheus metric check
    logger.info("[E2E TESTS] /metrics/ endpoint works correctly.")

def test_predict_accuracy(client):
    """Test the /predict/ endpoint and ensure at least 50% accuracy."""
    logger.info("[E2E TESTS] Testing /predict/ accuracy with multiple test samples...")

    total_samples = len(TEST_DF)
    correct_predictions = 0
    errors = []


    for _, row in TEST_DF.iterrows():
        text, expected_label = row["phrase_text"], row["label"]

        response = client.post("/predict/", json={"text": text})
        assert response.status_code == 200, f"Failed for input: {text}"

        data = response.json()
        assert "prediction" in data, f"No prediction in response: {data}"
        assert isinstance(data["prediction"], (str, int)), "Prediction is not a valid type"

        prediction = data["prediction"]

        if prediction == expected_label:
            correct_predictions += 1
        else:
            errors.append(f"Mismatch: '{text}' -> Expected: {expected_label}, Got: {prediction}")

    accuracy = correct_predictions / total_samples
    logger.info(f"[E2E TESTS] Model Accuracy: {accuracy:.2%} ({correct_predictions}/{total_samples})")

    # Log incorrect predictions for debugging
    if errors:
        logger.warning("[E2E TESTS] ! Some incorrect predictions:")
        for err in errors[:5]:  # Show only first 5 errors
            logger.warning(err)

    # Ensure at least 50% accuracy
    assert accuracy > 0.5, f"Model accuracy below threshold: {accuracy:.2%}"

    logger.info("[E2E TESTS] /predict/ endpoint meets the minimum accuracy requirement.")

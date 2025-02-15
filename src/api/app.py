import os

import yaml
import joblib
from pydantic import BaseModel, Field
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from src.utils.logger import inference_logger as logger

from src.utils.config_loader import load_config

config = load_config("deployment.yaml")

# Load model once at startup
MODEL_PATH = os.path.join(config["export"]["src_dir"],config["MODEL_PATH"])
logger.info(f"[FASTAPI] Loading model from: {MODEL_PATH}")
try:
    model_instance = joblib.load(MODEL_PATH)
    logger.info("[FASTAPI] Model loaded successfully.")
except Exception as e:
    logger.error(f"[FASTAPI] Error loading model: {e}")
    model_instance = None  # Prevents crashes if model fails to load

# Initialize FastAPI app
app = FastAPI(
    title="Text Classification API",
    description="FastAPI-based service for text classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


#Define Input schema
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, title="Text to classify")

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter("api_requests_total",
                        "Total number of API requests",
                        ["endpoint"])
REQUEST_ERRORS = Counter("api_request_errors_total",
                         "Total number of API request errors",
                         ["endpoint"])
INFERENCE_LATENCY = Histogram("inference_latency_seconds",
                              "Time spent processing inference requests")

# Enable CORS for cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict/")
def predict(request: TextRequest):
    """Predict the label for the given text input."""
    logger.info(f"[FASTAPI] Received request: {request.dict()}")  # Debug log

    REQUEST_COUNT.labels(endpoint="predict").inc()

    if model_instance is None:
        logger.error("[FASTAPI] Prediction request received, but model is not loaded.")
        REQUEST_ERRORS.labels(endpoint="predict").inc()
        raise HTTPException(status_code=503, detail="Model not available")

    start_time = time.time()
    #Inference pass
    try:
        if isinstance(model_instance, dict) and "tokenizer" in model_instance:
            inputs =  model_instance["tokenizer"](request.text,
                                                  return_tensors="pt",
                                                  padding=True,
                                                  truncation=True,
                                                  max_length=128)
            outputs = model_instance['FT_Camembert'](**inputs)
            prediction = outputs.logits.argmax().item()

        elif hasattr(model_instance, "predict"):
            #joblib sklearn based model
            prediction = int(model_instance.predict([request.text])[0])
        else:
            logger.error("Unsupported model type for prediction.")
            REQUEST_ERRORS.labels(endpoint="predict").inc()
            raise HTTPException(status_code=500, detail="Unsupported model type")

        latency = time.time() - start_time
        INFERENCE_LATENCY.observe(latency)

        logger.info(f"Prediction: '{request.text}' -> {prediction} (Latency: {latency:.4f}s)")
        return {"text": request.text, "prediction": prediction, "latency": latency}

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        REQUEST_ERRORS.labels(endpoint="predict").inc()
        raise HTTPException(status_code=500, detail="Internal Server Error check logs for details")


@app.get("/health/")
def health_check():
    """Health check endpoint to ensure the API is running."""
    status = "healthy" if model_instance is not None else "model_unavailable"
    return {"status": status}

@app.get("/metrics/")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

logger.info("FastAPI server started successfully.")

export PYTHONPATH := $(shell pwd)

# Run model training
train:
	python src/training/train_sklearn_pipeline.py

# Run the FastAPI server
run:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Run all tests
e2e-test:
	pytest test/e2e_test/test_api.py

# Build the Docker image
docker-build:
	docker build -t text-classifier-api .

# Run the Docker container
docker-run:
	docker run --rm -it -p 8000:8000 text-classifier-api

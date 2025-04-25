import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://happily-flowing-pelican.ngrok-free.app")
MODEL_ARTIFACT_NAME = "model-flan-t5-base-original"
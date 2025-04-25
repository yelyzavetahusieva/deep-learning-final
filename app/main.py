from fastapi import FastAPI, HTTPException, Body
import mlflow
from app import models
from app import utils
from app.config import MLFLOW_TRACKING_URI

app = FastAPI(title="MLflow Model Server", description="Serve multiple models using FastAPI", version="1.0")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@app.get("/models", response_model=models.GetAllRunsResponse)
def get_models():
    try:
        return utils.get_all_runs()
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/models/{run_id}", response_model=models.ModelDetails)
def get_model_details(run_id: str):
    """Return model params and metadata for a specific run."""
    try:
        return utils.get_model_metadata(run_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict", response_model=models.PredictionResponse)
def predict(request: models.PredictionRequest):
    try:
        if not request.models:
            runs = utils.list_all_models()
            run_ids = runs["run_id"].tolist()
        else:
            run_ids = request.models
        preds = utils.run_inference(run_ids, request.input_text)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

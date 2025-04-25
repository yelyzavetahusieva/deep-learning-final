from pydantic import BaseModel, RootModel
from typing import List, Optional, Dict

class RunDetails(BaseModel):
    run_id: str
    experiment_name: str
    artifact_uri: Optional[str] = None
    params: Optional[Dict[str, str]] = None
    metrics: Optional[Dict[str, float]] = None
    tags: Optional[Dict[str, str]] = None

class GetAllRunsResponse(RootModel[List["RunDetails"]]):
    pass


class ModelDetails(BaseModel):
    run_id: str
    status: str
    start_time: Optional[int]
    params: Dict[str, str]
    tags: Dict[str, str]
    metrics: Dict[str, float]

class PredictionRequest(BaseModel):
    input_text: str
    models: Optional[List[str]] = None 

class PredictionResponse(BaseModel):
    predictions: Dict[str, str]


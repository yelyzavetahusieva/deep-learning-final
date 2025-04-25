import mlflow
from pathlib import Path
from app.config import MODEL_ARTIFACT_NAME


MLFLOW_DIR = Path("mlruns") 


def get_all_runs():
    experiments = mlflow.search_experiments()
    all_runs = []
    for exp in experiments:
        runs = mlflow.search_runs(exp.experiment_id)
        for _, run in runs.iterrows():
            run_data = {
                "run_id": run.run_id,
                "experiment_name": exp.name,
                "artifact_uri": run.artifact_uri if 'artifact_uri' in run else None,
                "params": run.params if 'params' in run else None,
                "metrics": run.metrics if 'metrics' in run else None,
                "tags": run.tags if 'tags' in run else None,
            }
            all_runs.append({key: value for key, value in run_data.items() if value is not None})
    return all_runs


def get_model_metadata(run_id: str):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "params": run.data.params,
        "tags": run.data.tags,
        "metrics": run.data.metrics,
    }


def run_inference(run_ids, input_text):
    client = mlflow.tracking.MlflowClient()
    results = {}

    for run_id in run_ids:
        model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
        print(model_uri)
        try:
            model = mlflow.transformers.load_model(model_uri)
            print(model_uri)
            output = model([input_text])[0]['generated_text']
            results[run_id] = output
        except Exception as e:
            results[run_id] = f"Error: {str(e)}"

    return results

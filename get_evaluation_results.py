import mlflow
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("databricks")

mlflow_client = mlflow.tracking.MlflowClient()

runs = ["743a7255ab56485687b698cf9ea8f391", "e068367232d640d3831b33bd089e32ce"]

mlflow.set_experiment(experiment_id="2945663160559455")

for run in runs:
    logger.info(f"Getting evaluation results for run: {run}")
    result = mlflow_client.get_run(run)
    logger.info(f"Run results: {result.data.metrics}")
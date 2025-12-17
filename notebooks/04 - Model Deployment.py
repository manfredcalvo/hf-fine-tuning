# Databricks notebook source
# MAGIC %pip install --upgrade \
# MAGIC     transformers>=4.46.0 \
# MAGIC     accelerate>=1.2.0 \
# MAGIC     peft>=0.12.0 \
# MAGIC     trl \
# MAGIC     datasets\
# MAGIC     mlflow \
# MAGIC     databricks-agents
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("CATALOG", "meli_demo")
dbutils.widgets.text("SCHEMA", "default")
dbutils.widgets.text("MODEL_NAME", "deepseek_rag_chat_model")
dbutils.widgets.text("ENDPOINT_NAME", "deepseek_ft_playground_rag")

# COMMAND ----------

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
MODEL_NAME = dbutils.widgets.get("MODEL_NAME")
ENDPOINT_NAME = dbutils.widgets.get("ENDPOINT_NAME")
MODEL_VERSION = dbutils.jobs.taskValues.get(taskKey = "Log_Model", key = "model_version", debugValue = "")

# COMMAND ----------

from mlflow.tracking import MlflowClient

def get_latest_uc_model_version(uc_model_name: str) -> str:
    """
    Return the latest (highest numeric) version of a Unity Catalog or Workspace
    registered model using its full registered name.

    Example UC name: 'main.catalog.schema.model'
    Example workspace name: 'my_registered_model'
    """
    client = MlflowClient()
    # Get all versions registered under this name
    versions = client.search_model_versions(f"name = '{uc_model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model {uc_model_name}")

    latest = max(versions, key=lambda mv: int(mv.version))
    return latest.version

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
MODEL_VERSION = MODEL_VERSION or get_latest_uc_model_version(UC_MODEL_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creación de un Endpoint de Model Serving con el SDK de Databricks
# MAGIC
# MAGIC Crea o actualiza un endpoint de **Model Serving** en Databricks de forma *idempotente*, utilizando el **SDK de Databricks para Python**.  
# MAGIC El siguiente código genera el endpoint `deepseek_ft`, enlazado con el modelo registrado en el **Unity Catalog**, con autoescalado y GPU A10G (`GPU_MEDIUM`).

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)
from datetime import timedelta

def create_or_update_serving_endpoint_and_wait(
    name: str,
    config: EndpointCoreConfigInput
):
    """
    Create or update a Databricks serving endpoint and block until it finishes
    deploying (READY or FAILED), using the SDK's *and_wait helpers.

    Parameters
    ----------
    name : str
        Serving endpoint name.
    config : EndpointCoreConfigInput
        Config of the endpoint with entities to be served.

    Returns
    -------
    EndpointCoreInfo
        Final endpoint info once deployment succeeds.

    Raises
    ------
    RuntimeError
        If deployment ends in FAILED.
    """
    w = WorkspaceClient()

    # Decide whether to create or update
    try:
        _ = w.serving_endpoints.get(name)
        exists = True
    except Exception:
        exists = False

    if exists:
        served_entities = config.served_entities
        # Update config and wait until deployment completes
        ep = w.serving_endpoints.update_config_and_wait(
            name=name,
            served_entities=served_entities,
            timeout=timedelta(minutes=30),
        )
    else:
        # Create new endpoint and wait until deployment completes
        ep = w.serving_endpoints.create_and_wait(
            name=name,
            config=config,
            timeout=timedelta(minutes=30)
        )

    # Optional: extra safety check on final state
    if ep.state and ep.state.ready == "FAILED":
        raise RuntimeError(
            f"Serving endpoint {name} deployment failed: {ep.state.message}"
        )

    return ep

# COMMAND ----------

from databricks.sdk.service.serving import (ServedEntityInput, 
                                            ServingModelWorkloadType, TrafficConfig, Route, EndpointCoreConfigInput)

SERVED_ENTITY_NAME = F"{MODEL_NAME}-{MODEL_VERSION}"

served = [
    ServedEntityInput(
        name=SERVED_ENTITY_NAME,
        entity_name=UC_MODEL_NAME,
        entity_version=MODEL_VERSION,
        workload_type=ServingModelWorkloadType.GPU_MEDIUM,
        workload_size="Small",
        scale_to_zero_enabled=True
    )
]

traffic_config = TrafficConfig(routes=[Route(served_entity_name=SERVED_ENTITY_NAME, traffic_percentage=100)])

endpoint_config = EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=served, traffic_config=traffic_config)

create_or_update_serving_endpoint_and_wait(ENDPOINT_NAME, endpoint_config)


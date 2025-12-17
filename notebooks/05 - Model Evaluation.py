# Databricks notebook source
# MAGIC %md
# MAGIC # Evaluación de Modelos con MLflow
# MAGIC
# MAGIC Este notebook evalúa múltiples modelos de lenguaje utilizando **MLflow GenAI Evaluate**, comparando sus respuestas con métricas personalizadas y predefinidas.
# MAGIC
# MAGIC **Proceso:**
# MAGIC 1. Cargar dataset de evaluación desde Unity Catalog
# MAGIC 2. Configurar funciones de predicción para cada endpoint
# MAGIC 3. Definir scorers (Guidelines y custom)
# MAGIC 4. Ejecutar evaluaciones con nested runs
# MAGIC 5. Comparar resultados en MLflow UI
# MAGIC
# MAGIC **Herramientas:**
# MAGIC - MLflow GenAI Evaluate
# MAGIC - Scorers personalizados (Guidelines)
# MAGIC - Métricas predefinidas (Safety, Relevance, Correctness)

# COMMAND ----------

# Instalar databricks-agents para evaluación de modelos
%pip install -U databricks-agents
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuración

# COMMAND ----------

dbutils.widgets.text("CATALOG", "meli_demo")
dbutils.widgets.text("SCHEMA", "default")
dbutils.widgets.multiselect("MODEL_LIST", "deepseek_ft_playground_rag", ["deepseek_ft_playground_rag", "databricks-llama-4-maverick"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cargar Dataset de Evaluación

# COMMAND ----------

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
MODEL_LIST = dbutils.widgets.get("MODEL_LIST").split(",")

evaluation_dataset_table = f"{CATALOG}.{SCHEMA}.chat_completion_evaluation_dataset"
chat_eval_df = spark.table(evaluation_dataset_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparar Datos para Evaluación
# MAGIC
# MAGIC Convertir el dataset al formato esperado por MLflow Evaluate:
# MAGIC - **inputs**: Mensajes de entrada (system + user)
# MAGIC - **expectations**: Respuesta esperada del assistant

# COMMAND ----------

import pandas as pd

eval_pandas = chat_eval_df.toPandas()

evals = []
for _, row in eval_pandas.iterrows():
    eval_record = {
         "inputs": {"inputs": row["messages"][:2]},
         "expectations": {"expected_response": row["messages"][2].get("content")}
    }
    evals.append(eval_record)

eval_df = pd.DataFrame.from_records(evals)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Función de Predicción para Endpoints
# MAGIC
# MAGIC Crea un wrapper que adapta el formato según el tipo de modelo:
# MAGIC - **DeepSeek**: Formato de texto plano
# MAGIC - **Foundation Models**: Formato de mensajes estructurado

# COMMAND ----------

from mlflow.deployments import get_deploy_client
from databricks.sdk import WorkspaceClient

workspace_client = WorkspaceClient()


def _get_endpoint_task_type(workspace_client, endpoint_name: str) -> str:
    """Get the task type of a serving endpoint."""
    try:
        ep = workspace_client.serving_endpoints.get(endpoint_name)
        return ep.task if ep.task else "llm/v1/chat"
    except Exception:
        return "llm/v1/chat"
        
def create_predict_fn(endpoint):
    """Crea función de predicción para un endpoint específico."""
    def predict_fn(inputs):

        task_type = _get_endpoint_task_type(workspace_client, endpoint)
        if task_type == "llm/v1/chat":
            input_data = {"messages": inputs.tolist()}
        else:
            input_data = {"input": inputs.tolist()}

        # Hacer predicción
        client = get_deploy_client("databricks")
        response = client.predict(endpoint=endpoint, inputs=input_data)

        # Extraer respuesta según formato
        if response.get("choices"):
            response = response["choices"][0]["message"]["content"]
        else:
            response = response['output'][0]['content'][0]['text']
        return response
    
    return predict_fn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Definir Scorers Personalizados
# MAGIC
# MAGIC **Guidelines**: Evalúan criterios específicos usando modelos LLM como jueces

# COMMAND ----------

from mlflow.genai.scorers import Guidelines

english = Guidelines(
    name="english",
    guidelines=["La respuesta debe estar en inglés."]
)

clarity = Guidelines(
    name="clarity",
    guidelines=["La respuesta debe ser clara, coherente y concisa."],
    model="databricks:/databricks-gpt-oss-120b",
)

rhyme = Guidelines(
    name="rhyme",
    guidelines=["La respuesta debe rimar."],
)

# COMMAND ----------

from mlflow.genai import scorer
from mlflow.entities import Feedback

@scorer(name="count_Y_scorer")
def count_Y_scorer(inputs, outputs, trace):
    """
    Scorer personalizado: cuenta letras 'Y' en la respuesta.
    Ejemplo de métrica cuantitativa custom.
    """
    # Extraer texto de la respuesta
    if isinstance(outputs, str):
        text = outputs
    elif isinstance(outputs, dict):
        for k in ("answer", "response", "outputs", "text", "content"):
            if k in outputs:
                v = outputs[k]
                text = v if isinstance(v, str) else ""
                break
        else:
            text = ""
    else:
        text = ""

    count = sum(1 for ch in text if ch.lower() == "y")

    return [
        Feedback(
            name="count_Y",
            value=count,
            rationale=f"Cantidad de 'Y' encontradas: {count}"
        )
    ]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ejecutar Evaluación

# COMMAND ----------

MODEL_LIST

_get_endpoint_task_type(workspace_client, MODEL_LIST[0])

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import RelevanceToQuery, Safety, Correctness

# Crear parent run para agrupar evaluaciones
with mlflow.start_run(run_name="multi_model_evaluation") as parent_run:
    
    # Evaluar cada endpoint
    for endpoint in MODEL_LIST:
        print(f"Evaluando endpoint: {endpoint}")
        
        # Nested run para cada modelo
        with mlflow.start_run(run_name=f"evaluation_{endpoint}", nested=True):
            mlflow.genai.evaluate(
                data=eval_df,
                predict_fn=create_predict_fn(endpoint),
                scorers=[
                    Safety(),
                    RelevanceToQuery(),
                    Correctness(),
                    english, 
                    clarity, 
                    rhyme,
                    count_Y_scorer
                ],
            )
            mlflow.log_param("endpoint", endpoint)
    
    print(f"\nEvaluación completada. Revisa los resultados en MLflow UI.")
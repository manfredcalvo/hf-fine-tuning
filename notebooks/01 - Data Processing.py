# Databricks notebook source
# MAGIC %md
# MAGIC # Install libraries

# COMMAND ----------

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

# COMMAND ----------

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")

# COMMAND ----------

train_data_table_name = "chat_completion_training_dataset"
eval_data_table_name = "chat_completion_evaluation_dataset"

train_data_table_name_ft = "chat_completion_training_dataset_ft"
eval_data_table_name_ft = "chat_completion_evaluation_dataset_ft"


# COMMAND ----------

# MAGIC %md
# MAGIC ## Cargar Training Dataset
# MAGIC
# MAGIC Carga el chat completion dataset generado desde el notebook de RAG fine-tuning. Este dataset contiene mensajes conversacionales con system prompts, user queries con RAG context y las assistant responses esperadas.

# COMMAND ----------

# Cargar el training dataset desde la Delta table

training_dataset_table = f"{CATALOG}.{SCHEMA}.chat_completion_training_dataset"
evaluation_dataset_table = f"{CATALOG}.{SCHEMA}.chat_completion_evaluation_dataset"

chat_training_df = spark.table(training_dataset_table)
chat_eval_df = spark.table(evaluation_dataset_table)

print(f"Muestras de entrenamiento: {chat_training_df.count()}")
print(f"Muestras de evaluación: {chat_eval_df.count()}")

display(chat_training_df.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparación de Datos para SFTTrainer
# MAGIC
# MAGIC Convierte el Spark DataFrame al formato de Hugging Face Dataset. SFTTrainer espera un formato específico donde formatearemos los mensajes en texto usando la plantilla de chat de DeepSeek.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
import pandas as pd
from typing import List, Dict

def format_chat_template(messages: List) -> str:
    """
    Formatea los mensajes en el template de DeepSeek.
    DeepSeek utiliza el formato: System: ... \n\nUser: ... \n\nAssistant: ...
    """
    formatted_text = ""
    for message in messages:
        if isinstance(message, dict):
            role = message["role"]
            content = message["content"]
        elif isinstance(message, list) and len(message) >= 2:
            role = message[0]
            content = message[1]
        else:
            continue
            
        if role == "system":
            formatted_text += f"System: {content}\n\n"
        elif role == "user":
            formatted_text += f"User: {content}\n\n"
        elif role == "assistant":
            formatted_text += f"Assistant: {content}"
    return formatted_text
  
# Define the pandas UDF using a decorator and type hints
@pandas_udf(returnType=StringType())
def transform_data(s: pd.Series) -> pd.Series:
  results = []
  for val in s:
    result = format_chat_template(val)
    results.append(result)
  return pd.Series(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformar arreglo de mensajes a texto formateado para fine-tuning

# COMMAND ----------

chat_training_df_ft = chat_training_df.withColumn("text", transform_data(chat_training_df['messages']))
chat_eval_df_ft = chat_eval_df.withColumn("text", transform_data(chat_eval_df['messages']))

chat_eval_df_ft.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.{eval_data_table_name_ft}")

chat_training_df_ft.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.{train_data_table_name_ft}")


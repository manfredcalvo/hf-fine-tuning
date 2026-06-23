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
dbutils.widgets.text("VOLUME", "datasets")
dbutils.widgets.text("BASE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

# COMMAND ----------

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
VOLUME = dbutils.widgets.get("VOLUME")
BASE_MODEL = dbutils.widgets.get("BASE_MODEL")

# Construir el path del volumen a partir del catálogo y esquema
DATA_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# COMMAND ----------

train_data_table_name = "chat_completion_training_dataset"
eval_data_table_name = "chat_completion_evaluation_dataset"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cargar CSV desde Volumen a Delta Tables
# MAGIC
# MAGIC Lee los archivos CSV de entrenamiento y evaluación desde el volumen de Unity Catalog,
# MAGIC parsea la columna `messages` (almacenada como JSON) y escribe los resultados en tablas Delta crudas.

# COMMAND ----------

from pyspark.sql.functions import from_json, col
from pyspark.sql.types import ArrayType, StructType, StructField, StringType

# Esquema del arreglo de mensajes
message_schema = ArrayType(
    StructType([
        StructField("role", StringType(), True),
        StructField("content", StringType(), True),
    ])
)

# Leer los CSV desde el volumen
train_csv_df = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("escape", '"')
    .csv(f"{DATA_VOLUME_PATH}/chat_completion_training_dataset.csv")
)

eval_csv_df = (
    spark.read
    .option("header", True)
    .option("multiLine", True)
    .option("escape", '"')
    .csv(f"{DATA_VOLUME_PATH}/chat_completion_evaluation_dataset.csv")
)

# Parsear la columna messages de string JSON a arreglo de structs
chat_training_df = train_csv_df.withColumn("messages", from_json(col("messages"), message_schema))
chat_eval_df = eval_csv_df.withColumn("messages", from_json(col("messages"), message_schema))

# Guardar como tablas Delta
chat_training_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.{train_data_table_name}")
chat_eval_df.write.mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}.{eval_data_table_name}")

print(f"Muestras de entrenamiento cargadas: {chat_training_df.count()}")
print(f"Muestras de evaluación cargadas: {chat_eval_df.count()}")

display(chat_training_df.limit(3))

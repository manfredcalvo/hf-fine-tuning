# Databricks notebook source
# MAGIC %md
# MAGIC ## Installing libraries

# COMMAND ----------

# MAGIC %pip install --upgrade \
# MAGIC     transformers>=4.46.0 \
# MAGIC     accelerate>=1.2.0 \
# MAGIC     peft>=0.12.0 \
# MAGIC     trl \
# MAGIC     datasets\
# MAGIC     mlflow \
# MAGIC     databricks-agents \
# MAGIC     hf_transfer
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("CATALOG", "meli_demo")
dbutils.widgets.text("SCHEMA", "default")
dbutils.widgets.text("MODEL_NAME", "deepseek_rag_chat_model")
dbutils.widgets.text("MODEL_PATH", "/Volumes/lucas_catalog/default/models/deep_seek_ft_model_1/")

# COMMAND ----------

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
MODEL_NAME = dbutils.widgets.get("MODEL_NAME")
MODEL_PATH = dbutils.widgets.get("MODEL_PATH")

# COMMAND ----------

base_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
output_dir = "/tmp/deepseek_finetuned"
adapter_dir = "/tmp/deepseek_lora_adapters"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cargar el modelo en Float16
# MAGIC
# MAGIC Carga el modelo base en precisión float16 para entrenamiento. Este enfoque es óptimo al desplegar con MLflow transformers flavor, ya que evita la necesidad de recargar y convertir el modelo para inferencia.
# MAGIC
# MAGIC **Requisitos de memoria:**
# MAGIC - Entrenamiento: ~14-16GB VRAM (requiere A10 24GB o superior)
# MAGIC - Inferencia: ~16GB VRAM (igual que entrenamiento, sin conversión)

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token for batch training
tokenizer.padding_side = "right"  # Pad on the right side for causal LM

# Load model in float16 (no quantization)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    dtype=torch.float16,  # Use float16 instead of quantization
    device_map="auto",  # Automatically distribute model across available GPUs
    trust_remote_code=True,
    low_cpu_mem_usage=True  # Optimize CPU memory during loading
)

# Prepare model for training
model.config.use_cache = False  # Disable KV cache for training
model.config.pretraining_tp = 1  # Set tensor parallelism to 1

print(f"Model loaded: {base_model}")
print(f"Model dtype: {model.dtype}")
print(f"Model memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
print(f"Expected VRAM usage: ~14-16GB during training")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load dataset

# COMMAND ----------

train_data_table_name_ft = "chat_completion_training_dataset_ft"
eval_data_table_name_ft = "chat_completion_evaluation_dataset_ft"

# COMMAND ----------

import datasets
from typing import List, Dict

chat_training_df_ft = spark.read.table(f"{CATALOG}.{SCHEMA}.{train_data_table_name_ft}")
chat_eval_df_ft = spark.read.table(f"{CATALOG}.{SCHEMA}.{eval_data_table_name_ft}")

train_data = chat_training_df_ft.collect()
eval_data = chat_eval_df_ft.collect()

train_records = [row.asDict() for row in train_data]
eval_records = [row.asDict() for row in eval_data]

train_dataset = datasets.Dataset.from_list(train_records)
eval_dataset = datasets.Dataset.from_list(eval_records)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configurar LoRA para Fine-Tuning Eficiente en Parámetros
# MAGIC
# MAGIC LoRA (Low-Rank Adaptation) nos permite ajustar solo una pequeña cantidad de parámetros adicionales, haciendo el entrenamiento mucho más eficiente y reduciendo los requisitos de memoria.

# COMMAND ----------

from peft import LoraConfig, get_peft_model

if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# Configuración de LoRA
lora_config = LoraConfig(
    r=16,  # Rango de las matrices de actualización
    lora_alpha=32,  # Factor de escalado para LoRA
    target_modules=[
        "q_proj",      # Query projection
        "k_proj",      # Key projection
        "v_proj",      # Value projection
        "o_proj",      # Output projection
        "gate_proj",   # Gate projection
        "up_proj",     # Up projection
        "down_proj"    # Down projection
    ],  # Apuntar a todas las capas de atención y MLP
    lora_dropout=0.05,  # Dropout para capas LoRA
    bias="none",  # No entrenar parámetros de bias
    task_type="CAUSAL_LM"  # Tipo de tarea para lenguaje causal
)

# Aplicar LoRA al modelo
model = get_peft_model(model, lora_config)

# Imprimir parámetros entrenables
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parámetros entrenables: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print(f"Parámetros totales: {total_params:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configurar SFTTrainer para Fine-Tuning Supervisado
# MAGIC
# MAGIC Configura los argumentos de entrenamiento e inicializa el SFTTrainer (Supervised Fine-Tuning Trainer) de la librería TRL, optimizado para instruction fine-tuning.

# COMMAND ----------

from trl import SFTConfig, SFTTrainer

# Configuración correcta usando SFTConfig
training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=1,  # Número de epochs de entrenamiento
    per_device_train_batch_size=2,  # Batch size por dispositivo GPU
    per_device_eval_batch_size=2,  # Batch size de evaluación por dispositivo GPU
    gradient_accumulation_steps=8,  # Acumulación de gradientes (batch efectivo = 16)
    gradient_checkpointing=True,  # Activar para reducir uso de memoria
    optim="adamw_torch",  # Optimizador AdamW de PyTorch
    learning_rate=2e-4,  # Learning rate para LoRA
    lr_scheduler_type="cosine",  # Scheduler de learning rate tipo coseno
    warmup_ratio=0.03,  # Proporción de warmup
    logging_steps=10,  # Frecuencia de logging
    eval_strategy="steps",  # Estrategia de evaluación por pasos
    eval_steps=500,  # Evaluar cada 500 pasos
    save_strategy="steps",  # Estrategia de guardado por pasos
    save_steps=500,  # Guardar checkpoint cada 500 pasos
    save_total_limit=2,  # Limitar a 2 checkpoints
    fp16=False,  # No usar fp16, usar bfloat16
    bf16=True,  # Usar bfloat16 para mayor estabilidad
    max_grad_norm=0.3,  # Clipping de gradiente
    max_steps=-1,  # Entrenar por número de epochs
    report_to="mlflow",  # Reportar a MLflow
    seed=42,
    packing=True,      # Reemplaza group_by_length, empaqueta secuencias para mayor eficiencia
    max_length=512,    # Longitud máxima de secuencia, requerido al usar packing
)

# Inicializar SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    args=training_args,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entrenamiento del Modelo
# MAGIC
# MAGIC Inicia el proceso de fine-tuning. Esto entrenará los LoRA adapters sobre el modelo DeepSeek quantizado usando el chat completion dataset. El progreso del entrenamiento y las métricas serán registradas en MLflow.

# COMMAND ----------

import mlflow

# Iniciar MLflow run para seguimiento del entrenamiento
with mlflow.start_run() as run:
    # Entrenar el modelo
    print("Iniciando entrenamiento...")
    trainer.train()
    
    # Guardar los LoRA adapters fine-tuned
    print(f"Guardando LoRA adapters en {adapter_dir}")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    
    # Registrar parámetros en MLflow
    mlflow.log_param("base_model", base_model)
    mlflow.log_param("lora_r", lora_config.r)
    mlflow.log_param("lora_alpha", lora_config.lora_alpha)
    mlflow.log_param("learning_rate", training_args.learning_rate)
    mlflow.log_param("num_epochs", training_args.num_train_epochs)
    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("effective_batch_size", training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    
    eval_results = trainer.evaluate()

    for key, value in eval_results.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(f"eval_{key}", value)
    print(f"Entrenamiento finalizado. MLflow run ID: {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prueba de Inferencia
# MAGIC
# MAGIC Evalúa el modelo fine-tuned con una pregunta de ejemplo del evaluation dataset para verificar su desempeño en RAG-based question answering.

# COMMAND ----------

import torch

# Obtener una muestra del conjunto de evaluación (sin la respuesta del assistant)
sample_text = eval_dataset[0]["text"]
# Extraer solo las partes de system + user (eliminar la respuesta del assistant para la prueba)
input_text = sample_text.split("Assistant:")[0].strip() + "\n\nAssistant:"

print("=" * 80)
print("ENTRADA:")
print("=" * 80)
print(input_text)
print("\n" + "=" * 80)
print("RESPUESTA DEL MODELO:")
print("=" * 80)

# Tokenizar la entrada y mover al dispositivo del modelo
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# Generar respuesta usando model.generate()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )

# Decodificar la respuesta completa
full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extraer solo la parte generada (después del prompt de entrada)
if full_response.startswith(input_text):
    generated_response = full_response[len(input_text):].strip()
else:
    # Alternativa: intentar extraer después de "Assistant:"
    parts = full_response.split("Assistant:", 1)
    generated_response = parts[1].strip() if len(parts) > 1 else full_response

print(generated_response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guardar el modelo fine-tuned en un volumen

# COMMAND ----------

merged_model = model.merge_and_unload()
merged_model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)
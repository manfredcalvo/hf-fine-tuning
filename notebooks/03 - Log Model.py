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
dbutils.widgets.text("train_data_table_name_ft", "chat_completion_training_dataset_ft")
dbutils.widgets.text("test_data_table_name_ft", "chat_completion_eval_dataset_ft")

# COMMAND ----------

CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
MODEL_NAME = dbutils.widgets.get("MODEL_NAME")
MODEL_PATH = dbutils.widgets.get("MODEL_PATH")
train_data_table_name_ft = dbutils.widgets.get("train_data_table_name_ft")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Guardar el modelo en MLflow usando pyfunc Flavor
# MAGIC
# MAGIC 1. Crea un pipeline de HuggingFace con el modelo fusionado  
# MAGIC 2. Regístralo directamente con `mlflow.pyfunc.log_model()`
# MAGIC
# MAGIC MLflow gestiona automáticamente:
# MAGIC - Serialización/deserialización del modelo
# MAGIC - Conservación del tokenizer
# MAGIC - Gestión de dependencias
# MAGIC - Procesamiento de entrada/salida
# MAGIC - Soporte para Spark UDF

# COMMAND ----------

# MAGIC %%writefile agent.py
# MAGIC import mlflow
# MAGIC import uuid
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC     to_chat_completions_input,
# MAGIC )
# MAGIC from transformers import pipeline as hf_pipeline, AutoModelForCausalLM, AutoTokenizer
# MAGIC import torch
# MAGIC from transformers import TextIteratorStreamer 
# MAGIC from threading import Thread
# MAGIC from typing import Generator
# MAGIC
# MAGIC class FineTuneChatAgent(ResponsesAgent):
# MAGIC     def __init__(self):
# MAGIC         self.pipeline = None
# MAGIC         self.system_prompt = "You are a highly knowledgeable and professional Databricks Support Agent. Your goal is to assist users with their questions and issues related to Databricks. Answer questions as precisely and accurately as possible, providing clear and concise information. If you do not know the answer, respond with \"I don't know.\" Be polite and professional in your responses. Provide accurate and detailed information related to Databricks. If the question is unclear, ask for clarification."
# MAGIC
# MAGIC     def load_context(self, context):
# MAGIC
# MAGIC         device = "cuda" if torch.cuda.is_available() else "cpu"
# MAGIC         
# MAGIC         model_path = context.artifacts["model_path"]
# MAGIC         
# MAGIC         model = AutoModelForCausalLM.from_pretrained(
# MAGIC             model_path,
# MAGIC             dtype=torch.float16,
# MAGIC             device_map="auto" if device == "cuda" else None,
# MAGIC             trust_remote_code=True,
# MAGIC             low_cpu_mem_usage=True
# MAGIC         )
# MAGIC         
# MAGIC         tokenizer = AutoTokenizer.from_pretrained(
# MAGIC             model_path,
# MAGIC             trust_remote_code=True
# MAGIC         )
# MAGIC         
# MAGIC         self.pipeline = hf_pipeline(
# MAGIC             "text-generation",
# MAGIC             model=model,
# MAGIC             tokenizer=tokenizer,
# MAGIC         )
# MAGIC
# MAGIC     def format_chat_template(self, messages) -> str:
# MAGIC         """
# MAGIC         Formatea los mensajes en el template de DeepSeek.
# MAGIC         DeepSeek utiliza el formato: System: ... \n\nUser: ... \n\nAssistant: ...
# MAGIC         """
# MAGIC         formatted_text = ""
# MAGIC         for message in messages:
# MAGIC             if isinstance(message, dict):
# MAGIC                 role = message["role"]
# MAGIC                 content = message["content"]
# MAGIC             elif isinstance(message, list) and len(message) >= 2:
# MAGIC                 role = message[0]
# MAGIC                 content = message[1]
# MAGIC             else:
# MAGIC                 continue
# MAGIC                 
# MAGIC             if role == "system":
# MAGIC                 formatted_text += f"System: {content}\n\n"
# MAGIC             elif role == "user":
# MAGIC                 formatted_text += f"User: {content}\n\n"
# MAGIC             elif role == "assistant":
# MAGIC                 formatted_text += f"Assistant: {content}"
# MAGIC         return formatted_text + "\n\nAssistant:"
# MAGIC
# MAGIC     def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)
# MAGIC
# MAGIC     def generate_output(self, messages_with_system):
# MAGIC
# MAGIC       model, tokenizer = self.pipeline.model, self.pipeline.tokenizer
# MAGIC       
# MAGIC       prompt = self.format_chat_template(messages_with_system)
# MAGIC
# MAGIC       inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# MAGIC       streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, decode_kwargs={"clean_up_tokenization_spaces": False})
# MAGIC
# MAGIC       # Set up generation arguments including max tokens and streamer
# MAGIC       generation_args = {
# MAGIC           "max_new_tokens": 512,
# MAGIC           "do_sample": True,
# MAGIC           "temperature": 0.7,
# MAGIC           "top_p": 0.95,
# MAGIC           "repetition_penalty": 1.15,
# MAGIC           "pad_token_id": tokenizer.eos_token_id,
# MAGIC           "streamer": streamer,
# MAGIC           **inputs
# MAGIC       }
# MAGIC       item_id = str(uuid.uuid4())
# MAGIC
# MAGIC       thread = Thread(
# MAGIC           target=model.generate,
# MAGIC           kwargs=generation_args,
# MAGIC       )
# MAGIC
# MAGIC       thread.start()
# MAGIC       acc_text = ""
# MAGIC       
# MAGIC       for text_token in streamer:
# MAGIC         acc_text += text_token
# MAGIC         yield ResponsesAgentStreamEvent(
# MAGIC               **self.create_text_delta(delta=text_token, item_id=item_id),
# MAGIC           )
# MAGIC       
# MAGIC       thread.join()
# MAGIC
# MAGIC       yield ResponsesAgentStreamEvent(
# MAGIC             type="response.output_item.done",
# MAGIC             item=self.create_text_output_item(
# MAGIC                 text=acc_text,
# MAGIC                 id=item_id,
# MAGIC             ),
# MAGIC         )
# MAGIC
# MAGIC
# MAGIC     def predict_stream(
# MAGIC         self, request: ResponsesAgentRequest
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         messages = to_chat_completions_input([i.model_dump() for i in request.input])
# MAGIC         messages_with_system = [{"role": "system", "content": self.system_prompt}] + messages
# MAGIC         for chunk in self.generate_output(messages_with_system):
# MAGIC           yield chunk
# MAGIC
# MAGIC AGENT = FineTuneChatAgent()
# MAGIC mlflow.models.set_model(AGENT)

# COMMAND ----------

import mlflow
import transformers
import accelerate

model_name_responses = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

dataset_info = mlflow.data.from_spark(
    spark.table(f"{CATALOG}.{SCHEMA}.{train_data_table_name_ft}"),
    table_name=f"{CATALOG}.{SCHEMA}.{train_data_table_name_ft}",
    version="1.0"
)

with mlflow.start_run():
    
    mlflow.log_input(dataset_info, context="training")

    artifacts = {
        "model_path": MODEL_PATH
    }
        
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="agent.py",
        artifacts=artifacts,
        pip_requirements=[
                        f"transformers=={transformers.__version__}",
                        f"torch==2.7.0",
                        f"accelerate=={accelerate.__version__}",
                        "sentencepiece",
                        "protobuf",
                    ],
        registered_model_name=model_name_responses,

    )

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(logged_agent_info.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the model and run inference on it

# COMMAND ----------

loaded_model.predict(    {
        "input": [{"role": "user", "content": "what is mlflow?"}],
        "context": {"conversation_id": "123", "user_id": "456"},
    })

# COMMAND ----------

dbutils.jobs.taskValues.set("model_version", logged_agent_info.registered_model_version)
# HuggingFace LLM Fine-Tuning on Databricks

A Databricks Asset Bundle that orchestrates an end-to-end fine-tuning pipeline for `DeepSeek-R1-Distill-Llama-8B` — from data preparation to a deployed, evaluated Model Serving endpoint.

## Pipeline Overview

```
Data Processing → Fine Tuning → Log Model → Deploy → Evaluate
```

Each stage is a Databricks notebook run as a sequential job task. GPU-intensive tasks (`Fine Tuning` and `Log Model`) are queued to a `GPU_SERVERLESS` queue.

## Notebooks

| Notebook | Description |
|---|---|
| `01 - Data Processing` | Reads chat completion Delta tables and formats messages into DeepSeek's plain-text template |
| `02 - Fine Tuning Model` | Loads the base model in float16, applies LoRA adapters, trains with SFTTrainer, and saves the merged model to a Volume |
| `03 - Log Model` | Wraps the fine-tuned model as an MLflow `pyfunc` ResponsesAgent with streaming support and registers it in Unity Catalog |
| `04 - Model Deployment` | Creates or updates a Model Serving endpoint (GPU_MEDIUM / A10G, scale-to-zero enabled) |
| `05 - Model Evaluation` | Evaluates one or more endpoints side-by-side using MLflow GenAI Evaluate with Safety, Relevance, Correctness, and custom scorers |

## Requirements

- Databricks CLI with Asset Bundles support (`databricks bundle`)
- Access to `e2-demo-field-eng.cloud.databricks.com`
- A Unity Catalog volume to store the fine-tuned model weights
- Training and evaluation datasets pre-loaded as Delta tables:
  - `{catalog}.{schema}.chat_completion_training_dataset`
  - `{catalog}.{schema}.chat_completion_evaluation_dataset`

## Configuration

Bundle variables are defined in `databricks.yml` and can be overridden per target:

| Variable | Default | Description |
|---|---|---|
| `catalog` | `meli_demo` | Unity Catalog catalog name |
| `schema` | `default` | Schema name |
| `model_name` | `deepseek_rag_chat_model` | Registered model name in UC |
| `model_path` | `/Volumes/lucas_catalog/default/models/deep_seek_ft_model_1/` | Volume path to save the merged model |
| `endpoint_name` | `deepseek_ft_playground_rag` | Model Serving endpoint name |
| `model_list` | `deepseek_ft_playground_rag,databricks-llama-4-maverick` | Comma-separated endpoints to evaluate |

## Deploying the Bundle

```bash
# Validate the bundle configuration
databricks bundle validate

# Deploy to the dev target (default)
databricks bundle deploy

# Deploy to a specific target
databricks bundle deploy --target dev
```

## Running the Pipeline

**Via the Databricks CLI:**
```bash
databricks bundle run FineTuning_Pipeline
```

**Via the helper script** (triggers the job and polls until completion):
```bash
python call_job.py
```

The script times out after 3 hours. Update `job_id` and `job_parameters` in `call_job.py` as needed.

## Fetching Evaluation Results

To retrieve MLflow evaluation metrics for specific runs locally:

```bash
python get_evaluation_results.py
```

Update the `runs` list in the script with the MLflow run IDs you want to inspect.

## Fine-Tuning Details

- **Base model:** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **Method:** LoRA (rank 16, alpha 32) targeting all attention and MLP projection layers
- **Trainable parameters:** ~1–2% of total
- **Training precision:** bfloat16
- **Hardware:** A10 24GB GPU or higher (GPU_SERVERLESS queue)
- **Estimated VRAM:** ~14–16 GB during training

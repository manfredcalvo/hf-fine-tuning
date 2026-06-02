import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs as jobs_svc

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_job_and_wait(
    job_id: int,
    job_parameters: dict | None = None,
    timeout_sec: int = 3600,
    poll_interval_sec: int = 10,
) -> jobs_svc.Run:
    """
    Trigger a Databricks job and wait until it reaches a terminal state.

    Parameters
    ----------
    job_id : int
        The ID of the existing Databricks job.
    job_parameters : dict, optional
        Job parameters to override (for jobs that use parameters).
    timeout_sec : int
        Maximum time to wait for the run to complete.
    poll_interval_sec : int
        Seconds between status polls.

    Returns
    -------
    jobs.Run
        The final run object.

    Raises
    ------
    TimeoutError
        If the run does not finish within timeout_sec.
    RuntimeError
        If the run ends in a FAILED or INTERNAL_ERROR state.
    """
    w = WorkspaceClient()

    # Start the run
    run_now_resp = w.jobs.run_now(
        job_id=job_id,
        job_parameters=job_parameters,
    )
    run_id = run_now_resp.run_id

    deadline = time.time() + timeout_sec
    last_state = None
    logger.info(f"Running job {job_id}")
    while time.time() < deadline:
        run = w.jobs.get_run(run_id=run_id)
        state = run.state
        life_cycle = state.life_cycle_state
        result = state.result_state
        logger.info(f"Checking status of job run: {run_id}")
        logger.info(f"Life-cycle state: {life_cycle}")
        logger.info(f"Result state: {result}")
        # Terminal lifecycle: TERMINATED, SKIPPED, or INTERNAL_ERROR
        if life_cycle.value in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            if result.value in ("SUCCESS", None):
                return run
            raise RuntimeError(
                f"Job run {run_id} failed: "
                f"life_cycle_state={life_cycle}, result_state={result}, "
                f"state_message={state.state_message}"
            )

        last_state = (life_cycle, result)
        time.sleep(poll_interval_sec)

    raise TimeoutError(
        f"Timed out waiting for job {job_id} run {run_id} to finish; "
        f"last_state={last_state}"
    )


job_parameters = {"CATALOG": "meli_demo", "ENDPOINT_NAME": "deepseek_ft_playground_rag",
              "MODEL_LIST": "deepseek_ft_playground_rag,databricks-llama-4-maverick",
              "MODEL_NAME": "deepseek_rag_chat_model", "MODEL_PATH": "/Volumes/lucas_catalog/default/models/deep_seek_ft_model_1/",
              "SCHEMA": "default"}

run = run_job_and_wait(
    job_id=904873261067620,
    job_parameters=job_parameters,
    timeout_sec=10800,
)
print("Run finished with state:", run.state)
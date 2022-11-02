import pickle
import logging
from .db import get_unique_id as get_unique_id 
from .schemas.job import JobStatus
from .schemas.api.model import PredModelCreationJob, PredModelBase, PredModel
from .mappings.proloaf import convert_pred_model_to_proloaf, convert_pred_model_to_api

logger = logging.getLogger("uvicorn.access")
logger.setLevel(logging.DEBUG)

def parse_create_model(json_message_body, job_db = None) -> PredModelCreationJob:
    job = PredModelCreationJob(**json_message_body)
    job.status = JobStatus.doing
    if job_db is not None:
        job_db.set(f"model_{job.job_id}",job.json())
    return job

def handle_create_model(basemodel:PredModelBase, model_db = None, model_object_db = None) -> PredModel:
    model = convert_pred_model_to_proloaf(PredModel(**basemodel.dict(), model_id = get_unique_id()))
    model.model.init_model()
    logger.debug(f"{model.model.initialized = }")
    if model_db is not None:
        api_model = convert_pred_model_to_api(model)
        model_db.set(f"model_{model.model_id}", api_model.json())
    if model_object_db is not None:
        model_object_db.set(f"model_{model.model_id}", pickle.dumps(model))
    return model
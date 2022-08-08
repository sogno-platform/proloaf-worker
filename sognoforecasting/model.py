import pickle
from .db import get_unique_id as get_unique_id 
from .schemas.job import JobStatus
from .schemas.model import PredModelCreationJob, PredModelBase, PredModel


def parse_create_model(json_message_body, job_db = None) -> PredModelCreationJob:
    job = PredModelCreationJob(**json_message_body)
    job.status = JobStatus.doing
    if job_db is not None:
        job_db.set(f"{job.job_id}",job.json())
    return job

def handle_create_model(basemodel:PredModelBase, model_db = None) -> PredModel:
    model = PredModel(**basemodel.get_config(), model_id = get_unique_id()) 
    model.model.init_model()
    print(f"{model.model.initialized = }")
    if model_db is not None:
        model_db.set(f"model_{model.model_id}", pickle.dumps(model))
    return model
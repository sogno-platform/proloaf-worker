import pickle
import json
from typing import ByteString
from proloaf.modelhandler import ModelWrapper
from .db import redis_job, redis_model, get_unique_id
from .schemas.job import JobStatus
from .schemas.model import PredModelCreationJob, PredModelBase, PredModel


def parse_create_model(msg_body:ByteString):
    job = PredModelCreationJob(**json.loads(msg_body))#, default=str))
    job.status = JobStatus.doing
    redis_job.set(f"model_{job.job_id}",job.json())
    return job

def handle_create_model(basemodel:PredModelBase):
    model_wrapper = ModelWrapper(**basemodel.model_definition.dict())
    model_wrapper.init_model()
    model = PredModel(**basemodel.dict(),model_id=get_unique_id(), model_object=model_wrapper)
    redis_model.set(f"model_{model.model_id}", pickle.dumps(model))
    return model
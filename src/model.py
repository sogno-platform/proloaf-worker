import pickle
import json
from typing import ByteString
from proloaf.modelhandler import ModelWrapper
from .db import redis_job, redis_model, get_unique_id
from .schemas.job import JobStatus
from .schemas.model import PredModelCreationJob, PredModelBase, PredModel


def parse_create_model(json_message_body) -> PredModelCreationJob:
    job = PredModelCreationJob(**json_message_body)#, default=str))
    job.status = JobStatus.doing
    # redis_job.set(f"model_{job.job_id}",job.json())
    return job

def handle_create_model(basemodel:PredModelBase) -> PredModel:
    basemodel.model.init_model()
    # model_wrapper.init_model()
    model = PredModel(**basemodel.get_config(),model_id=2)
    # redis_model.set(f"model_{model.model_id}", pickle.dumps(model))
    return model
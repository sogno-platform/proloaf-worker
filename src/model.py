import json
import pickle
import torch
from typing import ByteString
from proloaf.modelhandler import ModelWrapper
from .db import dummy_job_db as job_db # TODO replace with real database redis_job
from .db import dummy_model_db as model_db # TODO replace with real database redis_model
from .db import dummy_get_unique_id as get_unique_id # TODO get_unique_id
from .schemas.job import JobStatus
from .schemas.model import PredModelCreationJob, PredModelBase, PredModel


def parse_create_model(json_message_body) -> PredModelCreationJob:
    job = PredModelCreationJob(**json_message_body)
    job.status = JobStatus.doing
    job_db.set(f"{job.job_id}",job.get_config())
    return job

def handle_create_model(basemodel:PredModelBase) -> PredModel:
    model = PredModel(**basemodel.get_config(), model_id=2)
    model.model.init_model()
    compare = torch.load("model.pkl")
    torch.save(model,"model.pkl")
    print(f"{model.model.initialized = }")
    model_db.set(f"model_{model.model_id}", model)
    test = model_db.get(f"model_{model.model_id}")
    print(f"{test.model.initialized = }")
    return model
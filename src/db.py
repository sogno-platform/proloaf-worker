import os
import pickle
import torch
from redis import from_url as redis_from_url
from .settings import settings
import pathlib

redis_meta = redis_from_url(
    url=f"redis://{settings.redis_host}:{settings.redis_port}",
    username=settings.redis_username,
    password=settings.redis_password.get_secret_value(),
    db=0,
)

redis_job = redis_from_url(
    url=f"redis://{settings.redis_host}:{settings.redis_port}",
    username=settings.redis_username,
    password=settings.redis_password.get_secret_value(),
    db=1,
)

redis_model = redis_from_url(
    url=f"redis://{settings.redis_host}:{settings.redis_port}",
    username=settings.redis_username,
    password=settings.redis_password.get_secret_value(),
    db=2,
)


def get_unique_id():
    return redis_meta.incr("_next_unique_id")


dummy_next_id = 1


def dummy_get_unique_id():
    ret = dummy_next_id
    dummy_next_id += 1
    return ret


## Dummy DB (actuall files)
class DummyDB:
    def __init__(self, folder_path: str = "./data/"):
        self.folder_path = folder_path
        pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

    def get(self, name: str):
        # return torch.load(os.path.join(self.folder_path, name))
        with open(os.path.join(self.folder_path, name), "rb") as in_file:
            return pickle.load(in_file)

    def set(self, name: str, value: str):
        # return torch.save(value,os.path.join(self.folder_path, name))
        with open(os.path.join(self.folder_path, name), "wb") as out_file:
            pickle.dump(value, out_file)


dummy_meta_db = DummyDB("./data/meta/")
dummy_model_db = DummyDB("./data/model/")
dummy_job_db = DummyDB("./data/job/")

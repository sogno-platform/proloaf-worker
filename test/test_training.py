import pytest
from sognoforecasting.training import parse_run_training, handle_run_training
from sognoforecasting.model import handle_create_model
from sognoforecasting.schemas.job import JobStatus
from sognoforecasting.db import DummyDB
from .test_model import creation_job, model_creation_job_dict


@pytest.fixture
def training_job_dict():
    training_json = {
        "optimizer_name": "adam",
        "learning_rate": 9.9027931032814e-05,
        "max_epochs": 1,
        "early_stopping": {"patience": 7, "verbose": False, "delta": 0.0},
        "batch_size": 32,
        "history_horizon": 147,
        "forecast_horizon": 24,
    }
    training_base_json = {"data": {"id": 1}, "training": training_json, "model": 2}
    training_job_json = {"resource": training_base_json, "job_id": 2}

    return training_job_json


@pytest.fixture
def empty_model_db():
    return DummyDB("PYTEST_TMPDIR/model")

@pytest.fixture
def empty_job_db():
    return DummyDB("PYTEST_TMPDIR/job")

@pytest.fixture
def untrained_model_db(empty_model_db, creation_job):
    handle_create_model(creation_job.resource, empty_model_db)
    return empty_model_db

@pytest.fixture
def training_job(training_job_dict):
    return parse_run_training(training_job_dict)


def test_parse_run_training(training_job_dict, empty_job_db):
    job = parse_run_training(training_job_dict, empty_job_db)
    assert job.status == JobStatus.doing


def test_handle_run_training(training_job, untrained_model_db):
    pred_model = handle_run_training(training_job.resource, untrained_model_db)

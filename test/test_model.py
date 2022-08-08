import pytest
from sognoforecasting.model import parse_create_model, handle_create_model
from sognoforecasting.schemas.job import JobStatus
from sognoforecasting.db import DummyDB

@pytest.fixture
def model_creation_job_dict():
    model_wrapper_json = {
        "training": {
            # "optimizer_name": "adam",
            # "learning_rate": 9.9027931032814e-05,
            # "max_epochs": 1,
            # "early_stopping": {"patience": 7, "verbose": False, "delta": 0.0},
            # "batch_size": 32,
            # "history_horizon": 147,
            # "forecast_horizon": 24,
        },
        "model": {
            "rel_linear_hidden_size": 1.0,
            "rel_core_hidden_size": 1.0,
            "core_layers": 1,
            "dropout_fc": 0.4,
            "dropout_core": 0.3,
            "core_net": "torch.nn.LSTM",
            "relu_leak": 0.1,
        },
        "name": "opsd_recurrent",
        "target_id": ["DE_load_actual_entsoe_transparency"],
        "encoder_features": [
            "AT_load_actual_entsoe_transparency",
            "DE_load_actual_entsoe_transparency",
            "DE_temperature",
            "DE_radiation_direct_horizontal",
            "DE_radiation_diffuse_horizontal",
        ],
        "decoder_features": [
            "hour_0",
            "hour_1",
            "hour_2",
            "hour_3",
            "hour_4",
            "hour_5",
            "hour_6",
            "hour_7",
            "hour_8",
            "hour_9",
            "hour_10",
            "hour_11",
            "hour_12",
            "hour_13",
            "hour_14",
            "hour_15",
            "hour_16",
            "hour_17",
            "hour_18",
            "hour_19",
            "hour_20",
            "hour_21",
            "hour_22",
            "hour_23",
            "month_1",
            "month_2",
            "month_3",
            "month_4",
            "month_5",
            "month_6",
            "month_7",
            "month_8",
            "month_9",
            "month_10",
            "month_11",
            "month_12",
            "weekday_0",
            "weekday_1",
            "weekday_2",
            "weekday_3",
            "weekday_4",
            "weekday_5",
            "weekday_6",
            "hour_sin",
            # "weekday_sin",
            "mnth_sin",
        ],
        "metric": "NllGauss",
        "metric_options": {},
    }
    pred_model_base_json = {
        "name": "test_model",
        "model_type": "model1",
        "model": model_wrapper_json,
    }
    pred_model_creation_job_json = {
        "resource": pred_model_base_json,
        "job_id": 1,
        "status": "pending",
    }
    return pred_model_creation_job_json

@pytest.fixture
def creation_job(model_creation_job_dict):
    return parse_create_model(model_creation_job_dict)

@pytest.fixture
def empty_job_db():
    return DummyDB("PYTEST_TMPDIR/job")

def test_parse_create_model(model_creation_job_dict, empty_job_db):
    job = parse_create_model(model_creation_job_dict, empty_job_db)
    assert job.status == JobStatus.doing

def test_handle_create_model(creation_job,empty_job_db):
    pred_model = handle_create_model(creation_job.resource, empty_job_db)
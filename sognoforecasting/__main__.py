import datetime
import pickle
from typing import Any, ByteString, Callable, Dict, List, Tuple, Union
import asyncio
import aio_pika
import json
from sognoforecasting.schemas.model import PredModelCreationJob
from sognoforecasting.schemas.prediction import PredictionJob
from sognoforecasting.schemas.training import TrainingJob
from sognojq.amqp import AmqpListener
from sognoforecasting.model import parse_create_model, handle_create_model
from sognoforecasting.training import parse_run_training, handle_run_training
from sognoforecasting.prediction import parse_make_prediction, handle_make_prediction
from sognoforecasting.settings import settings
from sognoforecasting.schemas.job import Job, JobStatus
from sognoforecasting.db import redis_job as job_db  # TODO replace with real database redis_job
from sognoforecasting.db import (
    redis_model as model_db,
)  # TODO replace with real database redis_model
from sognoforecasting.db import get_unique_id as get_unique_id  # TODO get_unique_id


def finish_job(job: Job, result, success=True, job_db=None):
    job.result = result
    if success:
        job.status = JobStatus.success
    else:
        job.status = JobStatus.failed
    if job_db is not None:
        print(f"{job.json() = }")
        # XXX there should be a better way to distiguish between the jobtypes if jobs are in a full database
        if isinstance(job, PredictionJob):
            pre = "pred_"
        if isinstance(job, TrainingJob):
            pre = "train_"
        if isinstance(job, PredModelCreationJob):
            pre = "model_"
        job_db.set(f"{pre}{job.job_id}", job.json())


def handle_msg_faulty(msg: aio_pika.IncomingMessage):
    print(f"Message could not be parsed")
    pass


def parse_routing(key: str) -> Tuple[str]:
    print(f"Got message with key {key}")
    key_parts = key.split(".")
    job_type = key_parts[-2]
    job_id = key_parts[-1]
    return job_type, job_id


def handle_job(msg: aio_pika.IncomingMessage):
    job_type, job_id = parse_routing(msg.routing_key)
    try:
        json_msg_body = json.loads(msg.body)
    except json.JSONDecodeError:
        job = Job(job_id=job_id)
        finish_job(
            job, result="Message was not valid JSON", success=False, job_db=job_db
        )
        return False
    if job_type == "model":
        parse_func = parse_create_model
        handle_func = handle_create_model
    if job_type == "prediction":
        parse_func = parse_make_prediction
        handle_func = handle_make_prediction
    if job_type == "training":
        parse_func = parse_run_training
        handle_func = handle_run_training
    try:
        job = parse_func(json_msg_body, job_db=job_db)
    except (ValueError, TypeError):
        finish_job(
            job,
            result="Message was valid JSON, but could not be deserialized.",
            success=False,
            job_db=job_db,
        )
        return False
    try:
        result = handle_func(job.resource, model_db=model_db)
    except RuntimeError:
        finish_job(
            job,
            result=None,
            success=False,
            job_db=job_db,  # TODO result can not be a String atm "Job could not be processed."
        )
        return False
    finish_job(job, result, job_db=job_db)
    return True


async def main(test=False):
    if test:

        # Model creation
        model_wrapper_json = {
            "name": "string",
            "model_type": "model1",
            "model": {
                "training": {
                    "optimizer_name": "adam",
                    "learning_rate": 0.0001,
                    "max_epochs": 2,
                    "early_stopping": {"patience": 7, "verbose": False, "delta": 0},
                    "batch_size": 32,
                    "history_horizon": 24,
                    "forecast_horizon": 24,
                },
                "model": {
                    "rel_linear_hidden_size": 1,
                    "rel_core_hidden_size": 1,
                    "core_layers": 1,
                    "dropout_fc": 0,
                    "dropout_core": 0,
                    "core_net": "torch.nn.GRU",
                    "relu_leak": 0.01,
                },
                "name": "model",
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
                    "mnth_sin",
                ],
                "metric": "NllGauss",
                "metric_options": {},
            },
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

        creation_job = parse_create_model(pred_model_creation_job_json, job_db=job_db)
        result_model = handle_create_model(creation_job.resource, model_db=model_db)
        finish_job(creation_job, result_model)

        # Training
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
        training_job = parse_run_training(training_job_json, job_db=job_db)
        result_training = handle_run_training(training_job.resource, model_db=model_db)
        finish_job(training_job, result_training, job_db=job_db)

        # Prediction
        prediction_json = {
            "input_data": {"id": 1},
            "prediction_horizon": datetime.datetime.now(),
            "model": 2,
        }
        prediction_job_json = {"resource": prediction_json, "job_id": 3}
        prediction_job = parse_make_prediction(prediction_job_json, job_db=job_db)
        result_prediction = handle_make_prediction(
            prediction_job.resource, model_db=model_db
        )
        finish_job(prediction_job, result_prediction)
        print(f"{result_prediction.output_data.index = }")

    else:
        print("starting")
        amqp_listener = AmqpListener(
            amqp_host=settings.amqp_host,
            amqp_port=settings.amqp_port,
            amqp_username=settings.amqp_username,
            amqp_password=settings.amqp_password.get_secret_value(),
            amqp_queue_name=settings.amqp_queue,
        )
        print("binding to exchange")
        await amqp_listener.bind_to_exchange(
            exchange_name=settings.amqp_exchange, routing_key=settings.amqp_routing_key
        )
        print("listening")
        while True:
            msg = await amqp_listener.get_message()
            await msg.ack()
            try:
                handle_job(msg)
            except RuntimeError:
                handle_msg_faulty(msg)


asyncio.run(main())

import datetime
from typing import Any, ByteString, Callable, Dict
import asyncio
import aio_pika
import json
from sognojq.amqp import AmqpListener
from .model import parse_create_model, handle_create_model
from .training import parse_run_training, handle_run_training
from .prediction import parse_make_prediction, handle_make_prediction
from .settings import settings
from .schemas.job import Job, JobStatus
from .db import dummy_job_db as job_db  # TODO redis_job


def finish_job(job: Job, result, success=True):
    job.result = result
    if success:
        job.status = JobStatus.success
    else:
        job.status = JobStatus.failed
    job_db.set(str(job.job_id), job.get_config())


def handle_msg_faulty(msg: aio_pika.IncomingMessage):
    print(f"Message could not be parsed")
    pass


# TODO handle_func should not be Any to Any
def _handle_msg_generic(
    msg: aio_pika.IncomingMessage,
    parse_func: Callable[[ByteString], Job],
    handle_func: Callable[[Any], Any],
):
    try:
        json_msg_body = json.loads(msg.body)
        job = parse_func(json_msg_body)
        # TODO this should not be a generic exception
    except Exception as exc:
        finish_job(job, result="job description could not be parsed", success=False)
        msg.ack()
        return False
    msg.ack()
    try:
        result = handle_func(job.resource)
    # TODO this should not be a generic exception
    except Exception as exc:
        finish_job(
            job, result="job was parsed but failed during execution", success=False
        )
        return False
    finish_job(job, result=result)
    return True


def handle_msg(msg: aio_pika.IncomingMessage):
    key = msg.routing_key
    job_type = key.split(".")[1]
    if job_type == "model":
        return _handle_msg_generic(
            msg, parse_func=parse_create_model, handle_func=handle_create_model
        )
    if job_type == "prediciton":
        return _handle_msg_generic(
            msg, parse_func=parse_make_prediction, handle_func=handle_make_prediction
        )
    if job_type == "training":
        return _handle_msg_generic(
            msg, parse_func=parse_run_training, handle_func=handle_run_training
        )
    handle_msg_faulty(msg)
    return False


async def main(test=False):
    if test:

        # Model creation
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

        creation_job = parse_create_model(pred_model_creation_job_json)
        # result_model = handle_create_model(creation_job.resource)
        # finish_job(creation_job, result_model)

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
        training_job = parse_run_training(training_job_json)
        # result_training = handle_run_training(training_job.resource)
        # finish_job(training_job, result_training)

        # Prediction
        prediction_json = {
            "input_data": {"id": 1},
            "prediction_horizon": datetime.datetime.now(),
            "model": 2,
        }
        prediction_job_json = {"resource": prediction_json, "job_id": 3}
        prediction_job = parse_make_prediction(prediction_job_json)
        result_prediction = handle_make_prediction(prediction_job.resource)
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
            handle_msg(msg)


asyncio.run(main(True))

from typing import Any, ByteString, Callable, Dict
import asyncio
import aio_pika
from sognojq.amqp import AmqpListener
from .model import parse_create_model, handle_create_model
from .training import parse_run_training, handle_run_training
from .prediction import parse_make_prediction, handle_make_prediction
from .settings import settings
from .schemas.job import Job, JobStatus
from .db import redis_job


def finish_job(job: Job, result, success=True):
    Job.result = result
    if success:
        Job.status = JobStatus.success
    redis_job.set(str(Job.job_id), Job.json())


def handle_msg_faulty(msg: aio_pika.IncomingMessage):
    pass


# TODO handle_func should not be Any to Any
def _handle_msg_generic(
    msg: aio_pika.IncomingMessage,
    parse_func: Callable[[ByteString], Job],
    handle_func: Callable[[Any], Any],
):
    try:
        job = parse_func(msg.body)
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
        return _handle_msg_generic(msg,parse_func=parse_create_model,handle_func=handle_create_model)
    if job_type == "prediciton":
        return _handle_msg_generic(msg,parse_func=parse_make_prediction,handle_func=handle_make_prediction)
    if job_type == "training":
        return _handle_msg_generic(msg,parse_func=parse_run_training,handle_func=handle_run_training)
    handle_msg_faulty(msg)
    return False


async def main():
    amqp_listener = AmqpListener(
        amqp_username=settings.amqp_username,
        amqp_password=settings.amqp_password.get_secret_value(),
        amqp_queue_name=settings.amqp_queue,
    )
    await amqp_listener.bind_to_exchange(
        exchange_name=settings.amqp_exchange, routing_key=settings.amqp_routing_key
    )
    while True:
        msg = await amqp_listener.get_message()

asyncio.run(main())
import torch
from typing import Any, ByteString, Dict
import pandas as pd
import pickle
from .schemas.job import JobStatus
from .schemas.training import TrainingJob, TrainingBase, TrainingResult
from .schemas.model import PredModel
from .db import redis_job, redis_model, get_unique_id
from proloaf import tensorloader as tl
from proloaf import datahandler as dh
from proloaf.modelhandler import ModelWrapper


def parse_run_training(json_message_body):
    job = TrainingJob(**json_message_body)  # , default=str))
    job.status = JobStatus.doing
    # redis_job.set(f"model_{job.job_id}", job.json())
    return job


def handle_run_training(basetraining: TrainingBase):
    training_def = basetraining.training
    if isinstance(basetraining.model, int):
        # pyd_model = pickle.loads(redis_model.get(f"model_{basetraining.model}"))
        pyd_model = torch.load("model.pkl")
        # pyd_model.reinitialize()
        print("")
        print(pyd_model)
        model = pyd_model.model
    else:
        model = ModelWrapper(
            **basetraining.model.model_definition.dict(),
            early_stopping_margin=training_def.early_stopping_margin,
            early_stopping_patience=training_def.early_stopping_patience,
            max_epochs=training_def.max_epochs,
            learning_rate=training_def.learning_rate,
            optimizer_name=training_def.optimizer_name,
            # XXX horizons might end up here in the end depending on proloaf chnges
        )
        model.init_model()
        pyd_model = PredModel(
            **basetraining.model.dict(),
            model=model,
            model_id=1,  # TODO model_id=get_unique_id()
        )
    pyd_model.model.training = training_def
    df = pd.read_csv("proloaf-worker/opsd.csv", sep=";")
    df_train, df_val = dh.split(df, [0.9])
    # TODO add selection of source
    train_data = tl.TimeSeriesData(df_train, **training_def.dict())
    val_data = tl.TimeSeriesData(df_val, **training_def.dict())
    print(model.dict())
    model.run_training(
        train_data=train_data,
        validation_data=val_data,
        batch_size=training_def.batch_size,
    )

    pyd_model.date_trained = model.training.training_start_time
    pyd_model.predicted_feature = model.target_id[0]
    # TODO add expected dataformat to pyd_model
    # TODO redis_model.set(f"model_{pyd_model.model_id}", pickle.dumps(pyd_model))

    dt = model.training.training_end_time - model.training.training_start_time
    return TrainingResult(
        training=model.training.get_config(),
        model=pyd_model,
        data=basetraining.data, # TODO data should be selfdocumenting
        actual_training_time=dt,
        validation_error=model.training.validation_loss,
        training_error=model.training.training_loss,
    )

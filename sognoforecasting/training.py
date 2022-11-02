from datetime import datetime
import logging
import pandas as pd
import pickle
from .schemas.job import JobStatus
from .schemas.api.training import TrainingJob, TrainingResult
from .schemas.api.training import TrainingBase as ApiTrainingBase
from .mappings.proloaf import convert_pred_model_to_api, convert_training_to_api, convert_training_to_proloaf
from .schemas.proloaf.model import PredModel
from .db import get_unique_id as get_unique_id
from proloaf import tensorloader as tl
from proloaf import datahandler as dh
from proloaf.modelhandler import ModelWrapper

logger = logging.getLogger("sogno.forecasting.worker")

def parse_run_training(json_message_body, job_db = None):
    job = TrainingJob(**json_message_body)  # , default=str))
    job.status = JobStatus.doing
    if job_db is not None:
        job_db.set(f"train_{job.job_id}", job.json())
    return job


def handle_run_training(basetraining: ApiTrainingBase, model_db = None, model_object_db = None):
    training_def = convert_training_to_proloaf(basetraining.training)
    if isinstance(basetraining.model, int):
        logger.info("loading model for training")
        try:
            pyd_model: PredModel = pickle.loads(model_object_db.get(f"model_{basetraining.model}"))
            # pyd_model = torch.load("model.pkl")
            # pyd_model.reinitialize()
            model = pyd_model.model
        except TypeError as exc:
            raise TypeError(f"model with id {basetraining.model} could not be loaded.") from exc
    else:
        model = ModelWrapper(
            **basetraining.model.model_definition.dict(),
            early_stopping_margin=training_def.early_stopping_margin,
            early_stopping_patience=training_def.early_stopping_patience,
            max_epochs=training_def.max_epochs,
            learning_rate=training_def.learning_rate,
            optimizer_name=training_def.optimizer_name,
            # XXX horizons might end up here in the end depending on proloaf changes
        )
        model.init_model()
        pyd_model = PredModel(
            **basetraining.model.dict(),
            model=model,
            model_id=get_unique_id()
        )
    pyd_model.model.training = training_def
    df = pd.read_csv("./opsd.csv", sep=";") #./proloaf-worker/
    df_train, df_val = dh.split(df, [0.9])
    # TODO add selection of source
    train_data = tl.TimeSeriesData(df_train, **training_def.dict())
    val_data = tl.TimeSeriesData(df_val, **training_def.dict())
    
    logger.info("start training")
    model.run_training(
        train_data=train_data,
        validation_data=val_data,
        batch_size=training_def.batch_size,
    )
    logger.info("done training")

    pyd_model.date_trained = datetime.utcnow() # TODO This is a perf_counter proloaf
    pyd_model.predicted_feature = model.target_id[0]
    # TODO add expected dataformat to pyd_model
    if model_db is not None:
        logger.info(f"saving model metadata in model_db as model_{pyd_model.model_id}")
        model_db.set(f"model_{pyd_model.model_id}", pyd_model.json())

    if model_object_db is not None:
        logger.info(f"saving model metadata in model_object_db as model_{pyd_model.model_id}")
        model_object_db.set(f"model_{pyd_model.model_id}", pickle.dumps(pyd_model))



    dt = model.training.training_end_time - model.training.training_start_time
    return TrainingResult(
        training=convert_training_to_api(model.training),
        model=convert_pred_model_to_api(pyd_model),
        data=basetraining.data,  # TODO data should be selfdocumenting
        actual_training_time=dt,
        validation_error=model.training.validation_loss,
        training_error=model.training.training_loss,
    )

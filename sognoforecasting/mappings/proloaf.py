from typing import Any, Dict
import logging
from sognoforecasting.schemas.api.proloaf_api_models import ModelWrapper as ApiModel
from sognoforecasting.schemas.api.proloaf_api_models import TrainingRun as ApiTraining
from sognoforecasting.schemas.api.proloaf_api_models import (
    EncoderDecoder as ApiEncoderDecoder,
)
from sognoforecasting.schemas.api.model import PredModel as ApiPredModel

logger = logging.getLogger("sogno.forecasting.worker")

try:
    from proloaf.modelhandler import ModelWrapper, TrainingRun
    from proloaf.models import EncoderDecoder
    from sognoforecasting.schemas.proloaf.model import PredModel

    Model = ModelWrapper
    proloaf_available = True
except ImportError as err:
    logger.exception("proloaf could not be loaded, proloaf classes will not be available.")
    proloaf_available = False

if proloaf_available:

    def convert_training_to_proloaf(training: ApiTraining) -> TrainingRun:
        if training is None:
            return None
        logger.debug(f"{training = }")
        return TrainingRun(
            optimizer_name=training.optimizer_name,
            learning_rate=training.learning_rate,
            max_epochs=training.max_epochs,
            early_stopping={"patience": training.patience, "delta": training.delta},
            batch_size=training.batch_size,
            history_horizon=training.history_horizon,
            forecast_horizon=training.forecast_horizon,
        )

    def convert_training_to_api(training: TrainingRun) -> ApiTraining:
        if training is None:
            return None
        return ApiTraining(
            optimizer_name=training.optimizer_name,
            learning_rate=training.learning_rate,
            max_epochs=training.max_epochs,
            # early_stopping=training.early_stopping,
            batch_size=training.batch_size,
            history_horizon=training.history_horizon,
            forecast_horizon=training.forecast_horizon,
            patience=training.early_stopping.patience,
            delta=training.early_stopping.delta,
        )

    def convert_encoder_decoder_to_proloaf(model: ApiEncoderDecoder) -> Dict[str, Any]:
        if model is None:
            return None

        return dict(
            rel_linear_hidden_size=model.rel_linear_hidden_size,
            rel_core_hidden_size=model.rel_core_hidden_size,
            core_layers=model.core_layers,
            dropout_fc=model.dropout_fc,
            dropout_core=model.dropout_core,
            core_net=model.core_net,
            relu_leak=model.relu_leak,
        )

    def convert_encoder_decoder_to_api(model: EncoderDecoder) -> Dict[str,Any]:
        if model is None:
            return None

        return dict(
            rel_linear_hidden_size=model.rel_linear_hidden_size,
            rel_core_hidden_size=model.rel_core_hidden_size,
            core_layers=model.core_layers,
            dropout_fc=model.dropout_fc,
            dropout_core=model.dropout_core,
            core_net=model.core_net,
            relu_leak=model.relu_leak,
        )

    def convert_model_wrapper_to_proloaf(model_def: ApiModel) -> ModelWrapper:
        if model_def is None:
            return None
        return ModelWrapper(
            training=convert_training_to_proloaf(model_def.training),
            model=convert_encoder_decoder_to_proloaf(model_def.model),
            name=model_def.name,
            target_id=model_def.target_id,
            encoder_features=model_def.encoder_features,
            decoder_features=model_def.decoder_features,
            metric=model_def.metric,
            metric_options=model_def.metric_options,
        )

    def convert_model_wrapper_to_api(model_def: ModelWrapper) -> ApiModel:
        if model_def is None:
            return None
        return ModelWrapper(
            training=convert_training_to_api(model_def.training),
            model=convert_encoder_decoder_to_api(model_def.model),
            name=model_def.name,
            target_id=model_def.target_id,
            encoder_features=model_def.encoder_features,
            decoder_features=model_def.decoder_features,
            metric=model_def.metric,
            metric_options=model_def.metric_options,
        )

    def convert_pred_model_to_proloaf(pred_model: ApiPredModel) -> PredModel:
        if pred_model is None:
            return None
        return PredModel(
            name=pred_model.name,
            model_type=pred_model.model_type,
            model=convert_model_wrapper_to_proloaf(pred_model.model),
            model_id=pred_model.model_id,
            date_trained=pred_model.date_trained,
            date_hyperparameter_tuned=pred_model.date_hyperparameter_tuned,
            predicted_feature=pred_model.predicted_feature,
            expected_data_format=pred_model.expected_data_format,
        )

    def convert_pred_model_to_api(pred_model: PredModel) -> ApiPredModel:
        if pred_model is None:
            return None
        return ApiPredModel(
            name=pred_model.name,
            model_type=pred_model.model_type,
            model=convert_model_wrapper_to_api(pred_model.model),
            model_id=pred_model.model_id,
            date_trained=pred_model.date_trained,
            date_hyperparameter_tuned=pred_model.date_hyperparameter_tuned,
            predicted_feature=pred_model.predicted_feature,
            expected_data_format=pred_model.expected_data_format,
        )

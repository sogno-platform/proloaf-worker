from random import triangular


try:
    from proloaf.modelhandler import ModelWrapper, TrainingRun
    from proloaf.models import EncoderDecoder
    from openstefapi.app.schemas.v2.model import Model

    Model = ModelWrapper
except ImportError as err:
    print("proloaf could not be loaded, proloaf classes will not be available.")
    proloaf = None

if proloaf is not None:

    def convert_training(training) -> TrainingRun:
        return TrainingRun(
            optimizer_name=training.optimizer_name,
            learning_rate=training.learning_rate,
            max_epochs=training.max_epochs,
            early_stopping=training.early_stopping,
            batch_size=training.batch_size,
            history_horizon=training.history_horizon,
            forecast_horizon=training.forecast_horizon,
        )

    def convert_EncoderDecoder(model) -> ModelWrapper:
        return EncoderDecoder(
            rel_linear_hidden_size=model.rel_linear_hidden_size,
            rel_core_hidden_size=model.rel_core_hidden_size,
            core_layers=model.core_layers,
            dropout_fc=model.dropout_fc,
            dropout_core=model.dropout_core,
            core_net=model.core_net,
            relu_leak=model.relu_leak,
        )

    def convert_model_wrapper(model_def: Model) -> ModelWrapper:
        plf_model = ModelWrapper(
            training=convert_training(model_def.training),
            model=convert_EncoderDecoder(model_def.model),
            name=model_def.name,
            target_id=model_def.target_id,
            encoder_features=model_def.encoder_features,
            decoder_features=model_def.decoder_features,
            metric=model_def.metric,
            metric_options=model_def.metric_options
        )
        return plf_model

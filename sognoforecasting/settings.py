from pydantic import BaseSettings, SecretStr

class Settings(BaseSettings):
    # Redis settings
    redis_host:str = "172.17.0.1"
    redis_port:int = 30415
    redis_username = "default"
    redis_password:SecretStr = "testpw"

    # AMQP settings
    amqp_host:str = "172.17.0.1"
    amqp_port:int = 30305
    amqp_username:str = "user"
    amqp_password:SecretStr = "testpw"
    amqp_exchange:str = "forecastingjobs"
    amqp_queue:str = "plf_job_queue"
    amqp_routing_key: str = "forecasting.#"

    logging_level = "debug"

settings = Settings()
from pydantic import BaseSettings, SecretStr

class Settings(BaseSettings):
    # Redis settings
    redis_host:str = "localhost"
    redis_port:int = 6379
    redis_username = "default"
    redis_password:SecretStr = "testpw"

    # AMQP settings
    amqp_host:str = "localhost"
    amqp_port:int = 5672
    amqp_username:str = "user"
    amqp_password:SecretStr = "testpw"
    amqp_exchange:str = "forecastingjobs"
    amqp_queue:str = "plf_job_queue"
    amqp_routing_key: str = "forecasting.#"

settings = Settings()
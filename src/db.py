from redis import from_url as redis_from_url
from .settings import settings


redis_meta = redis_from_url(
    url=f"redis://{settings.redis_host}:{settings.redis_port}",
    username=settings.redis_username,
    password=settings.redis_password.get_secret_value(),
    db=0,
)

redis_job = redis_from_url(
    url=f"redis://{settings.redis_host}:{settings.redis_port}",
    username=settings.redis_username,
    password=settings.redis_password.get_secret_value(),
    db=1,
)

redis_model = redis_from_url(
    url=f"redis://{settings.redis_host}:{settings.redis_port}",
    username=settings.redis_username,
    password=settings.redis_password.get_secret_value(),
    db=2,
)


def get_unique_id():
    return redis_meta.incr("_next_unique_id")

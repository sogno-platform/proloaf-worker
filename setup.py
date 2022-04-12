from setuptools import setup

setup(
    name= "sogno-forcasting-worker",
    version="0.1",
    description="Classes for interfacing with AMQP",
    author="Florian Oppermann",
    url="https://github.com/sogno-platform/sogno-job-queue",
    packages=[""],
    package_dir={'':'src'},
    license="Apache-2.0 License",
    python_requires=">=3.8",
    # install_requires=["aio_pika >= 7.0"] # TODO
)